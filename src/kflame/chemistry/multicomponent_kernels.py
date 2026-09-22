"""Compiled Dixon--Lewis assembly and exact block elimination.

Formulation adapted from Cantera 3.2 MultiTransport.cpp (BSD-3-Clause).
See CANTERA_TRANSPORT_LICENSE.txt.
"""
import math
import numpy as np
from kflame.chemistry.mechanism import R_UNIV

try:
    from numba import njit, prange
except ImportError:
    def njit(**kwargs):
        return lambda function: function
    prange = range

_MIN_C_INTERNAL = 1e-3


@njit(cache=True, parallel=True)
def evaluate_faces(temperature, pressure, y, cp_r, mw, inv_w, log_eps_pair,
                   epsilon, crot, zrot, frot298, viscosity_fit, diffusion_fit,
                   astar_fit, bstar_fit, cstar_fit, dense):
    """Independent face systems with one BLAS thread per Numba worker."""
    faces = temperature.size
    n = mw.size
    rho = np.empty(faces)
    conductivity = np.empty(faces)
    wmix = np.empty(faces)
    multi = np.empty((faces, n, n))
    thermal = np.empty((faces, n))
    for face in prange(faces):
        t = temperature[face]
        logt = math.log(t)
        sqrt_t = math.sqrt(t)
        denom = 0.0
        for k in range(n):
            denom += y[k, face] * inv_w[k]
        mix_weight = 1.0 / max(denom, 1e-300)
        x = np.empty(n)
        mu = np.empty(n)
        rot_relax = np.empty(n)
        bdiff = np.empty((n, n))
        a = np.empty((n, n))
        b = np.empty((n, n))
        c = np.empty((n, n))
        for k in range(n):
            x[k] = max(y[k, face] * mix_weight * inv_w[k], 1e-20)
            value = viscosity_fit[k, 4]
            for degree in range(3, -1, -1):
                value = value * logt + viscosity_fit[k, degree]
            mu[k] = sqrt_t * value * value
            tr = epsilon[k] / t
            sqrt_tr = math.sqrt(tr)
            frot = (1.0 + 0.5 * math.pi ** 1.5 * sqrt_tr
                    + (0.25 * math.pi ** 2 + 2.0) * tr
                    + math.pi ** 1.5 * sqrt_tr * tr)
            rot_relax[k] = max(1.0, zrot[k]) * frot298[k] / frot
            for j in range(n):
                z = logt - log_eps_pair[k, j]
                aa, bb, cc = astar_fit[k, j, 8], bstar_fit[k, j, 8], cstar_fit[k, j, 8]
                for degree in range(7, -1, -1):
                    aa = aa * z + astar_fit[k, j, degree]
                    bb = bb * z + bstar_fit[k, j, degree]
                    cc = cc * z + cstar_fit[k, j, degree]
                a[k, j], b[k, j], c[k, j] = aa, bb, cc
                value = diffusion_fit[k, j, 4]
                for degree in range(3, -1, -1):
                    value = value * logt + diffusion_fit[k, j, degree]
                bdiff[k, j] = t * sqrt_t * value
            bdiff[k, k] = 1.2 * R_UNIV * t * mu[k] * a[k, k] / mw[k]
        l00 = np.empty((n, n))
        for k in range(n):
            summed = 0.0
            for j in range(n):
                if k != j:
                    summed += x[j] / bdiff[k, j]
            summed /= mw[k]
            for j in range(n):
                l00[k, j] = 16.0 * t / 25.0 * x[j] * (mw[j] * summed + x[k] / bdiff[k, j]) if k != j else 0.0
        inverse = np.linalg.solve(l00, np.eye(n))
        dt, lam = thermal_system(t, x, np.ascontiguousarray(cp_r[:, face]), bdiff,
                                 a, b, c, rot_relax, mw, crot, mu, l00, inverse, dense)
        prefactor = 16.0 * t * mix_weight / (25.0 * pressure)
        for k in range(n):
            thermal[face, k] = dt[k]
            for j in range(n):
                multi[face, k, j] = prefactor * x[k] / mw[j] * (inverse[k, j] - inverse[k, k])
        rho[face] = pressure * mix_weight / (R_UNIV * t)
        conductivity[face] = lam
        wmix[face] = mix_weight
    return rho, conductivity, wmix, multi.transpose(1, 2, 0), thermal.T


@njit(cache=True)
def thermal_system(temperature, x, cp_r, bdiff, astar, bstar, cstar,
                   rot_relax, mw, crot, viscosity, l00, l00_inverse, dense):
    n = x.size
    l = np.zeros((3 * n, 3 * n), dtype=np.float64)
    l[:n, :n] = l00

    cinternal = np.asarray(cp_r, dtype=np.float64) - 2.5
    has_internal = cinternal > _MIN_C_INTERNAL
    mw_i = mw[:, None]
    mw_j = mw[None, :]
    x_i = x[:, None]
    x_j = x[None, :]

    # L00,10 and its transpose L10,00
    l01 = (
        -1.6
        * float(temperature)
        * x_i
        * x_j
        * mw_i
        * (1.2 * cstar.T - 1.0)
        / np.maximum((mw_j + mw_i) * bdiff.T, 1.0e-300)
    )
    sums01 = np.sum(l01, axis=0)
    for k in range(n):
        l01[k, k] -= sums01[k]
    l[:n, n : 2 * n] = l01
    l[n : 2 * n, :n] = l01.T

    # L10,10
    five_over_3pi = 5.0 / (3.0 * math.pi)
    # All N^2 coefficients are formed in array operations. The 3N solve
    # remains dense LAPACK work, but removing Python-level i/j loops is
    # material for the small blocks that occur at every flame face.
    wi = mw[:, None]
    wj = mw[None, :]
    xi = x[:, None]
    term1 = bdiff * (wi + wj) ** 2
    term2 = 4.0 * wj * astar * (
        1.0
        + five_over_3pi
        * (
            (crot / rot_relax)[None, :]
            + (crot / rot_relax)[:, None]
        )
    )
    constant1 = 16.0 * float(temperature) * x / 25.0
    wj2 = wj * wj
    l11 = (
        constant1[None, :]
        * xi
        * wi
        / (wj * term1)
        * (13.75 * wj2 - 3.0 * wj2 * bstar - term2 * wj)
    )
    diag11 = constant1 * np.sum(
        xi
        / term1
        * (
            7.5 * wj2
            + wi * wi * (6.25 - 3.0 * bstar)
            + term2 * wi
        ),
        axis=0,
    )
    for k in range(n):
        l11[k, k] -= diag11[k]
    l[n : 2 * n, n : 2 * n] = l11

    # L10,01, its transpose L01,10, and the internal-mode block.
    l12 = np.zeros((n, n), dtype=np.float64)
    internal_constant = np.zeros(n, dtype=np.float64)
    internal_constant[has_internal] = (
        32.0
        * float(temperature)
        * mw[has_internal]
        * x[has_internal]
        * crot[has_internal]
        / (
            5.0
            * math.pi
            * cinternal[has_internal]
            * rot_relax[has_internal]
        )
    )
    l12[:, :] = (
        xi
        * astar.T
        * internal_constant[None, :]
        / np.maximum((wi + wj) * bdiff.T, 1.0e-300)
    )
    sums12 = np.sum(l12, axis=0)
    for k in range(n):
        l12[k, k] += sums12[k]
    l[n : 2 * n, 2 * n :] = l12
    l[2 * n :, n : 2 * n] = l12.T

    l22 = np.zeros((n, n), dtype=np.float64)
    inv_bdiff = 1.0 / np.maximum(bdiff, 1.0e-300)

    constant1 = np.zeros(n, dtype=np.float64)
    constant2 = np.zeros(n, dtype=np.float64)
    constant1[has_internal] = (
        4.0 * float(temperature) * x[has_internal] / cinternal[has_internal]
    )
    constant2[has_internal] = (
        12.0
        * mw[has_internal]
        * crot[has_internal]
        / (
            5.0
            * math.pi
            * cinternal[has_internal]
            * rot_relax[has_internal]
        )
    )
    off_diagonal = np.ones((n, n), dtype=np.float64) - np.eye(n)
    total = inv_bdiff @ x + constant2 * np.sum(
        off_diagonal * astar * inv_bdiff * (x / mw)[None, :], axis=1
    )
    diagonal = np.ones(n, dtype=np.float64)
    diagonal[has_internal] = (
        -8.0
        / math.pi
        * mw[has_internal]
        * x[has_internal] ** 2
        * crot[has_internal]
        / (
            cinternal[has_internal] ** 2
            * R_UNIV
            * viscosity[has_internal]
            * rot_relax[has_internal]
        )
        - constant1[has_internal] * total[has_internal]
    )
    for k in range(n):
        l22[k, k] = diagonal[k]
    l[2 * n :, 2 * n :] = l22

    rhs = np.concatenate((np.zeros(n), x, np.where(has_internal, x, 0.0)))
    if dense:
        solution = np.linalg.solve(l, rhs)
    else:
        # Exact block elimination: E=L22 is diagonal and A=L00 has
        # already been inverted for the ordinary diffusion coefficients.
        # The remaining system is N by N, rather than 3N by 3N.
        if l00_inverse is None:
            l00_inverse = np.linalg.solve(l[:n, :n], np.eye(n))
        # ``l00_inverse`` can be an arbitrary-strided view at this boundary.
        # A contiguous copy avoids a slower generic BLAS path without changing
        # the values or the block-elimination arithmetic.
        ainv_b = np.ascontiguousarray(l00_inverse) @ l01
        d_einv = l12 / diagonal[None, :]
        schur = l11 - l01.T @ ainv_b - d_einv @ l12.T
        energy_rhs = x - d_einv @ rhs[2 * n:]
        energy = np.linalg.solve(schur, energy_rhs)
        solution = np.concatenate((
            -ainv_b @ energy,
            energy,
            (rhs[2 * n:] - l12.T @ energy) / diagonal,
        ))
    dthermal = 1.6 / R_UNIV * mw * x * solution[:n]
    # ``MultiTransport::thermalConductivity`` sums the two energetic
    # blocks (indices n through 3n), not only the translational/rotational
    # one.  Species without an internal mode have a zero third RHS entry.
    conductivity = -4.0 * float(
        np.dot(x, solution[n : 2 * n])
        + np.dot(np.where(has_internal, x, 0.0), solution[2 * n :])
    )
    return dthermal, conductivity
