"""Monchick--Mason collision integrals and transport fits without Cantera.

The tabulated data and fit conventions follow Cantera 3.2 (BSD-3-Clause);
see collision_integrals_mm.json and CANTERA_TRANSPORT_LICENSE.txt. These are
universal molecular collision data, not flame solutions or fitted Soret outputs.
"""
from __future__ import annotations

import json
import hashlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval

from kava.chemistry.mechanism import AVOGADRO, BOLTZMANN, R_UNIV
from kava.chemistry.thermo import NativeThermo


class CollisionIntegrals:
    def __init__(self, minimum: float, maximum: float):
        with (Path(__file__).parent / 'data' / 'collision_integrals_mm.json').open(encoding="utf-8") as stream:
            data = json.load(stream)
        self.temperatures = np.asarray(data["tstar22"])
        self.log_temperature = np.log(self.temperatures)
        delta = np.asarray(data["delta"])
        self.tables = np.stack([
            np.asarray(data[key]).reshape(-1, 8)[1:-1]
            if key != "omega22_table" else np.asarray(data[key]).reshape(37, 8)
            for key in ("omega22_table", "astar_table", "bstar_table", "cstar_table")
        ])
        self.delta_polynomials = polyfit(delta, self.tables.reshape(-1, 8).T, 6).T.reshape(4, 37, 7)
        lower = np.flatnonzero(minimum > self.temperatures)
        upper = np.flatnonzero(maximum > self.temperatures)
        nmin = int(lower[-1]) if lower.size else -1
        nmax = int(upper[-1] + 1) if upper.size else -1
        if nmin < 0 or nmin >= 36 or nmax < 0 or nmax > 36:
            nmin, nmax = 0, 36
        self.fit_slice = slice(nmin, nmax + 1)

    def values(self, delta: float) -> np.ndarray:
        if delta == 0.0:
            return self.tables[:, :, 0]
        return polyval(float(delta), self.delta_polynomials.transpose(2, 0, 1))

    def fits(self, delta: float) -> np.ndarray:
        region = self.fit_slice
        return polyfit(self.log_temperature[region], self.values(delta)[:, region].T, 8).T

    def evaluate(self, temperature: np.ndarray, delta: float) -> np.ndarray:
        """Local quadratic interpolation in log reduced temperature."""
        temperature = np.asarray(temperature, dtype=float)
        left = np.clip(np.searchsorted(self.temperatures, temperature, side="right") - 1, 0, 33)
        indices = left[..., None] + np.arange(3)
        nodes = self.log_temperature[indices]
        target = np.log(temperature)
        weights = np.empty_like(nodes)
        for k in range(3):
            others = [i for i in range(3) if i != k]
            weights[..., k] = ((target - nodes[..., others[0]]) * (target - nodes[..., others[1]])
                               / ((nodes[..., k] - nodes[..., others[0]]) * (nodes[..., k] - nodes[..., others[1]])))
        return np.sum(self.values(delta)[:, indices] * weights, axis=-1)


_FIT_CACHE = OrderedDict()
_CONDUCTIVITY_CACHE = OrderedDict()


def _array_key(values):
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value, dtype=np.float64)
        digest.update(str(array.shape).encode('ascii'))
        digest.update(array.tobytes())
    return digest.digest()


def native_conductivity_fits(mech, base):
    """Fit pure-species conductivity from NASA and molecular collision data.

    Parker rotational relaxation and internal-mode energy transport follow the
    gas-transport formulation documented in GasTransport.cpp (BSD-3-Clause;
    see CANTERA_TRANSPORT_LICENSE.txt). No exported property coefficients.
    The cache includes thermodynamics as well as every molecular input used.
    """
    key = _array_key((mech.min_temperature, mech.max_temperature,
                      mech.molecular_weights, mech.well_depth, mech.diameter,
                      mech.geometry, mech.rot_relax, mech.nasa_low,
                      mech.nasa_high, mech.nasa_Tmid, base._eps_pair,
                      base._delta_pair, base._reduced_mass))
    if key in _CONDUCTIVITY_CACHE:
        _CONDUCTIVITY_CACHE.move_to_end(key)
        return _CONDUCTIVITY_CACHE[key]
    tmin, tmax = mech.min_temperature, mech.max_temperature
    if not 0 < tmin < tmax:
        raise ValueError('No common positive temperature interval for conductivity fitting.')
    t = np.linspace(tmin, tmax, 50)
    logt, sqrt_t = np.log(t), np.sqrt(t)
    cp_r = NativeThermo(mech).cp_R(t)
    eps = np.asarray(base._eps_pair)
    integrals = CollisionIntegrals(float(np.min(tmin / eps)), float(np.max(tmax / eps)))
    coefficients = np.empty((mech.n_species, 5))

    def rotation_factor(tstar):
        return (1.0 + np.pi ** 1.5 / np.sqrt(tstar) * (0.5 + 1.0 / tstar)
                + (0.25 * np.pi ** 2 + 2.0) / tstar)

    for k in range(mech.n_species):
        tstar = t / mech.well_depth[k]
        collision = integrals.evaluate(tstar, float(base._delta_pair[k, k]))
        omega22, omega11 = collision[0], collision[0] / collision[1]
        sigma, weight = mech.diameter[k], mech.molecular_weights[k]
        mu = (5.0 / 16.0 * np.sqrt(np.pi * weight * BOLTZMANN * t / (1000.0 * AVOGADRO))
              / (omega22 * np.pi * sigma ** 2))
        self_diffusion = (3.0 / 16.0 * np.sqrt(2.0 * np.pi / base._reduced_mass[k, k])
                          * (BOLTZMANN * t) ** 1.5 / (np.pi * sigma ** 2 * omega11))
        fint = weight / (R_UNIV * t) * self_diffusion / mu
        crot = 0.0 if mech.geometry[k] == 0 else (1.0 if mech.geometry[k] == 1 else 1.5)
        relax = (mech.rot_relax[k] * rotation_factor(298.0 / mech.well_depth[k])
                 / rotation_factor(tstar))
        correction = 2.0 / np.pi * (2.5 - fint) / (relax + 2.0 / np.pi * (5.0 / 3.0 * crot + fint))
        conductivity = mu / weight * R_UNIV * (
            2.5 * (1.0 - correction * crot / 1.5) * 1.5
            + fint * (1.0 + correction) * crot + fint * (cp_r[k] - 2.5 - crot))
        target = conductivity / sqrt_t
        if not np.all(np.isfinite(target)) or np.any(target <= 0.0):
            raise ValueError(f'Nonpositive conductivity fit data for {mech.species_names[k]}')
        coefficients[k] = polyfit(logt, target, 4, w=1.0 / target)
    coefficients.setflags(write=False)
    _CONDUCTIVITY_CACHE[key] = coefficients
    if len(_CONDUCTIVITY_CACHE) > 8:
        _CONDUCTIVITY_CACHE.popitem(last=False)
    return coefficients


def native_transport_fits(mech, base):
    """Reuse immutable molecular fits across meshes and flames (eight entries).

    Key all numerical inputs, not a filename or species count. Pressure and
    composition enter evaluation, not the pressure-independent polynomial fits.
    """
    digest = hashlib.sha256()
    for value in (mech.min_temperature, mech.max_temperature, mech.molecular_weights,
                  mech.well_depth, mech.diameter, base._eps_pair,
                  base._sigma_pair, base._delta_pair):
        array = np.ascontiguousarray(value, dtype=np.float64)
        digest.update(str(array.shape).encode('ascii'))
        digest.update(array.tobytes())
    key = digest.digest()
    if key in _FIT_CACHE:
        _FIT_CACHE.move_to_end(key)
        return _FIT_CACHE[key]
    fits = _build_transport_fits(mech, base)
    for array in fits:
        array.setflags(write=False)
    _FIT_CACHE[key] = fits
    if len(_FIT_CACHE) > 8:
        _FIT_CACHE.popitem(last=False)
    return fits


def _build_transport_fits(mech, base):
    """Fit viscosity, binary diffusion and A*, B*, C* from molecular data.

    Fits use the common NASA temperature interval. Relative least squares is
    evaluated natively; no pre-exported, mechanism-indexed polynomial is read.
    """
    eps = np.asarray(base._eps_pair)
    sigma = np.asarray(base._sigma_pair)
    delta = np.asarray(base._delta_pair)
    if np.any(eps <= 0) or np.any(sigma <= 0):
        raise ValueError("Multicomponent transport requires positive collision diameter and well depth.")
    tmin, tmax = mech.min_temperature, mech.max_temperature
    if not 0 < tmin < tmax:
        raise ValueError("No common positive temperature interval for transport fitting.")
    integrals = CollisionIntegrals(float(np.min(tmin / eps)), float(np.max(tmax / eps)))
    temperature = np.linspace(tmin, tmax, 50)
    logt = np.log(temperature)
    n = mech.n_species
    stars = np.empty((3, n, n, 9))
    viscosity = np.empty((n, 5))
    diffusion = np.empty((n, n, 5))
    fit_cache = {}
    for i in range(n):
        omega22 = integrals.evaluate(temperature / mech.well_depth[i], delta[i, i])[0]
        mu = (5.0 / 16.0 * np.sqrt(np.pi * mech.molecular_weights[i] * BOLTZMANN * temperature / (1000.0 * AVOGADRO))
              / (omega22 * np.pi * mech.diameter[i] ** 2))
        target = np.sqrt(mu / np.sqrt(temperature))
        viscosity[i] = polyfit(logt, target, 4, w=1.0 / target)
        for j in range(i, n):
            d = float(delta[i, j])
            if d not in fit_cache:
                fit_cache[d] = integrals.fits(d)[1:]
            stars[:, i, j] = stars[:, j, i] = fit_cache[d]
            values = integrals.evaluate(temperature / eps[i, j], d)
            omega11 = values[0] / values[1]
            # mech molecular weights are kg/kmol; reduced mass is kg/molecule.
            reduced_mass = (mech.molecular_weights[i] * mech.molecular_weights[j]
                            / (mech.molecular_weights[i] + mech.molecular_weights[j]) / (1000.0 * AVOGADRO))
            target = (3.0 / 16.0 * np.sqrt(2.0 * np.pi / reduced_mass)
                      * BOLTZMANN ** 1.5 / (np.pi * sigma[i, j] ** 2 * omega11))
            diffusion[i, j] = diffusion[j, i] = polyfit(logt, target, 4, w=1.0 / target)
    return stars, viscosity, diffusion
