"""Residual and Jacobian assembly for the 1-D free-flame solver."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.linalg import get_lapack_funcs, lu_factor, lu_solve
from scipy.sparse.linalg import splu

from state import C_T, C_U, C_Y, build_transient_mask

try:
    from numba import njit
except Exception:  # pragma: no cover - optional acceleration
    njit = None

_NUMBA_AVAILABLE = njit is not None


def _outlet_species_flux_bc(problem) -> bool:
    mode = str(getattr(problem, "outlet_species_bc", "zero_gradient")).strip().lower()
    return mode in ("cantera", "cantera_flux", "flux", "total_flux", "outflow")


def _profile_start(problem) -> float:
    return perf_counter() if getattr(problem, "_profile", None) is not None else 0.0


def _profile_record(problem, key: str, t0: float, count: int = 1) -> None:
    profile = getattr(problem, "_profile", None)
    if profile is None:
        return
    entry = profile.setdefault(key, {"time_s": 0.0, "count": 0})
    entry["time_s"] += perf_counter() - t0
    entry["count"] += int(count)


def _profile_return(problem, key: str, t0: float, value):
    _profile_record(problem, key, t0)
    return value


def _state_views(x: np.ndarray, n_points: int, n_species: int):
    x_r = np.asarray(x, dtype=float).reshape(n_points, 2 + n_species)
    return x_r[:, C_U], x_r[:, C_T], x_r[:, C_Y:].T


def _residual_backend(problem):
    return getattr(problem, "residual_backend", None) or getattr(problem, "backend", None)


def _jacobian_backend(problem):
    return getattr(problem, "jacobian_backend", None) or getattr(problem, "backend", None)


# ---------------------------------------------------------------------------
#  Helpers de flujo difusivo
# ---------------------------------------------------------------------------
def _corrected_flux(Y_L: np.ndarray, Y_R: np.ndarray,
                    rho_f: float, D_f: np.ndarray, dz: float,
                    W: np.ndarray, W_mix_f: float,
                    basis: str = "molar") -> np.ndarray:
    """
    Flujo de especie corregido en la cara j+1/2.
    Idéntico a Flow1D::updateDiffFluxes (mixture-averaged, sin Soret).
    """
    if basis in ("molar", "mole"):
        W_mix_L = 1.0 / np.dot(Y_L, 1.0 / W)
        W_mix_R = 1.0 / np.dot(Y_R, 1.0 / W)
        X_L = Y_L * W_mix_L / W
        X_R = Y_R * W_mix_R / W
        dphi = (X_R - X_L) / dz
        J_star = -rho_f * (W / W_mix_f) * D_f * dphi
    else:
        dphi = (Y_R - Y_L) / dz
        J_star = -rho_f * D_f * dphi
    # Corrección para asegurar suma de flujos nula
    # Cantera usa Y_L (el nodo j) en vez del punto medio j+1/2 para distribuir el error:
    # m_flux(k,j) += sum*Y(x,k,j);
    return J_star - Y_L * J_star.sum()


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _assemble_residual_numba_core(
        F, u, T, Y, z, rho, cp_n, omega, hk_n, lam_face, flux,
        invW, Y_in, T_prof, has_T_prof, solve_energy, j_fixed,
        T_fixed, T_in, upwind_factor, outlet_species_flux,
    ):
        n_pts = z.shape[0]
        n_sp = Y.shape[0]
        nv = 2 + n_sp

        for i in range(F.shape[0]):
            F[i] = 0.0

        dz0 = z[1] - z[0]
        F[C_U] = -(rho[1] * u[1] - rho[0] * u[0]) / dz0
        if solve_energy:
            F[C_T] = T[0] - T_in
        else:
            F[C_T] = T[0] - (T_prof[0] if has_T_prof else T_in)

        mdot_in = rho[0] * u[0]
        k_exc = 0
        y_max = Y[0, 0]
        y_sum = 0.0
        for k in range(n_sp):
            yk = Y[k, 0]
            y_sum += yk
            if yk > y_max:
                y_max = yk
                k_exc = k
            F[C_Y + k] = -(flux[k, 0] + mdot_in * yk) + mdot_in * Y_in[k]
        F[C_Y + k_exc] = 1.0 - y_sum

        for j in range(1, n_pts - 1):
            b = j * nv
            dzm = z[j] - z[j - 1]
            dzp = z[j + 1] - z[j]
            dz2 = z[j + 1] - z[j - 1]

            if j_fixed >= 0 and j == j_fixed:
                if solve_energy:
                    F[b + C_U] = T[j] - T_fixed
                else:
                    F[b + C_U] = rho[j] * u[j] - rho[0] * 0.3
            elif j_fixed >= 0 and j > j_fixed:
                F[b + C_U] = -(rho[j] * u[j] - rho[j - 1] * u[j - 1]) / dzm
            else:
                F[b + C_U] = -(rho[j + 1] * u[j + 1] - rho[j] * u[j]) / dzp

            rho_j = rho[j]
            rho_u_j = rho_j * u[j]
            jloc = j if u[j] > 0.0 else j + 1
            dz_up = z[jloc] - z[jloc - 1]

            if solve_energy:
                dTdz_up = (T[jloc] - T[jloc - 1]) / dz_up
                if upwind_factor < 1.0:
                    dTdz_cent = T[j-1]*(-dzp)/(dzm*dz2) + T[j]*(dzp-dzm)/(dzp*dzm) + T[j+1]*dzm/(dzp*dz2)
                    dTdz = upwind_factor * dTdz_up + (1.0 - upwind_factor) * dTdz_cent
                else:
                    dTdz = dTdz_up
                cond = -2.0 * (
                    lam_face[j] * (T[j + 1] - T[j]) / dzp
                    - lam_face[j - 1] * (T[j] - T[j - 1]) / dzm
                ) / dz2

                en_sum = 0.0
                for k in range(n_sp):
                    dhk_dz = (hk_n[k, jloc] - hk_n[k, jloc - 1]) / dz_up
                    flx = 0.5 * (flux[k, j - 1] + flux[k, j])
                    en_sum += hk_n[k, j] * omega[k, j] * invW[k]
                    en_sum += flx * dhk_dz * invW[k]

                cp_j = cp_n[j]
                F[b + C_T] = (-cp_j * rho_u_j * dTdz - cond - en_sum) / (rho_j * cp_j)
            else:
                F[b + C_T] = T[j] - (T_prof[j] if has_T_prof else T_in)

            for k in range(n_sp):
                dYdz_up = (Y[k, jloc] - Y[k, jloc - 1]) / dz_up
                if upwind_factor < 1.0:
                    dYdz_cent = Y[k, j-1]*(-dzp)/(dzm*dz2) + Y[k, j]*(dzp-dzm)/(dzp*dzm) + Y[k, j+1]*dzm/(dzp*dz2)
                    dYdz = upwind_factor * dYdz_up + (1.0 - upwind_factor) * dYdz_cent
                else:
                    dYdz = dYdz_up
                conv = rho_u_j * dYdz
                diff = 2.0 * (flux[k, j] - flux[k, j - 1]) / dz2
                F[b + C_Y + k] = (omega[k, j] - conv - diff) / rho_j

        b = (n_pts - 1) * nv
        F[b + C_U] = rho[n_pts - 1] * u[n_pts - 1] - rho[n_pts - 2] * u[n_pts - 2]
        if solve_energy:
            F[b + C_T] = T[n_pts - 1] - T[n_pts - 2]
        else:
            F[b + C_T] = T[n_pts - 1] - (T_prof[n_pts - 1] if has_T_prof else T[n_pts - 2])

        k_exc = 0
        y_max = Y[0, n_pts - 1]
        y_sum = 0.0
        mdot_out = rho[n_pts - 1] * u[n_pts - 1]
        for k in range(n_sp):
            yk = Y[k, n_pts - 1]
            y_sum += yk
            if yk > y_max:
                y_max = yk
                k_exc = k
            if outlet_species_flux:
                F[b + C_Y + k] = flux[k, n_pts - 2] + mdot_out * yk
            else:
                F[b + C_Y + k] = yk - Y[k, n_pts - 2]
        F[b + C_Y + k_exc] = 1.0 - y_sum
else:
    _assemble_residual_numba_core = None


def _apply_transient_terms(
    F: np.ndarray,
    x: np.ndarray,
    problem,
    rdt: float,
    x_old: np.ndarray | None,
) -> np.ndarray:
    if rdt > 0.0 and x_old is not None:
        x_old = np.asarray(x_old, dtype=float)
        mask = build_transient_mask(
            int(problem.n_points), int(problem.n_species),
            solve_energy=bool(problem.solve_energy),
        )
        F -= mask * rdt * (x - x_old)
    return F


# ---------------------------------------------------------------------------
#  Función residual principal
# ---------------------------------------------------------------------------
def residual(
    x: np.ndarray,
    problem,
    rdt: float = 0.0,
    x_old: np.ndarray | None = None,
) -> np.ndarray:
    """
    Residual acoplado completo.

    Parametros
    ----------
    x      : vector de estado actual, layout por-punto entrelazado
    problem: FreeFlameProblem
    rdt    : 1/dt  (0.0 = modo estacionario; > 0 = modo transitorio)
    x_old  : solucion en el paso n   (requerida si rdt > 0)
    """
    t_profile = _profile_start(problem)
    residual_backend = _residual_backend(problem)
    x = np.asarray(x, dtype=float)
    n_pts = int(problem.n_points)
    n_sp = int(problem.n_species)
    nv = 2 + n_sp
    backend = residual_backend
    z = problem.z
    W = backend.W
    invW = backend.invW
    basis = str(getattr(problem, "flux_gradient_basis", "molar")).lower()

    # ------------------------------------------------------------------
    #  1. Propiedades nodales
    # ------------------------------------------------------------------
    u, T, Y = _state_views(x, n_pts, n_sp)

    # ``omega`` and ``hk_n`` are caller-owned output buffers for the fused
    # backend. The grid backend already allocates/returns rho, cp and lambda;
    # do not copy those arrays into a second set on every residual evaluation.
    omega = np.empty((n_sp, n_pts))
    hk_n = np.empty((n_sp, n_pts))

    try:
        if hasattr(backend, "eval_grid_thermo_kinetics_into"):
            rho_g, cp_g = backend.eval_grid_thermo_kinetics_into(T, Y, omega, hk_n)
            rho = _to_numpy(rho_g)
            cp_n = _to_numpy(cp_g)
            lam_n = np.zeros(n_pts, dtype=float)
        elif hasattr(backend, "eval_grid_into"):
            rho_g, _, cp_g, lam_g = backend.eval_grid_into(T, Y, omega, hk_n)
            rho = _to_numpy(rho_g)
            cp_n = _to_numpy(cp_g)
            lam_n = _to_numpy(lam_g)
        else:
            rho = np.empty(n_pts, dtype=float)
            cp_n = np.empty(n_pts, dtype=float)
            lam_n = np.empty(n_pts, dtype=float)
            eval_node_tk_into = getattr(backend, "eval_node_thermo_kinetics_into", None)
            eval_node_into = backend.eval_node_into
            for j in range(n_pts):
                if eval_node_tk_into is not None:
                    rho[j], cp_n[j] = eval_node_tk_into(
                        T[j], Y[:, j], omega[:, j], hk_n[:, j]
                    )
                    lam_n[j] = 0.0
                else:
                    rho[j], _, cp_n[j], lam_n[j] = eval_node_into(
                        T[j], Y[:, j], omega[:, j], hk_n[:, j]
                    )
    except Exception as exc:
        if bool(getattr(problem, "debug_residual_errors", False)):
            print(f"EXCEPTION IN RESIDUAL PROP: {exc}")
        problem.last_residual_error = str(exc)
        return _profile_return(problem, "residual_full", t_profile, np.full(x.size, 1.0e20))

    # ------------------------------------------------------------------
    #  2. Flujos difusivos en caras (n_pts-1 caras)
    # ------------------------------------------------------------------
    flux = np.empty((n_sp, n_pts - 1))

    try:
        if hasattr(backend, "eval_faces"):
            T_face = 0.5 * (T[:-1] + T[1:])
            Y_face = 0.5 * (Y[:, :-1] + Y[:, 1:])
            rho_f, D_f, lam_f, W_mix_f = backend.eval_faces(T_face, Y_face)
            rho_face = _to_numpy(rho_f)
            D_f = _to_numpy(D_f)
            lam_face = _to_numpy(lam_f)
            W_mix_f = _to_numpy(W_mix_f)
            dz_face = z[1:] - z[:-1]
            if basis in ("molar", "mole"):
                face_coeff = rho_face[None, :] * (W[:, None] / W_mix_f[None, :]) * D_f
            else:
                face_coeff = rho_face[None, :] * D_f
            flux[:] = _corrected_flux_frozen(Y[:, :-1], Y[:, 1:], face_coeff, dz_face, W, basis)
        else:
            eval_face = backend.eval_midpoint_full_transport
            rho_face = np.empty(n_pts - 1, dtype=float)
            lam_face = np.empty(n_pts - 1, dtype=float)
            for jf in range(n_pts - 1):
                dz_f = z[jf + 1] - z[jf]
                rho_f, D_f, lam_f, W_mix_f = eval_face(
                    T[jf], T[jf + 1], Y[:, jf], Y[:, jf + 1]
                )
                rho_face[jf] = rho_f
                lam_face[jf] = lam_f
                flux[:, jf] = _corrected_flux(
                    Y[:, jf], Y[:, jf + 1], rho_f, D_f, dz_f, W, W_mix_f, basis)
    except Exception as exc:
        if bool(getattr(problem, "debug_residual_errors", False)):
            print(f"EXCEPTION IN RESIDUAL FACE: {exc}")
        problem.last_residual_error = str(exc)
        return _profile_return(problem, "residual_full", t_profile, np.full(x.size, 1.0e20))

    if _assemble_residual_numba_core is not None and bool(
        getattr(problem, "use_numba_residual", True)
    ):
        try:
            F_numba = np.empty(x.size, dtype=float)
            T_prof = getattr(problem, "T_profile_fixed", None)
            has_T_prof = T_prof is not None
            T_prof_arr = (
                np.asarray(T_prof, dtype=float)
                if has_T_prof else np.zeros(n_pts, dtype=float)
            )
            j_fixed = -1 if problem.j_fixed is None else int(problem.j_fixed)
            T_fixed = (
                float(problem.T_fixed_point)
                if problem.T_fixed_point is not None else 0.0
            )
            t_assembly = _profile_start(problem)
            _assemble_residual_numba_core(
                F_numba, u, T, Y, np.asarray(z, dtype=float), rho, cp_n,
                omega, hk_n, lam_face, flux, invW, np.asarray(problem.Y_in, dtype=float),
                T_prof_arr, bool(has_T_prof), bool(problem.solve_energy),
                j_fixed, T_fixed, float(problem.T_in), float(getattr(problem, "upwind_factor", 1.0)),
                bool(_outlet_species_flux_bc(problem)),
            )
            _profile_record(problem, "residual_assembly_numba", t_assembly)
            _apply_transient_terms(F_numba, x, problem, rdt, x_old)
            if not np.all(np.isfinite(F_numba)):
                F_numba = np.full(x.size, 1.0e20)
            return _profile_return(problem, "residual_full", t_profile, F_numba)
        except Exception as exc:
            problem.last_residual_error = f"Numba residual fallback: {exc}"
            if bool(getattr(problem, "debug_residual_errors", False)):
                print(problem.last_residual_error)

    # ------------------------------------------------------------------
    #  3. Ensamblar residual bloque a bloque
    # ------------------------------------------------------------------
    F = np.empty(x.size)

    # --- j=0 (inlet / left boundary) ---
    _left_bc(F, u, T, Y, rho, flux[:, 0], problem, nv, n_sp)

    # --- j=1 .. N-2 (interior) ---
    for j in range(1, n_pts - 1):
        b = j * nv
        dzm = z[j] - z[j - 1]
        dzp = z[j + 1] - z[j]
        dz2 = z[j + 1] - z[j - 1]
        fm = flux[:, j - 1]
        fp = flux[:, j]

        # U / continuity equation (always algebraic, diag=0)
        j_fixed = problem.j_fixed
        if j_fixed is not None and j == j_fixed:
            # Free flame: replace continuity with fixed-point constraint.
            if problem.solve_energy:
                F[b + C_U] = T[j] - float(problem.T_fixed_point)
            else:
                F[b + C_U] = rho[j] * u[j] - rho[0] * 0.3
        elif j_fixed is not None and j > j_fixed:
            F[b + C_U] = -(rho[j] * u[j] - rho[j - 1] * u[j - 1]) / dzm
        else:
            F[b + C_U] = -(rho[j + 1] * u[j + 1] - rho[j] * u[j]) / dzp

        # T / energy equation (differential if solve_energy, diag=1)
        if problem.solve_energy:
            F[b + C_T] = _energy_residual(
                u, T, Y, rho, cp_n, lam_n, hk_n, omega, lam_face, fm, fp,
                j, z, dzm, dzp, dz2, n_sp, W, invW, problem)
        else:
            T_prof = getattr(problem, "T_profile_fixed", None)
            F[b + C_T] = T[j] - (T_prof[j] if T_prof is not None else problem.T_in)

        # Y_k / species equations (differential, diag=1)
        rho_u_j = rho[j] * u[j]
        for k in range(n_sp):
            jloc = j if u[j] > 0.0 else j + 1
            dz_up = z[jloc] - z[jloc - 1]
            dYdz_up = (Y[k, jloc] - Y[k, jloc - 1]) / dz_up
            upwind_factor = float(getattr(problem, "upwind_factor", 1.0))
            if upwind_factor < 1.0:
                dYdz_cent = Y[k, j-1]*(-dzp)/(dzm*dz2) + Y[k, j]*(dzp-dzm)/(dzp*dzm) + Y[k, j+1]*dzm/(dzp*dz2)
                dYdz = upwind_factor * dYdz_up + (1.0 - upwind_factor) * dYdz_cent
            else:
                dYdz = dYdz_up
            conv = rho_u_j * dYdz
            diff = 2.0 * (fp[k] - fm[k]) / dz2
            F[b + C_Y + k] = (omega[k, j] - conv - diff) / rho[j]

    # --- j=N-1 (outlet / right boundary) ---
    _right_bc(F, u, T, Y, rho, flux[:, -1], problem, nv, n_sp, n_pts)

    # ------------------------------------------------------------------
    #  4. Término transitorio (idéntico a "rsd[n] -= rdt*(x-x_old)" en Cantera)
    # ------------------------------------------------------------------
    _apply_transient_terms(F, x, problem, rdt, x_old)

    if not np.all(np.isfinite(F)):
        return _profile_return(problem, "residual_full", t_profile, np.full(x.size, 1.0e20))

    return _profile_return(problem, "residual_full", t_profile, F)


# ---------------------------------------------------------------------------
#  Condiciones de frontera
# ---------------------------------------------------------------------------
def _left_bc(F: np.ndarray, u, T, Y, rho, flux0, problem, nv, n_sp):
    """
    j=0 : inlet de gas fresco.
    Reproduce Flow1D::evalContinuity + evalEnergy + evalSpecies en jmin=0.
    """
    b = 0
    dz0 = problem.z[1] - problem.z[0]

    # Continuity (forward difference, algebraic)
    F[b + C_U] = -(rho[1] * u[1] - rho[0] * u[0]) / dz0

    # Temperature
    if bool(problem.solve_energy):
        F[b + C_T] = T[0] - float(problem.T_in)
    else:
        T_prof = getattr(problem, "T_profile_fixed", None)
        F[b + C_T] = T[0] - (float(T_prof[0]) if T_prof is not None
                              else float(problem.T_in))

    # Species: inlet condition with excess-species normalisation
    mdot_in = rho[0] * u[0]
    Y_in = problem.Y_in
    F[b + C_Y:b + C_Y + n_sp] = (-(flux0 + mdot_in * Y[:, 0])
                                 + mdot_in * Y_in)
    k_exc = int(np.argmax(Y[:, 0]))
    F[b + C_Y + k_exc] = 1.0 - float(Y[:, 0].sum())


def _right_bc(F: np.ndarray, u, T, Y, rho, flux_last, problem, nv, n_sp, n_pts):
    """
    j=N-1 : outlet (free-flow).
    Reproduce Outlet1D::eval + Flow1D::evalContinuity/evalEnergy/evalSpecies
    en jmax=N-1.
    """
    b = (n_pts - 1) * nv

    # Continuity (algebraic)
    F[b + C_U] = rho[-1] * u[-1] - rho[-2] * u[-2]

    # Temperature
    if bool(problem.solve_energy):
        F[b + C_T] = T[-1] - T[-2]          # zero-gradient outlet
    else:
        T_prof = getattr(problem, "T_profile_fixed", None)
        F[b + C_T] = T[-1] - (float(T_prof[-1]) if T_prof is not None
                               else T[-2])

    # Species: production default is zero-gradient; cantera_flux matches Flow1D.
    k_exc = int(np.argmax(Y[:, -1]))
    if _outlet_species_flux_bc(problem):
        mdot_out = rho[-1] * u[-1]
        F[b + C_Y:b + C_Y + n_sp] = flux_last + mdot_out * Y[:, -1]
    else:
        F[b + C_Y:b + C_Y + n_sp] = Y[:, -1] - Y[:, -2]
    F[b + C_Y + k_exc] = 1.0 - float(Y[:, -1].sum())


# ---------------------------------------------------------------------------
#  Ecuación de energía interior
# ---------------------------------------------------------------------------
def _energy_residual(u, T, Y, rho, cp_n, lam_n, hk_n, omega, lam_face,
                     fm, fp, j, z, dzm, dzp, dz2, n_sp, W, invW, problem):
    """Residual de la ecuación de energía en el punto interior j."""
    rho_u_j = rho[j] * u[j]

    # Upwind para convección
    jloc = j if u[j] > 0.0 else j + 1
    dz_up = z[jloc] - z[jloc - 1]
    dTdz_up = (T[jloc] - T[jloc - 1]) / dz_up
    upwind_factor = float(getattr(problem, "upwind_factor", 1.0))
    if upwind_factor < 1.0:
        dTdz_cent = T[j-1]*(-dzp)/(dzm*dz2) + T[j]*(dzp-dzm)/(dzp*dzm) + T[j+1]*dzm/(dzp*dz2)
        dTdz = upwind_factor * dTdz_up + (1.0 - upwind_factor) * dTdz_cent
    else:
        dTdz = dTdz_up

    # Conducción centrada
    lam_m = lam_face[j - 1]
    lam_p = lam_face[j]
    cond = -2.0 * (lam_p * (T[j + 1] - T[j]) / dzp
                   - lam_m * (T[j] - T[j - 1]) / dzm) / dz2

    # Gradientes de entalpía (upwind)
    dhk_dz = (hk_n[:, jloc] - hk_n[:, jloc - 1]) / dz_up

    flx = 0.5 * (fm + fp)
    en_sum = float(np.dot(hk_n[:, j] * omega[:, j], invW) +
                   np.dot(flx * dhk_dz, invW))

    return (-cp_n[j] * rho_u_j * dTdz - cond - en_sum) / (rho[j] * cp_n[j])


# ---------------------------------------------------------------------------
#  Evaluacion local tipo Cantera (OneDim::eval(j, ...))
# ---------------------------------------------------------------------------
def _to_numpy(arr):
    return arr.get() if hasattr(arr, "get") else arr


def build_local_jacobian_cache(x: np.ndarray, problem) -> dict:
    """
    Build cache used by local Jacobian evaluations.

    Fidelity note:
    - Thermodynamic/kinetic nodal properties are available and may be updated
      locally per perturbed point.
    - Transport properties are frozen from the base state, matching Cantera's
      Jacobian path where transport is not updated by default during eval(j,...).
    """
    t_profile = _profile_start(problem)
    x = np.asarray(x, dtype=float)
    n_pts = int(problem.n_points)
    n_sp = int(problem.n_species)

    backend = _jacobian_backend(problem)
    z = problem.z
    W = np.asarray(backend.W, dtype=float)
    invW = np.asarray(backend.invW, dtype=float)
    basis = str(getattr(problem, "flux_gradient_basis", "molar")).lower()

    u, T, Y = _state_views(x, n_pts, n_sp)

    # Nodal thermo/kinetics at base state
    rho = np.empty(n_pts, dtype=float)
    cp_n = np.empty(n_pts, dtype=float)
    lam_n = np.empty(n_pts, dtype=float)
    omega = np.empty((n_sp, n_pts), dtype=float)
    hk_n = np.empty((n_sp, n_pts), dtype=float)

    if hasattr(backend, "eval_grid_thermo_kinetics_into"):
        rho_g, cp_g = backend.eval_grid_thermo_kinetics_into(T, Y, omega, hk_n)
        rho[:] = _to_numpy(rho_g)
        cp_n[:] = _to_numpy(cp_g)
        lam_n.fill(0.0)
        omega[:] = _to_numpy(omega)
        hk_n[:] = _to_numpy(hk_n)
    elif hasattr(backend, "eval_grid_into"):
        rho_g, _, cp_g, lam_g = backend.eval_grid_into(T, Y, omega, hk_n)
        rho[:] = _to_numpy(rho_g)
        cp_n[:] = _to_numpy(cp_g)
        lam_n[:] = _to_numpy(lam_g)
        omega[:] = _to_numpy(omega)
        hk_n[:] = _to_numpy(hk_n)
    else:
        eval_node_into = backend.eval_node_into
        for j in range(n_pts):
            rho[j], _, cp_n[j], lam_n[j] = eval_node_into(T[j], Y[:, j], omega[:, j], hk_n[:, j])

    # Face transport at base state (frozen during local Jacobian eval)
    n_faces = max(0, n_pts - 1)
    lam_face = np.empty(n_faces, dtype=float)
    face_coeff = np.empty((n_sp, n_faces), dtype=float)
    dz_face = z[1:] - z[:-1]

    if n_faces > 0:
        if hasattr(backend, "eval_faces"):
            T_face = 0.5 * (T[:-1] + T[1:])
            Y_face = 0.5 * (Y[:, :-1] + Y[:, 1:])
            rho_f, D_f, lam_f, W_mix_f = backend.eval_faces(T_face, Y_face)
            rho_f = _to_numpy(rho_f)
            D_f = _to_numpy(D_f)
            lam_f = _to_numpy(lam_f)
            W_mix_f = _to_numpy(W_mix_f)
            lam_face[:] = lam_f
            if basis in ("molar", "mole"):
                face_coeff[:] = rho_f[None, :] * (W[:, None] / W_mix_f[None, :]) * D_f
            else:
                face_coeff[:] = rho_f[None, :] * D_f
        else:
            eval_face = backend.eval_midpoint_full_transport
            for jf in range(n_faces):
                rho_f, D_f, lam_f, W_mix_f = eval_face(T[jf], T[jf + 1], Y[:, jf], Y[:, jf + 1])
                lam_face[jf] = lam_f
                if basis in ("molar", "mole"):
                    face_coeff[:, jf] = rho_f * (W / W_mix_f) * D_f
                else:
                    face_coeff[:, jf] = rho_f * D_f

    row_offsets = np.arange(2 + n_sp, dtype=np.int32)
    local_rows = []
    for j in range(n_pts):
        p0 = max(0, j - 1)
        p1 = min(n_pts - 1, j + 1)
        blocks = np.arange(p0, p1 + 1, dtype=np.int32)[:, None] * (2 + n_sp)
        local_rows.append((blocks + row_offsets[None, :]).ravel())

    T_prof = getattr(problem, "T_profile_fixed", None)
    has_T_prof = T_prof is not None
    T_prof_arr = (
        np.asarray(T_prof, dtype=float)
        if has_T_prof else np.zeros(n_pts, dtype=float)
    )
    j_fixed = -1 if problem.j_fixed is None else int(problem.j_fixed)
    T_fixed = (
        float(problem.T_fixed_point)
        if problem.T_fixed_point is not None else 0.0
    )

    return _profile_return(problem, "jacobian_cache", t_profile, {
        "n_pts": n_pts,
        "n_sp": n_sp,
        "nv": 2 + n_sp,
        "z": np.asarray(z, dtype=float),
        "W": W,
        "invW": invW,
        "basis": basis,
        "rho": rho,
        "cp_n": cp_n,
        "lam_n": lam_n,
        "omega": omega,
        "hk_n": hk_n,
        "lam_face": lam_face,
        "face_coeff": face_coeff,
        "dz_face": dz_face,
        "local_rows": local_rows,
        "Y_in": np.asarray(problem.Y_in, dtype=float),
        "T_prof_arr": T_prof_arr,
        "has_T_prof": bool(has_T_prof),
        "solve_energy": bool(problem.solve_energy),
        "j_fixed": int(j_fixed),
        "T_fixed": float(T_fixed),
        "T_in": float(problem.T_in),
        "basis_molar": basis in ("molar", "mole"),
        "rho0": float(rho[0]),
    })


def _corrected_flux_frozen(Y_L: np.ndarray, Y_R: np.ndarray,
                           face_coeff: np.ndarray, dz: np.ndarray,
                           W: np.ndarray, basis: str) -> np.ndarray:
    """
    Corrected diffusive flux with frozen transport coefficients.
    """
    if basis in ("molar", "mole"):
        W_mix_L = 1.0 / np.sum(Y_L / W[:, None], axis=0)
        W_mix_R = 1.0 / np.sum(Y_R / W[:, None], axis=0)
        X_L = Y_L * (W_mix_L[None, :] / W[:, None])
        X_R = Y_R * (W_mix_R[None, :] / W[:, None])
        dphi = (X_R - X_L) / dz[None, :]
    else:
        dphi = (Y_R - Y_L) / dz[None, :]

    J_star = -face_coeff * dphi
    return J_star - Y_L * np.sum(J_star, axis=0, keepdims=True)


def residual_local_rows(x: np.ndarray, problem, center_j: int,
                        cache: dict | None = None):
    """
    Evaluate only residual rows affected by perturbation at point center_j.

    Returns
    -------
    rows : ndarray[int]
    vals : ndarray[float]
    """
    t_profile = _profile_start(problem)
    x = np.asarray(x, dtype=float)

    if cache is None:
        cache = build_local_jacobian_cache(x, problem)

    n_pts = int(cache["n_pts"])
    n_sp = int(cache["n_sp"])
    nv = int(cache["nv"])

    if n_pts < 2:
        raise ValueError("Se requieren al menos 2 puntos de malla.")

    z = cache["z"]
    W = cache["W"]
    invW = cache["invW"]
    basis = cache["basis"]
    solve_energy = bool(problem.solve_energy)
    j_fixed = problem.j_fixed
    T_prof = getattr(problem, "T_profile_fixed", None)
    T_in = float(problem.T_in)
    Y_in = problem.Y_in

    u, T, Y = _state_views(x, n_pts, n_sp)

    j_center = int(np.clip(center_j, 0, n_pts - 1))
    p0 = max(0, j_center - 1)
    p1 = min(n_pts - 1, j_center + 1)

    # Needed node range for equations p0..p1
    n0 = max(0, p0 - 1)
    n1 = min(n_pts - 1, p1 + 1)

    # Start from cached base nodal properties
    rho_local = cache["rho"][n0:n1 + 1].copy()
    cp_local = cache["cp_n"][n0:n1 + 1].copy()
    omega_local = cache["omega"][:, n0:n1 + 1].copy()
    hk_local = cache["hk_n"][:, n0:n1 + 1].copy()

    # Recompute only perturbed node thermo/kinetics (strictly enough + faster)
    backend = _jacobian_backend(problem)
    eval_node_tk_into = getattr(backend, "eval_node_thermo_kinetics_into", None)
    eval_node_into = backend.eval_node_into
    j = j_center
    if n0 <= j <= n1:
        jl = j - n0
        om = np.empty(n_sp, dtype=float)
        hk = np.empty(n_sp, dtype=float)
        if eval_node_tk_into is not None:
            rhoj, cpj = eval_node_tk_into(T[j], Y[:, j], om, hk)
        else:
            rhoj, _, cpj, _ = eval_node_into(T[j], Y[:, j], om, hk)
        rho_local[jl] = rhoj
        cp_local[jl] = cpj
        omega_local[:, jl] = om
        hk_local[:, jl] = hk

    off = -n0

    # Needed face range
    f0 = max(0, p0 - 1)
    f1 = min(n_pts - 2, p1)
    n_faces = f1 - f0 + 1

    flux_local = np.empty((n_sp, n_faces), dtype=float)
    if n_faces > 0:
        YL = Y[:, f0:f1 + 1]
        YR = Y[:, f0 + 1:f1 + 2]
        dz = cache["dz_face"][f0:f1 + 1]
        coeff = cache["face_coeff"][:, f0:f1 + 1]
        flux_local[:] = _corrected_flux_frozen(YL, YR, coeff, dz, W, basis)

    lam_face = cache["lam_face"]

    rows_out = cache["local_rows"][j_center]
    vals_out = np.empty(rows_out.size, dtype=float)
    out_i = 0

    for j in range(p0, p1 + 1):
        sl = slice(out_i, out_i + nv)
        block = vals_out[sl]

        if j == 0:
            dz0 = z[1] - z[0]
            rho0 = rho_local[0 + off]
            rho1 = rho_local[1 + off]
            block[C_U] = -(rho1 * u[1] - rho0 * u[0]) / dz0

            if solve_energy:
                block[C_T] = T[0] - T_in
            else:
                block[C_T] = T[0] - (float(T_prof[0]) if T_prof is not None else T_in)

            mdot_in = rho0 * u[0]
            flux0 = flux_local[:, 0 - f0]
            block[C_Y:C_Y + n_sp] = (-(flux0 + mdot_in * Y[:, 0]) + mdot_in * Y_in)
            k_exc = int(np.argmax(Y[:, 0]))
            block[C_Y + k_exc] = 1.0 - float(Y[:, 0].sum())

        elif j == n_pts - 1:
            rho_n = rho_local[(n_pts - 1) + off]
            rho_nm1 = rho_local[(n_pts - 2) + off]
            block[C_U] = rho_n * u[-1] - rho_nm1 * u[-2]

            if solve_energy:
                block[C_T] = T[-1] - T[-2]
            else:
                block[C_T] = T[-1] - (float(T_prof[-1]) if T_prof is not None else T[-2])

            k_exc = int(np.argmax(Y[:, -1]))
            if _outlet_species_flux_bc(problem):
                mdot_out = rho_n * u[-1]
                flux_last = flux_local[:, (n_pts - 2) - f0]
                block[C_Y:C_Y + n_sp] = flux_last + mdot_out * Y[:, -1]
            else:
                block[C_Y:C_Y + n_sp] = Y[:, -1] - Y[:, -2]
            block[C_Y + k_exc] = 1.0 - float(Y[:, -1].sum())

        else:
            dzm = z[j] - z[j - 1]
            dzp = z[j + 1] - z[j]
            dz2 = z[j + 1] - z[j - 1]
            fm = flux_local[:, (j - 1) - f0]
            fp = flux_local[:, j - f0]
            jo = j + off

            if j_fixed is not None and j == j_fixed:
                if solve_energy:
                    block[C_U] = T[j] - float(problem.T_fixed_point)
                else:
                    block[C_U] = rho_local[jo] * u[j] - cache["rho"][0] * 0.3
            elif j_fixed is not None and j > j_fixed:
                block[C_U] = -(rho_local[jo] * u[j] - rho_local[jo - 1] * u[j - 1]) / dzm
            else:
                block[C_U] = -(rho_local[jo + 1] * u[j + 1] - rho_local[jo] * u[j]) / dzp

            rho_j = rho_local[jo]
            rho_u_j = rho_j * u[j]
            jloc = j if u[j] > 0.0 else j + 1
            dz_up = z[jloc] - z[jloc - 1]

            if solve_energy:
                dTdz_up = (T[jloc] - T[jloc - 1]) / dz_up
                upwind_factor = float(getattr(problem, "upwind_factor", 1.0))
                if upwind_factor < 1.0:
                    dTdz_cent = T[j-1]*(-dzp)/(dzm*dz2) + T[j]*(dzp-dzm)/(dzp*dzm) + T[j+1]*dzm/(dzp*dz2)
                    dTdz = upwind_factor * dTdz_up + (1.0 - upwind_factor) * dTdz_cent
                else:
                    dTdz = dTdz_up

                lam_m = lam_face[j - 1]
                lam_p = lam_face[j]
                cond = -2.0 * (lam_p * (T[j + 1] - T[j]) / dzp - lam_m * (T[j] - T[j - 1]) / dzm) / dz2

                jloc_o = jloc + off
                hk_j = hk_local[:, jo]
                om_j = omega_local[:, jo]
                dhk_dz = (hk_local[:, jloc_o] - hk_local[:, jloc_o - 1]) / dz_up
                flx = 0.5 * (fm + fp)
                en_sum = float(np.dot(hk_j * om_j, invW) + np.dot(flx * dhk_dz, invW))
                cp_j = cp_local[jo]
                block[C_T] = (-cp_j * rho_u_j * dTdz - cond - en_sum) / (rho_j * cp_j)
            else:
                block[C_T] = T[j] - (T_prof[j] if T_prof is not None else T_in)

            om_j = omega_local[:, jo]
            dYdz_up = (Y[:, jloc] - Y[:, jloc - 1]) / dz_up
            upwind_factor = float(getattr(problem, "upwind_factor", 1.0))
            if upwind_factor < 1.0:
                dYdz_cent = Y[:, j-1]*(-dzp)/(dzm*dz2) + Y[:, j]*(dzp-dzm)/(dzp*dzm) + Y[:, j+1]*dzm/(dzp*dz2)
                dYdz = upwind_factor * dYdz_up + (1.0 - upwind_factor) * dYdz_cent
            else:
                dYdz = dYdz_up
            conv = rho_u_j * dYdz
            diff = 2.0 * (fp - fm) / dz2
            block[C_Y:C_Y + n_sp] = (om_j - conv - diff) / rho_j

        out_i += nv

    return _profile_return(problem, "residual_local", t_profile, (rows_out, vals_out))


def _corrected_flux_frozen_batch(Y_L: np.ndarray, Y_R: np.ndarray,
                                 face_coeff: np.ndarray, dz: np.ndarray,
                                 W: np.ndarray, basis: str) -> np.ndarray:
    if basis in ("molar", "mole"):
        Wb = W[None, :, None]
        W_mix_L = 1.0 / np.sum(Y_L / Wb, axis=1)
        W_mix_R = 1.0 / np.sum(Y_R / Wb, axis=1)
        X_L = Y_L * (W_mix_L[:, None, :] / Wb)
        X_R = Y_R * (W_mix_R[:, None, :] / Wb)
        dphi = (X_R - X_L) / dz[None, None, :]
    else:
        dphi = (Y_R - Y_L) / dz[None, None, :]

    J_star = -face_coeff[None, :, :] * dphi
    return J_star - Y_L * np.sum(J_star, axis=1, keepdims=True)


def _take_point_values(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    batch = np.arange(arr.shape[0])
    return arr[batch, :, idx]


def _local_cache_buffer(cache: dict, key: str, shape: tuple[int, ...],
                        dtype=float) -> np.ndarray:
    buffers = cache.setdefault("_buffers", {})
    arr = buffers.get(key)
    if arr is None or arr.shape != shape or arr.dtype != np.dtype(dtype):
        arr = np.empty(shape, dtype=dtype)
        buffers[key] = arr
    return arr


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _fill_local_jacobian_block_numba(
        rows_arr, cols_arr, vals_arr, start, rows, cols, vals, threshold,
    ):
        pos = start
        n_cols = cols.shape[0]
        n_rows = rows.shape[0]
        if threshold <= 0.0:
            for i in range(n_cols):
                col = cols[i]
                for k in range(n_rows):
                    rows_arr[pos] = rows[k]
                    cols_arr[pos] = col
                    vals_arr[pos] = vals[i, k]
                    pos += 1
            return pos

        for i in range(n_cols):
            col = cols[i]
            for k in range(n_rows):
                v = vals[i, k]
                if abs(v) > threshold or rows[k] == col:
                    rows_arr[pos] = rows[k]
                    cols_arr[pos] = col
                    vals_arr[pos] = v
                    pos += 1
        return pos


    @njit(cache=True)
    def _fill_banded_jacobian_block_numba(
        ab, lower, upper, rows, cols, vals, threshold,
    ):
        n_cols = cols.shape[0]
        n_rows = rows.shape[0]
        diag_band = lower + upper
        for i in range(n_cols):
            col = cols[i]
            for k in range(n_rows):
                row = rows[k]
                v = vals[i, k]
                if threshold <= 0.0 or abs(v) > threshold or row == col:
                    band_row = diag_band + row - col
                    if 0 <= band_row < ab.shape[0]:
                        ab[band_row, col] = v


    @njit(cache=True)
    def _fill_block_tridiag_jacobian_block_numba(
        lower_blocks, diag_blocks, upper_blocks, nv, rows, cols, vals, threshold,
    ):
        n_cols = cols.shape[0]
        n_rows = rows.shape[0]
        for i in range(n_cols):
            col = cols[i]
            cb = col // nv
            cv = col - cb * nv
            for k in range(n_rows):
                row = rows[k]
                rb = row // nv
                rv = row - rb * nv
                v = vals[i, k]
                if threshold <= 0.0 or abs(v) > threshold or row == col:
                    if rb == cb:
                        diag_blocks[rb, rv, cv] = v
                    elif rb == cb + 1:
                        lower_blocks[cb, rv, cv] = v
                    elif rb + 1 == cb:
                        upper_blocks[rb, rv, cv] = v


    @njit(cache=True)
    def _assemble_local_batch_numba_core(
        vals_out, u, T, Y, rho_local, cp_local, omega_local, hk_local,
        z, lam_face, face_coeff, dz_face, W, invW, Y_in, T_prof,
        has_T_prof, solve_energy, j_fixed, T_fixed, T_in, p0, p1, n0,
        f0, n_faces, basis_molar, rho0_base, upwind_factor, outlet_species_flux,
    ):
        n_batch = u.shape[0]
        n_sp = Y.shape[1]
        nv = 2 + n_sp
        n_pts = z.shape[0]

        flux_local = np.empty((n_batch, n_sp, n_faces), dtype=np.float64)
        for ib in range(n_batch):
            for lf in range(n_faces):
                f = f0 + lf
                jl = f - n0
                jr = jl + 1
                dz = dz_face[f]
                sum_j = 0.0

                if basis_molar:
                    denom_l = 0.0
                    denom_r = 0.0
                    for k in range(n_sp):
                        denom_l += Y[ib, k, jl] / W[k]
                        denom_r += Y[ib, k, jr] / W[k]
                    W_mix_l = 1.0 / denom_l
                    W_mix_r = 1.0 / denom_r
                    for k in range(n_sp):
                        x_l = Y[ib, k, jl] * W_mix_l / W[k]
                        x_r = Y[ib, k, jr] * W_mix_r / W[k]
                        j_star = -face_coeff[k, f] * (x_r - x_l) / dz
                        flux_local[ib, k, lf] = j_star
                        sum_j += j_star
                else:
                    for k in range(n_sp):
                        j_star = -face_coeff[k, f] * (Y[ib, k, jr] - Y[ib, k, jl]) / dz
                        flux_local[ib, k, lf] = j_star
                        sum_j += j_star

                for k in range(n_sp):
                    flux_local[ib, k, lf] -= Y[ib, k, jl] * sum_j

        out_i = 0
        for j in range(p0, p1 + 1):
            jo = j - n0
            for ib in range(n_batch):
                base = out_i

                if j == 0:
                    dz0 = z[1] - z[0]
                    rho0 = rho_local[ib, 0 - n0]
                    rho1 = rho_local[ib, 1 - n0]
                    vals_out[ib, base + C_U] = -(rho1 * u[ib, 1 - n0] - rho0 * u[ib, 0 - n0]) / dz0

                    if solve_energy:
                        vals_out[ib, base + C_T] = T[ib, 0 - n0] - T_in
                    else:
                        vals_out[ib, base + C_T] = T[ib, 0 - n0] - (T_prof[0] if has_T_prof else T_in)

                    mdot_in = rho0 * u[ib, 0 - n0]
                    y_sum = 0.0
                    k_exc = 0
                    y_max = Y[ib, 0, 0 - n0]
                    for k in range(n_sp):
                        yk = Y[ib, k, 0 - n0]
                        y_sum += yk
                        if yk > y_max:
                            y_max = yk
                            k_exc = k
                        flux0 = flux_local[ib, k, 0 - f0]
                        vals_out[ib, base + C_Y + k] = -(flux0 + mdot_in * yk) + mdot_in * Y_in[k]
                    vals_out[ib, base + C_Y + k_exc] = 1.0 - y_sum

                elif j == n_pts - 1:
                    rho_n = rho_local[ib, (n_pts - 1) - n0]
                    rho_nm1 = rho_local[ib, (n_pts - 2) - n0]
                    vals_out[ib, base + C_U] = rho_n * u[ib, (n_pts - 1) - n0] - rho_nm1 * u[ib, (n_pts - 2) - n0]

                    if solve_energy:
                        vals_out[ib, base + C_T] = T[ib, (n_pts - 1) - n0] - T[ib, (n_pts - 2) - n0]
                    else:
                        vals_out[ib, base + C_T] = T[ib, (n_pts - 1) - n0] - (
                            T_prof[n_pts - 1] if has_T_prof else T[ib, (n_pts - 2) - n0]
                        )

                    y_sum = 0.0
                    k_exc = 0
                    y_max = Y[ib, 0, (n_pts - 1) - n0]
                    mdot_out = rho_n * u[ib, (n_pts - 1) - n0]
                    for k in range(n_sp):
                        yk = Y[ib, k, (n_pts - 1) - n0]
                        y_sum += yk
                        if yk > y_max:
                            y_max = yk
                            k_exc = k
                        if outlet_species_flux:
                            flux_last = flux_local[ib, k, (n_pts - 2) - f0]
                            vals_out[ib, base + C_Y + k] = flux_last + mdot_out * yk
                        else:
                            vals_out[ib, base + C_Y + k] = yk - Y[ib, k, (n_pts - 2) - n0]
                    vals_out[ib, base + C_Y + k_exc] = 1.0 - y_sum

                else:
                    dzm = z[j] - z[j - 1]
                    dzp = z[j + 1] - z[j]
                    dz2 = z[j + 1] - z[j - 1]

                    if j_fixed >= 0 and j == j_fixed:
                        if solve_energy:
                            vals_out[ib, base + C_U] = T[ib, jo] - T_fixed
                        else:
                            vals_out[ib, base + C_U] = rho_local[ib, jo] * u[ib, jo] - rho0_base * 0.3
                    elif j_fixed >= 0 and j > j_fixed:
                        vals_out[ib, base + C_U] = -(
                            rho_local[ib, jo] * u[ib, jo]
                            - rho_local[ib, jo - 1] * u[ib, jo - 1]
                        ) / dzm
                    else:
                        vals_out[ib, base + C_U] = -(
                            rho_local[ib, jo + 1] * u[ib, jo + 1]
                            - rho_local[ib, jo] * u[ib, jo]
                        ) / dzp

                    rho_j = rho_local[ib, jo]
                    rho_u_j = rho_j * u[ib, jo]
                    jloc = j if u[ib, jo] > 0.0 else j + 1
                    jloc_o = jloc - n0
                    jloc_m_o = jloc_o - 1
                    dz_up = z[jloc] - z[jloc - 1]
                    fm_i = (j - 1) - f0
                    fp_i = j - f0

                    if solve_energy:
                        dTdz_up = (T[ib, jloc_o] - T[ib, jloc_m_o]) / dz_up
                        if upwind_factor < 1.0:
                            dTdz_cent = T[ib, jo-1]*(-dzp)/(dzm*dz2) + T[ib, jo]*(dzp-dzm)/(dzp*dzm) + T[ib, jo+1]*dzm/(dzp*dz2)
                            dTdz = upwind_factor * dTdz_up + (1.0 - upwind_factor) * dTdz_cent
                        else:
                            dTdz = dTdz_up
                        cond = -2.0 * (
                            lam_face[j] * (T[ib, jo + 1] - T[ib, jo]) / dzp
                            - lam_face[j - 1] * (T[ib, jo] - T[ib, jo - 1]) / dzm
                        ) / dz2

                        en_sum = 0.0
                        for k in range(n_sp):
                            fm = flux_local[ib, k, fm_i]
                            fp = flux_local[ib, k, fp_i]
                            dhk_dz = (hk_local[ib, k, jloc_o] - hk_local[ib, k, jloc_m_o]) / dz_up
                            flx = 0.5 * (fm + fp)
                            en_sum += hk_local[ib, k, jo] * omega_local[ib, k, jo] * invW[k]
                            en_sum += flx * dhk_dz * invW[k]

                        cp_j = cp_local[ib, jo]
                        vals_out[ib, base + C_T] = (
                            -cp_j * rho_u_j * dTdz - cond - en_sum
                        ) / (rho_j * cp_j)
                    else:
                        vals_out[ib, base + C_T] = T[ib, jo] - (T_prof[j] if has_T_prof else T_in)

                    for k in range(n_sp):
                        fm = flux_local[ib, k, fm_i]
                        fp = flux_local[ib, k, fp_i]
                        dYdz_up = (Y[ib, k, jloc_o] - Y[ib, k, jloc_m_o]) / dz_up
                        if upwind_factor < 1.0:
                            dYdz_cent = Y[ib, k, jo-1]*(-dzp)/(dzm*dz2) + Y[ib, k, jo]*(dzp-dzm)/(dzp*dzm) + Y[ib, k, jo+1]*dzm/(dzp*dz2)
                            dYdz = upwind_factor * dYdz_up + (1.0 - upwind_factor) * dYdz_cent
                        else:
                            dYdz = dYdz_up
                        conv = rho_u_j * dYdz
                        diff = 2.0 * (fp - fm) / dz2
                        vals_out[ib, base + C_Y + k] = (omega_local[ib, k, jo] - conv - diff) / rho_j

            out_i += nv
else:
    _fill_local_jacobian_block_numba = None
    _fill_banded_jacobian_block_numba = None
    _fill_block_tridiag_jacobian_block_numba = None
    _assemble_local_batch_numba_core = None


def residual_local_rows_batch_perturbed(
    x: np.ndarray,
    problem,
    center_j: int,
    cols: np.ndarray,
    x_perturbed: np.ndarray,
    cache: dict | None = None,
    center_thermo: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
):
    """
    Evaluate local rows for point-wise perturbations without materializing a
    full (n_vars_per_point, n_state) state batch.
    """
    t_profile = _profile_start(problem)
    x = np.asarray(x, dtype=float)
    cols = np.asarray(cols, dtype=np.int32)
    x_perturbed = np.asarray(x_perturbed, dtype=float)

    if cache is None:
        cache = build_local_jacobian_cache(x, problem)

    if _assemble_local_batch_numba_core is None or not bool(
        getattr(problem, "use_numba_local_jacobian", True)
    ):
        n_batch = int(cols.size)
        x_batch = np.broadcast_to(x, (n_batch, x.size)).copy()
        x_batch[np.arange(n_batch), cols] = x_perturbed
        return residual_local_rows_batch(x_batch, problem, center_j, cache=cache)

    n_batch = int(cols.size)
    n_pts = int(cache["n_pts"])
    n_sp = int(cache["n_sp"])
    nv = int(cache["nv"])

    j_center = int(np.clip(center_j, 0, n_pts - 1))
    p0 = max(0, j_center - 1)
    p1 = min(n_pts - 1, j_center + 1)
    n0 = max(0, p0 - 1)
    n1 = min(n_pts - 1, p1 + 1)
    local_len = n1 - n0 + 1
    center_l = j_center - n0

    x_r = x.reshape(n_pts, nv)
    base = j_center * nv
    local_vars = cols - base

    backend = _jacobian_backend(problem)
    u_local = _local_cache_buffer(cache, "u_local", (n_batch, local_len))
    T_local = _local_cache_buffer(cache, "T_local", (n_batch, local_len))
    Y_local = _local_cache_buffer(cache, "Y_local", (n_batch, n_sp, local_len))
    rho_local = _local_cache_buffer(cache, "rho_local", (n_batch, local_len))
    cp_local = _local_cache_buffer(cache, "cp_local", (n_batch, local_len))
    omega_local = _local_cache_buffer(cache, "omega_local", (n_batch, n_sp, local_len))
    hk_local = _local_cache_buffer(cache, "hk_local", (n_batch, n_sp, local_len))

    u_local[:] = x_r[n0:n1 + 1, C_U][None, :]
    T_local[:] = x_r[n0:n1 + 1, C_T][None, :]
    Y_base = x_r[n0:n1 + 1, C_Y:].T
    Y_local[:] = Y_base[None, :, :]

    for ib in range(n_batch):
        var = int(local_vars[ib])
        val = float(x_perturbed[ib])
        if var == C_U:
            u_local[ib, center_l] = val
        elif var == C_T:
            T_local[ib, center_l] = val
        elif var >= C_Y:
            Y_local[ib, var - C_Y, center_l] = val

    rho_local[:] = cache["rho"][n0:n1 + 1][None, :]
    cp_local[:] = cache["cp_n"][n0:n1 + 1][None, :]
    omega_local[:] = cache["omega"][:, n0:n1 + 1][None, :, :]
    hk_local[:] = cache["hk_n"][:, n0:n1 + 1][None, :, :]

    if center_thermo is not None:
        rho_j, cp_j, omega_j, hk_j = center_thermo
        rho_local[:, center_l] = np.asarray(rho_j, dtype=float)
        cp_local[:, center_l] = np.asarray(cp_j, dtype=float)
        omega_local[:, :, center_l] = np.asarray(omega_j, dtype=float).T
        hk_local[:, :, center_l] = np.asarray(hk_j, dtype=float).T
    elif hasattr(backend, "eval_grid_thermo_kinetics_into"):
        omega_j = _local_cache_buffer(cache, "omega_j", (n_sp, n_batch))
        hk_j = _local_cache_buffer(cache, "hk_j", (n_sp, n_batch))
        rho_j, cp_j = backend.eval_grid_thermo_kinetics_into(
            T_local[:, center_l],
            Y_local[:, :, center_l].T,
            omega_j,
            hk_j,
        )
        rho_local[:, center_l] = np.asarray(_to_numpy(rho_j), dtype=float)
        cp_local[:, center_l] = np.asarray(_to_numpy(cp_j), dtype=float)
        omega_local[:, :, center_l] = np.asarray(_to_numpy(omega_j), dtype=float).T
        hk_local[:, :, center_l] = np.asarray(_to_numpy(hk_j), dtype=float).T
    elif hasattr(backend, "eval_grid_into"):
        omega_j = _local_cache_buffer(cache, "omega_j", (n_sp, n_batch))
        hk_j = _local_cache_buffer(cache, "hk_j", (n_sp, n_batch))
        rho_j, _, cp_j, _ = backend.eval_grid_into(
            T_local[:, center_l],
            Y_local[:, :, center_l].T,
            omega_j,
            hk_j,
        )
        rho_local[:, center_l] = np.asarray(_to_numpy(rho_j), dtype=float)
        cp_local[:, center_l] = np.asarray(_to_numpy(cp_j), dtype=float)
        omega_local[:, :, center_l] = np.asarray(_to_numpy(omega_j), dtype=float).T
        hk_local[:, :, center_l] = np.asarray(_to_numpy(hk_j), dtype=float).T
    else:
        eval_node_tk_into = getattr(backend, "eval_node_thermo_kinetics_into", None)
        for ib in range(n_batch):
            om = np.empty(n_sp, dtype=float)
            hk = np.empty(n_sp, dtype=float)
            if eval_node_tk_into is not None:
                rhoj, cpj = eval_node_tk_into(
                    T_local[ib, center_l], Y_local[ib, :, center_l], om, hk
                )
            else:
                rhoj, _, cpj, _ = backend.eval_node_into(
                    T_local[ib, center_l], Y_local[ib, :, center_l], om, hk
                )
            rho_local[ib, center_l] = rhoj
            cp_local[ib, center_l] = cpj
            omega_local[ib, :, center_l] = om
            hk_local[ib, :, center_l] = hk

    f0 = max(0, p0 - 1)
    f1 = min(n_pts - 2, p1)
    n_faces = f1 - f0 + 1
    rows_out = cache["local_rows"][j_center]
    vals_out = _local_cache_buffer(cache, "vals_out", (n_batch, rows_out.size))

    try:
        _assemble_local_batch_numba_core(
            vals_out,
            u_local,
            T_local,
            Y_local,
            rho_local,
            cp_local,
            omega_local,
            hk_local,
            cache["z"],
            cache["lam_face"],
            cache["face_coeff"],
            cache["dz_face"],
            cache["W"],
            cache["invW"],
            cache["Y_in"],
            cache["T_prof_arr"],
            bool(cache["has_T_prof"]),
            bool(cache["solve_energy"]),
            int(cache["j_fixed"]),
            float(cache["T_fixed"]),
            float(cache["T_in"]),
            int(p0),
            int(p1),
            int(n0),
            int(f0),
            int(n_faces),
            bool(cache["basis_molar"]),
            float(cache["rho0"]),
            float(getattr(problem, "upwind_factor", 1.0)),
            bool(_outlet_species_flux_bc(problem)),
        )
    except Exception as exc:
        problem.last_residual_error = f"Numba local fallback: {exc}"
        n_batch = int(cols.size)
        x_batch = np.broadcast_to(x, (n_batch, x.size)).copy()
        x_batch[np.arange(n_batch), cols] = x_perturbed
        return residual_local_rows_batch(x_batch, problem, center_j, cache=cache)

    return _profile_return(
        problem, "residual_local_batch_numba", t_profile, (rows_out, vals_out)
    )


def residual_local_rows_batch(x_batch: np.ndarray, problem, center_j: int,
                              cache: dict | None = None):
    """
    Batched version of residual_local_rows for perturbations at one grid point.

    Transport is frozen from the base cache, matching the Cantera-style local
    Jacobian path. Only nodal thermo/kinetics at center_j are recomputed.
    """
    t_profile = _profile_start(problem)
    x_batch = np.asarray(x_batch, dtype=float)
    if x_batch.ndim != 2:
        raise ValueError("x_batch debe tener shape (n_batch, n_state).")

    if cache is None:
        cache = build_local_jacobian_cache(x_batch[0], problem)

    n_batch = int(x_batch.shape[0])
    n_pts = int(cache["n_pts"])
    n_sp = int(cache["n_sp"])
    nv = int(cache["nv"])
    if n_pts < 2:
        raise ValueError("Se requieren al menos 2 puntos de malla.")

    z = cache["z"]
    W = cache["W"]
    invW = cache["invW"]
    basis = cache["basis"]
    solve_energy = bool(problem.solve_energy)
    j_fixed = problem.j_fixed
    T_prof = getattr(problem, "T_profile_fixed", None)
    T_in = float(problem.T_in)
    Y_in = np.asarray(problem.Y_in, dtype=float)

    x_r = x_batch.reshape(n_batch, n_pts, nv)
    u = x_r[:, :, C_U]
    T = x_r[:, :, C_T]
    Y = x_r[:, :, C_Y:].transpose(0, 2, 1)

    j_center = int(np.clip(center_j, 0, n_pts - 1))
    p0 = max(0, j_center - 1)
    p1 = min(n_pts - 1, j_center + 1)
    n0 = max(0, p0 - 1)
    n1 = min(n_pts - 1, p1 + 1)
    local_len = n1 - n0 + 1

    rho_local = np.broadcast_to(cache["rho"][n0:n1 + 1], (n_batch, local_len)).copy()
    cp_local = np.broadcast_to(cache["cp_n"][n0:n1 + 1], (n_batch, local_len)).copy()
    omega_local = np.broadcast_to(
        cache["omega"][:, n0:n1 + 1][None, :, :],
        (n_batch, n_sp, local_len),
    ).copy()
    hk_local = np.broadcast_to(
        cache["hk_n"][:, n0:n1 + 1][None, :, :],
        (n_batch, n_sp, local_len),
    ).copy()

    if n0 <= j_center <= n1:
        jl = j_center - n0
        omega_j = np.empty((n_sp, n_batch), dtype=float)
        hk_j = np.empty((n_sp, n_batch), dtype=float)
        backend = _jacobian_backend(problem)
        if hasattr(backend, "eval_grid_thermo_kinetics_into"):
            rho_j, cp_j = backend.eval_grid_thermo_kinetics_into(
                T[:, j_center],
                Y[:, :, j_center].T,
                omega_j,
                hk_j,
            )
            rho_local[:, jl] = np.asarray(_to_numpy(rho_j), dtype=float)
            cp_local[:, jl] = np.asarray(_to_numpy(cp_j), dtype=float)
            omega_local[:, :, jl] = np.asarray(_to_numpy(omega_j), dtype=float).T
            hk_local[:, :, jl] = np.asarray(_to_numpy(hk_j), dtype=float).T
        elif hasattr(backend, "eval_grid_into"):
            rho_j, _, cp_j, _ = backend.eval_grid_into(
                T[:, j_center],
                Y[:, :, j_center].T,
                omega_j,
                hk_j,
            )
            rho_local[:, jl] = np.asarray(_to_numpy(rho_j), dtype=float)
            cp_local[:, jl] = np.asarray(_to_numpy(cp_j), dtype=float)
            omega_local[:, :, jl] = np.asarray(_to_numpy(omega_j), dtype=float).T
            hk_local[:, :, jl] = np.asarray(_to_numpy(hk_j), dtype=float).T
        else:
            eval_node_tk_into = getattr(backend, "eval_node_thermo_kinetics_into", None)
            for ib in range(n_batch):
                om = np.empty(n_sp, dtype=float)
                hk = np.empty(n_sp, dtype=float)
                if eval_node_tk_into is not None:
                    rhoj, cpj = eval_node_tk_into(T[ib, j_center], Y[ib, :, j_center], om, hk)
                else:
                    rhoj, _, cpj, _ = backend.eval_node_into(T[ib, j_center], Y[ib, :, j_center], om, hk)
                rho_local[ib, jl] = rhoj
                cp_local[ib, jl] = cpj
                omega_local[ib, :, jl] = om
                hk_local[ib, :, jl] = hk

    off = -n0
    f0 = max(0, p0 - 1)
    f1 = min(n_pts - 2, p1)
    n_faces = f1 - f0 + 1

    flux_local = np.empty((n_batch, n_sp, n_faces), dtype=float)
    if n_faces > 0:
        YL = Y[:, :, f0:f1 + 1]
        YR = Y[:, :, f0 + 1:f1 + 2]
        dz = cache["dz_face"][f0:f1 + 1]
        coeff = cache["face_coeff"][:, f0:f1 + 1]
        flux_local[:] = _corrected_flux_frozen_batch(YL, YR, coeff, dz, W, basis)

    lam_face = cache["lam_face"]
    rows_out = cache["local_rows"][j_center]
    vals_out = np.empty((n_batch, rows_out.size), dtype=float)
    out_i = 0
    batch_idx = np.arange(n_batch)

    for j in range(p0, p1 + 1):
        sl = slice(out_i, out_i + nv)
        block = vals_out[:, sl]

        if j == 0:
            dz0 = z[1] - z[0]
            rho0 = rho_local[:, 0 + off]
            rho1 = rho_local[:, 1 + off]
            block[:, C_U] = -(rho1 * u[:, 1] - rho0 * u[:, 0]) / dz0

            if solve_energy:
                block[:, C_T] = T[:, 0] - T_in
            else:
                block[:, C_T] = T[:, 0] - (float(T_prof[0]) if T_prof is not None else T_in)

            mdot_in = rho0 * u[:, 0]
            flux0 = flux_local[:, :, 0 - f0]
            left_species = (-(flux0 + mdot_in[:, None] * Y[:, :, 0]) + mdot_in[:, None] * Y_in[None, :])
            block[:, C_Y:C_Y + n_sp] = left_species
            k_exc = np.argmax(Y[:, :, 0], axis=1)
            block[batch_idx, C_Y + k_exc] = 1.0 - np.sum(Y[:, :, 0], axis=1)

        elif j == n_pts - 1:
            rho_n = rho_local[:, (n_pts - 1) + off]
            rho_nm1 = rho_local[:, (n_pts - 2) + off]
            block[:, C_U] = rho_n * u[:, -1] - rho_nm1 * u[:, -2]

            if solve_energy:
                block[:, C_T] = T[:, -1] - T[:, -2]
            else:
                block[:, C_T] = T[:, -1] - (float(T_prof[-1]) if T_prof is not None else T[:, -2])

            if _outlet_species_flux_bc(problem):
                mdot_out = rho_n * u[:, -1]
                flux_last = flux_local[:, :, (n_pts - 2) - f0]
                block[:, C_Y:C_Y + n_sp] = flux_last + mdot_out[:, None] * Y[:, :, -1]
            else:
                block[:, C_Y:C_Y + n_sp] = Y[:, :, -1] - Y[:, :, -2]
            k_exc = np.argmax(Y[:, :, -1], axis=1)
            block[batch_idx, C_Y + k_exc] = 1.0 - np.sum(Y[:, :, -1], axis=1)

        else:
            dzm = z[j] - z[j - 1]
            dzp = z[j + 1] - z[j]
            dz2 = z[j + 1] - z[j - 1]
            fm = flux_local[:, :, (j - 1) - f0]
            fp = flux_local[:, :, j - f0]
            jo = j + off

            if j_fixed is not None and j == j_fixed:
                if solve_energy:
                    block[:, C_U] = T[:, j] - float(problem.T_fixed_point)
                else:
                    block[:, C_U] = rho_local[:, jo] * u[:, j] - cache["rho"][0] * 0.3
            elif j_fixed is not None and j > j_fixed:
                block[:, C_U] = -(rho_local[:, jo] * u[:, j] - rho_local[:, jo - 1] * u[:, j - 1]) / dzm
            else:
                block[:, C_U] = -(rho_local[:, jo + 1] * u[:, j + 1] - rho_local[:, jo] * u[:, j]) / dzp

            rho_j = rho_local[:, jo]
            rho_u_j = rho_j * u[:, j]
            jloc = np.where(u[:, j] > 0.0, j, j + 1)
            jloc_m = jloc - 1
            dz_up = z[jloc] - z[jloc_m]
            jloc_o = jloc + off
            jloc_m_o = jloc_m + off

            if solve_energy:
                dTdz_up = (T[batch_idx, jloc] - T[batch_idx, jloc_m]) / dz_up
                upwind_factor = float(getattr(problem, "upwind_factor", 1.0))
                if upwind_factor < 1.0:
                    dTdz_cent = T[:, j-1]*(-dzp)/(dzm*dz2) + T[:, j]*(dzp-dzm)/(dzp*dzm) + T[:, j+1]*dzm/(dzp*dz2)
                    dTdz = upwind_factor * dTdz_up + (1.0 - upwind_factor) * dTdz_cent
                else:
                    dTdz = dTdz_up
                cond = -2.0 * (
                    lam_face[j] * (T[:, j + 1] - T[:, j]) / dzp
                    - lam_face[j - 1] * (T[:, j] - T[:, j - 1]) / dzm
                ) / dz2
                hk_j = hk_local[:, :, jo]
                om_j = omega_local[:, :, jo]
                hk_jloc = _take_point_values(hk_local, jloc_o)
                hk_jloc_m = _take_point_values(hk_local, jloc_m_o)
                dhk_dz = (hk_jloc - hk_jloc_m) / dz_up[:, None]
                flx = 0.5 * (fm + fp)
                en_sum = (
                    np.sum(hk_j * om_j * invW[None, :], axis=1)
                    + np.sum(flx * dhk_dz * invW[None, :], axis=1)
                )
                cp_j = cp_local[:, jo]
                block[:, C_T] = (-cp_j * rho_u_j * dTdz - cond - en_sum) / (rho_j * cp_j)
            else:
                block[:, C_T] = T[:, j] - (T_prof[j] if T_prof is not None else T_in)

            om_j = omega_local[:, :, jo]
            Y_jloc = _take_point_values(Y, jloc)
            Y_jloc_m = _take_point_values(Y, jloc_m)
            dYdz_up = (Y_jloc - Y_jloc_m) / dz_up[:, None]
            upwind_factor = float(getattr(problem, "upwind_factor", 1.0))
            if upwind_factor < 1.0:
                dYdz_cent = Y[:, :, j-1]*(-dzp)/(dzm*dz2) + Y[:, :, j]*(dzp-dzm)/(dzp*dzm) + Y[:, :, j+1]*dzm/(dzp*dz2)
                dYdz = upwind_factor * dYdz_up + (1.0 - upwind_factor) * dYdz_cent
            else:
                dYdz = dYdz_up
            conv = rho_u_j[:, None] * dYdz
            diff = 2.0 * (fp - fm) / dz2
            block[:, C_Y:C_Y + n_sp] = (om_j - conv - diff) / rho_j[:, None]

        out_i += nv

    return _profile_return(problem, "residual_local_batch", t_profile, (rows_out, vals_out))


# ---------------------------------------------------------------------------
#  Cantera-style local finite-difference Jacobian
# ---------------------------------------------------------------------------
class BandedJacobian:
    """General banded Jacobian in LAPACK dgbtrf storage."""

    def __init__(self, ab: np.ndarray, lower: int, upper: int):
        self.ab = np.asarray(ab, dtype=float)
        self.lower = int(lower)
        self.upper = int(upper)
        self.shape = (int(self.ab.shape[1]), int(self.ab.shape[1]))

    def copy(self) -> "BandedJacobian":
        out = BandedJacobian(self.ab.copy(), self.lower, self.upper)
        if hasattr(self, "_profile_problem"):
            setattr(out, "_profile_problem", getattr(self, "_profile_problem"))
        return out

    def diagonal(self) -> np.ndarray:
        return self.ab[self.lower + self.upper, :].copy()

    def setdiag(self, diag: np.ndarray) -> None:
        self.ab[self.lower + self.upper, :] = np.asarray(diag, dtype=float)


class BlockTridiagJacobian:
    """Block-tridiagonal Jacobian for point-interleaved 1-D flame states."""

    def __init__(self, lower: np.ndarray, diag: np.ndarray, upper: np.ndarray):
        self.lower = np.asarray(lower, dtype=float)
        self.diag = np.asarray(diag, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.n_blocks = int(self.diag.shape[0])
        self.block_size = int(self.diag.shape[1])
        n = self.n_blocks * self.block_size
        self.shape = (n, n)

    def copy(self) -> "BlockTridiagJacobian":
        out = BlockTridiagJacobian(
            self.lower.copy(),
            self.diag.copy(),
            self.upper.copy(),
        )
        if hasattr(self, "_profile_problem"):
            setattr(out, "_profile_problem", getattr(self, "_profile_problem"))
        if hasattr(self, "use_compiled_substitution"):
            out.use_compiled_substitution = bool(self.use_compiled_substitution)
        return out

    def diagonal(self) -> np.ndarray:
        # ``np.diagonal`` returns the block diagonals in point-major order.
        # This operation is used at every PTC timestep, so avoiding the Python
        # loop removes measurable overhead without changing any matrix entry.
        return np.diagonal(self.diag, axis1=1, axis2=2).reshape(-1).copy()

    def setdiag(self, diag: np.ndarray) -> None:
        d = np.asarray(diag, dtype=float).reshape(self.n_blocks, self.block_size)
        idx = np.arange(self.block_size)
        self.diag[:, idx, idx] = d

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        """Return the block-tridiagonal product ``J @ vector``.

        This deliberately stays in block form: the experimental shifted-LU
        correction in :func:`solve_linear` must evaluate the residual of the
        *new* pseudo-transient matrix without materialising a dense matrix.
        """
        x = np.asarray(vector, dtype=float).reshape(self.n_blocks, self.block_size)
        out = np.einsum("nij,nj->ni", self.diag, x, optimize=True)
        if self.n_blocks > 1:
            out[1:] += np.einsum("nij,nj->ni", self.lower, x[:-1], optimize=True)
            out[:-1] += np.einsum("nij,nj->ni", self.upper, x[1:], optimize=True)
        return out.ravel()


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _solve_block_tridiag_lu_numba(
        lu_blocks,
        pivots,
        lower_blocks,
        cprime,
        rhs,
    ):
        """Solve a factored block-tridiagonal system without Python calls.

        SciPy/LAPACK remains responsible for the pivoted factorization of each
        dense block. This kernel only fuses the many small pivot, triangular,
        forward-block and backward-block substitutions that otherwise require
        one Python-to-LAPACK call per spatial point.
        """
        n = lu_blocks.shape[0]
        nv = lu_blocks.shape[1]
        rhs_b = rhs.reshape((n, nv))
        y = np.empty_like(rhs_b)

        for i in range(n):
            # Block forward elimination: rhs_i - A_i y_{i-1}.
            for row in range(nv):
                value = rhs_b[i, row]
                if i > 0:
                    for col in range(nv):
                        value -= lower_blocks[i - 1, row, col] * y[i - 1, col]
                y[i, row] = value

            # Apply the sequential row interchanges returned by dgetrf.
            for row in range(nv):
                pivot = pivots[i, row]
                if pivot != row:
                    value = y[i, row]
                    y[i, row] = y[i, pivot]
                    y[i, pivot] = value

            # Unit-lower and upper triangular substitutions for LU_i.
            for row in range(1, nv):
                value = y[i, row]
                for col in range(row):
                    value -= lu_blocks[i, row, col] * y[i, col]
                y[i, row] = value

            for row in range(nv - 1, -1, -1):
                value = y[i, row]
                for col in range(row + 1, nv):
                    value -= lu_blocks[i, row, col] * y[i, col]
                y[i, row] = value / lu_blocks[i, row, row]

        solution = np.empty_like(rhs_b)
        for row in range(nv):
            solution[n - 1, row] = y[n - 1, row]
        for i in range(n - 2, -1, -1):
            for row in range(nv):
                value = y[i, row]
                for col in range(nv):
                    value -= cprime[i, row, col] * solution[i + 1, col]
                solution[i, row] = value

        return solution.reshape(rhs.size)
else:
    _solve_block_tridiag_lu_numba = None


def _banded_jacobian_cantera_local(fun, x: np.ndarray, problem, eps: float = 1e-5) -> sparse.csr_matrix:
    """
    Build steady Jacobian with Cantera-like local perturbations.

    This mirrors OneDim::evalJacobian:
    - base residual at x
    - perturb one variable x[col]
    - evaluate residual only for point neighborhood of col's grid point
    - write local rows into Jacobian

    ``residual_local_rows`` and ``build_local_jacobian_cache`` are defined in
    this module.
    """

    t_profile = _profile_start(problem)
    x = np.asarray(x, dtype=float)
    n_pts = int(problem.n_points)
    n_sp = int(problem.n_species)
    nv = 2 + n_sp
    n_total = n_pts * nv

    cache = build_local_jacobian_cache(x, problem)
    f0 = fun(x, problem)

    rel_perturb = float(getattr(problem, "jacobian_rel_perturb", eps))
    abs_perturb = float(getattr(problem, "jacobian_abs_perturb", 1e-10))
    threshold = float(getattr(problem, "jacobian_threshold", 0.0))

    xp = x.copy()
    max_rows_per_col = min(n_total, 3 * nv)
    capacity = max(1, n_total * max_rows_per_col)
    rows_arr = np.empty(capacity, dtype=np.int32)
    cols_arr = np.empty(capacity, dtype=np.int32)
    vals_arr = np.empty(capacity, dtype=float)
    nnz_total = 0

    def ensure_capacity(extra: int) -> None:
        nonlocal rows_arr, cols_arr, vals_arr
        required = nnz_total + int(extra)
        if required <= rows_arr.size:
            return
        new_size = max(required, rows_arr.size * 2)
        rows_new = np.empty(new_size, dtype=np.int32)
        cols_new = np.empty(new_size, dtype=np.int32)
        vals_new = np.empty(new_size, dtype=float)
        rows_new[:nnz_total] = rows_arr[:nnz_total]
        cols_new[:nnz_total] = cols_arr[:nnz_total]
        vals_new[:nnz_total] = vals_arr[:nnz_total]
        rows_arr = rows_new
        cols_arr = cols_new
        vals_arr = vals_new

    for j in range(n_pts):
        base = j * nv
        for n in range(nv):
            col = base + n
            xsave = float(x[col])

            dx = abs(xsave) * rel_perturb + abs_perturb
            if dx <= 0.0:
                dx = max(abs(rel_perturb), abs(eps), 1e-10)
            if xsave < 0.0:
                dx = -dx

            xp[col] = xsave + dx
            rdx = 1.0 / (xp[col] - xsave)

            rows, f_local = residual_local_rows(xp, problem, j, cache=cache)
            delta = f_local - f0[rows]

            if threshold > 0.0:
                keep = np.abs(delta) > threshold
                # Keep diagonal entry even if tiny (as in Cantera condition).
                kdiag = np.where(rows == col)[0]
                if kdiag.size > 0:
                    keep[int(kdiag[0])] = True
                rows_nz = rows[keep]
                vals_nz = delta[keep] * rdx
            else:
                rows_nz = rows
                vals_nz = delta * rdx

            nnz = int(rows_nz.size)
            if nnz:
                ensure_capacity(nnz)
                end = nnz_total + nnz
                rows_arr[nnz_total:end] = rows_nz
                cols_arr[nnz_total:end] = col
                vals_arr[nnz_total:end] = vals_nz
                nnz_total = end

            xp[col] = xsave

    t_sparse = _profile_start(problem)
    jmat = sparse.coo_matrix(
        (vals_arr[:nnz_total], (rows_arr[:nnz_total], cols_arr[:nnz_total])),
        shape=(n_total, n_total),
    ).tocsr()
    _profile_record(problem, "jacobian_sparse_assembly", t_sparse)
    return _profile_return(problem, "jacobian_build", t_profile, jmat)


def _banded_jacobian_batched_local(fun, x: np.ndarray, problem, eps: float = 1e-5) -> sparse.csr_matrix:
    """
    Cantera-style local Jacobian with batched perturbations per grid point.

    This keeps transport frozen like _banded_jacobian_cantera_local, but avoids
    per-column residual calls and materializes only the local stencil window.
    """
    t_profile = _profile_start(problem)
    x = np.asarray(x, dtype=float)
    n_pts = int(problem.n_points)
    n_sp = int(problem.n_species)
    nv = 2 + n_sp
    n_total = n_pts * nv

    cache = build_local_jacobian_cache(x, problem)
    f0 = fun(x, problem)

    rel_perturb = float(getattr(problem, "jacobian_rel_perturb", eps))
    abs_perturb = float(getattr(problem, "jacobian_abs_perturb", 1e-10))
    threshold = float(getattr(problem, "jacobian_threshold", 0.0))
    precomputed_thermo = None
    if bool(getattr(problem, "precompute_jacobian_thermo", False)):
        try:
            precomputed_thermo = _precompute_block_tridiag_center_thermo(
                x, problem, cache, rel_perturb, abs_perturb, eps
            )
        except Exception as exc:
            problem.last_jacobian_precompute_error = str(exc)

    max_rows_per_col = min(n_total, 3 * nv)
    capacity = max(1, n_total * max_rows_per_col)
    rows_arr = np.empty(capacity, dtype=np.int32)
    cols_arr = np.empty(capacity, dtype=np.int32)
    vals_arr = np.empty(capacity, dtype=float)
    nnz_total = 0

    def ensure_capacity(extra: int) -> None:
        nonlocal rows_arr, cols_arr, vals_arr
        required = nnz_total + int(extra)
        if required <= rows_arr.size:
            return
        new_size = max(required, rows_arr.size * 2)
        rows_new = np.empty(new_size, dtype=np.int32)
        cols_new = np.empty(new_size, dtype=np.int32)
        vals_new = np.empty(new_size, dtype=float)
        rows_new[:nnz_total] = rows_arr[:nnz_total]
        cols_new[:nnz_total] = cols_arr[:nnz_total]
        vals_new[:nnz_total] = vals_arr[:nnz_total]
        rows_arr = rows_new
        cols_arr = cols_new
        vals_arr = vals_new

    for j in range(n_pts):
        base = j * nv
        cols = np.arange(base, base + nv, dtype=np.int32)
        xsave = x[cols]
        dx = np.abs(xsave) * rel_perturb + abs_perturb
        dx[dx <= 0.0] = max(abs(rel_perturb), abs(eps), 1e-10)
        dx = np.where(xsave < 0.0, -dx, dx)

        center_thermo = None
        if precomputed_thermo is not None:
            rho_b, cp_b, omega_b, hk_b = precomputed_thermo
            center_thermo = (rho_b[j], cp_b[j], omega_b[:, j, :], hk_b[:, j, :])

        rows, f_batch = residual_local_rows_batch_perturbed(
            x, problem, j, cols, xsave + dx, cache=cache,
            center_thermo=center_thermo,
        )
        delta = f_batch - f0[rows][None, :]
        vals = delta / dx[:, None]

        if _fill_local_jacobian_block_numba is not None:
            ensure_capacity(int(cols.size * rows.size))
            nnz_total = int(_fill_local_jacobian_block_numba(
                rows_arr, cols_arr, vals_arr, int(nnz_total),
                rows, cols, np.ascontiguousarray(vals), float(threshold),
            ))
            continue

        for i, col in enumerate(cols):
            col_rows = rows
            col_vals = vals[i]
            if threshold > 0.0:
                keep = np.abs(col_vals) > threshold
                kdiag = np.where(col_rows == int(col))[0]
                if kdiag.size > 0:
                    keep[int(kdiag[0])] = True
                col_rows = col_rows[keep]
                col_vals = col_vals[keep]
            nnz = int(col_rows.size)
            if nnz:
                ensure_capacity(nnz)
                end = nnz_total + nnz
                rows_arr[nnz_total:end] = col_rows
                cols_arr[nnz_total:end] = int(col)
                vals_arr[nnz_total:end] = col_vals
                nnz_total = end

    t_sparse = _profile_start(problem)
    jmat = sparse.coo_matrix(
        (vals_arr[:nnz_total], (rows_arr[:nnz_total], cols_arr[:nnz_total])),
        shape=(n_total, n_total),
    ).tocsr()
    _profile_record(problem, "jacobian_sparse_assembly", t_sparse)
    return _profile_return(problem, "jacobian_build", t_profile, jmat)


def _banded_jacobian_lapack_local(fun, x: np.ndarray, problem, eps: float = 1e-5) -> BandedJacobian:
    """
    Build the same local finite-difference Jacobian directly in LAPACK
    general-banded storage for dgbtrf/dgbtrs.
    """
    t_profile = _profile_start(problem)
    x = np.asarray(x, dtype=float)
    n_pts = int(problem.n_points)
    n_sp = int(problem.n_species)
    nv = 2 + n_sp
    n_total = n_pts * nv

    # A perturbation at point j affects residual rows j-1:j+1. In the
    # point-interleaved state layout this spans at most about 2*nv entries
    # above/below the diagonal.
    bandwidth = int(max(1, 2 * nv))
    lower = bandwidth
    upper = bandwidth
    ab = np.zeros((2 * lower + upper + 1, n_total), dtype=float)

    cache = build_local_jacobian_cache(x, problem)
    f0 = fun(x, problem)

    rel_perturb = float(getattr(problem, "jacobian_rel_perturb", eps))
    abs_perturb = float(getattr(problem, "jacobian_abs_perturb", 1e-10))
    threshold = float(getattr(problem, "jacobian_threshold", 0.0))
    diag_band = lower + upper
    precomputed_thermo = None
    if bool(getattr(problem, "precompute_jacobian_thermo", False)):
        try:
            precomputed_thermo = _precompute_block_tridiag_center_thermo(
                x, problem, cache, rel_perturb, abs_perturb, eps
            )
        except Exception as exc:
            problem.last_jacobian_precompute_error = str(exc)

    for j in range(n_pts):
        base = j * nv
        cols = np.arange(base, base + nv, dtype=np.int32)
        xsave = x[cols]
        dx = np.abs(xsave) * rel_perturb + abs_perturb
        dx[dx <= 0.0] = max(abs(rel_perturb), abs(eps), 1e-10)
        dx = np.where(xsave < 0.0, -dx, dx)

        center_thermo = None
        if precomputed_thermo is not None:
            rho_b, cp_b, omega_b, hk_b = precomputed_thermo
            center_thermo = (rho_b[j], cp_b[j], omega_b[:, j, :], hk_b[:, j, :])

        rows, f_batch = residual_local_rows_batch_perturbed(
            x, problem, j, cols, xsave + dx, cache=cache,
            center_thermo=center_thermo,
        )
        delta = f_batch - f0[rows][None, :]
        vals = delta / dx[:, None]

        if _fill_banded_jacobian_block_numba is not None:
            _fill_banded_jacobian_block_numba(
                ab, int(lower), int(upper), rows, cols,
                np.ascontiguousarray(vals), float(threshold),
            )
            continue

        for i, col in enumerate(cols):
            col_rows = rows
            col_vals = vals[i]
            if threshold > 0.0:
                keep = np.abs(col_vals) > threshold
                kdiag = np.where(col_rows == int(col))[0]
                if kdiag.size > 0:
                    keep[int(kdiag[0])] = True
                col_rows = col_rows[keep]
                col_vals = col_vals[keep]
            band_rows = diag_band + col_rows.astype(np.int64) - int(col)
            valid = (band_rows >= 0) & (band_rows < ab.shape[0])
            ab[band_rows[valid], int(col)] = col_vals[valid]

    t_banded = _profile_start(problem)
    jmat = BandedJacobian(ab, lower=lower, upper=upper)
    _profile_record(problem, "jacobian_banded_assembly", t_banded)
    return _profile_return(problem, "jacobian_build", t_profile, jmat)


def _precompute_block_tridiag_center_thermo(
    x: np.ndarray,
    problem,
    cache: dict,
    rel_perturb: float,
    abs_perturb: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Evaluate center-node thermo/kinetics for all local perturbations."""
    backend = _jacobian_backend(problem)
    eval_tk = getattr(backend, "eval_grid_thermo_kinetics_into", None)
    use_eval_grid = False
    if eval_tk is None:
        eval_tk = getattr(backend, "eval_grid_into", None)
        use_eval_grid = eval_tk is not None
    if eval_tk is None:
        return None

    t_profile = _profile_start(problem)
    n_pts = int(cache["n_pts"])
    n_sp = int(cache["n_sp"])
    nv = int(cache["nv"])
    x_r = np.asarray(x, dtype=float).reshape(n_pts, nv)

    T_all = np.repeat(x_r[:, C_T], nv).astype(float, copy=True)
    Y_all = np.repeat(x_r[:, C_Y:].T, nv, axis=1).astype(float, copy=True)

    for j in range(n_pts):
        base = j * nv
        xsave = x_r[j]
        dx = np.abs(xsave) * rel_perturb + abs_perturb
        dx[dx <= 0.0] = max(abs(rel_perturb), abs(eps), 1e-10)
        dx = np.where(xsave < 0.0, -dx, dx)
        T_all[base + C_T] = xsave[C_T] + dx[C_T]
        for k in range(n_sp):
            Y_all[k, base + C_Y + k] = xsave[C_Y + k] + dx[C_Y + k]

    n_total = int(T_all.size)
    omega = np.empty((n_sp, n_total), dtype=float)
    hk = np.empty((n_sp, n_total), dtype=float)

    if use_eval_grid:
        rho, _, cp, _ = eval_tk(T_all, Y_all, omega, hk)
    else:
        rho, cp = eval_tk(T_all, Y_all, omega, hk)

    rho_b = np.asarray(_to_numpy(rho), dtype=float).reshape(n_pts, nv)
    cp_b = np.asarray(_to_numpy(cp), dtype=float).reshape(n_pts, nv)
    omega_b = np.asarray(_to_numpy(omega), dtype=float).reshape(n_sp, n_pts, nv)
    hk_b = np.asarray(_to_numpy(hk), dtype=float).reshape(n_sp, n_pts, nv)
    _profile_record(problem, "jacobian_precompute_thermochem", t_profile)
    return rho_b, cp_b, omega_b, hk_b


def refresh_block_tridiag_jacobian_columns(
    fun,
    x: np.ndarray,
    problem,
    reference: BlockTridiagJacobian,
    block_indices: np.ndarray | list[int],
    eps: float = 1e-5,
) -> BlockTridiagJacobian:
    """Refresh selected *column blocks* of a local block-tridiagonal Jacobian.

    This is a deliberately narrow quasi-Newton operation.  A one-dimensional
    residual column associated with node ``j`` can only write rows
    ``j-1, j, j+1``.  Re-evaluating that column block therefore updates at most
    the three corresponding block entries while retaining the tridiagonal
    structure exactly.  The caller chooses the nodes from an a-posteriori
    linearisation defect and remains responsible for refactorising the result.

    Unlike a full Jacobian build, the optional batched thermochemistry
    precomputation is intentionally not used here: evaluating it for all
    nodes would remove the benefit of a localized refresh.  The operation is
    only useful when a small subset is requested; callers should fall back to
    a complete rebuild when most blocks are active.
    """
    if not isinstance(reference, BlockTridiagJacobian):
        raise TypeError("Local refresh requires a BlockTridiagJacobian.")

    t_profile = _profile_start(problem)
    x = np.asarray(x, dtype=float)
    n_pts = int(problem.n_points)
    n_sp = int(problem.n_species)
    nv = 2 + n_sp
    expected = n_pts * nv
    if x.size != expected:
        raise ValueError("State size does not match the local Jacobian grid.")
    if reference.n_blocks != n_pts or reference.block_size != nv:
        raise ValueError("Reference Jacobian does not match the current grid.")

    requested = np.asarray(block_indices, dtype=int).ravel()
    requested = requested[(requested >= 0) & (requested < n_pts)]
    requested = np.unique(requested)
    if requested.size == 0:
        return reference.copy()

    refreshed = reference.copy()
    cache = build_local_jacobian_cache(x, problem)
    f0 = np.asarray(fun(x, problem), dtype=float)
    if f0.shape != x.shape or not np.all(np.isfinite(f0)):
        raise RuntimeError("Non-finite steady residual during local Jacobian refresh.")

    rel_perturb = float(getattr(problem, "jacobian_rel_perturb", eps))
    abs_perturb = float(getattr(problem, "jacobian_abs_perturb", 1e-10))
    threshold = float(getattr(problem, "jacobian_threshold", 0.0))

    for j in requested:
        # A column block j occupies the diagonal block j and the two adjacent
        # off-diagonal blocks.  Clear it before rewriting so thresholding does
        # not retain entries from an earlier linearisation point.
        refreshed.diag[j, :, :] = 0.0
        if j < n_pts - 1:
            refreshed.lower[j, :, :] = 0.0
        if j > 0:
            refreshed.upper[j - 1, :, :] = 0.0

        base = int(j) * nv
        cols = np.arange(base, base + nv, dtype=np.int32)
        xsave = x[cols]
        dx = np.abs(xsave) * rel_perturb + abs_perturb
        dx[dx <= 0.0] = max(abs(rel_perturb), abs(eps), 1e-10)
        dx = np.where(xsave < 0.0, -dx, dx)

        rows, f_batch = residual_local_rows_batch_perturbed(
            x, problem, int(j), cols, xsave + dx, cache=cache,
        )
        vals = (f_batch - f0[rows][None, :]) / dx[:, None]

        if _fill_block_tridiag_jacobian_block_numba is not None:
            _fill_block_tridiag_jacobian_block_numba(
                refreshed.lower,
                refreshed.diag,
                refreshed.upper,
                int(nv),
                rows,
                cols,
                np.ascontiguousarray(vals),
                float(threshold),
            )
            continue

        for i, col in enumerate(cols):
            cb = int(col) // nv
            cv = int(col) - cb * nv
            col_rows = rows
            col_vals = vals[i]
            if threshold > 0.0:
                keep = np.abs(col_vals) > threshold
                kdiag = np.where(col_rows == int(col))[0]
                if kdiag.size > 0:
                    keep[int(kdiag[0])] = True
                col_rows = col_rows[keep]
                col_vals = col_vals[keep]
            for row, value in zip(col_rows, col_vals):
                rb = int(row) // nv
                rv = int(row) - rb * nv
                if rb == cb:
                    refreshed.diag[rb, rv, cv] = value
                elif rb == cb + 1:
                    refreshed.lower[cb, rv, cv] = value
                elif rb + 1 == cb:
                    refreshed.upper[rb, rv, cv] = value

    refreshed.use_compiled_substitution = bool(
        getattr(reference, "use_compiled_substitution", True)
    )
    _profile_record(problem, "jacobian_local_refresh", t_profile)
    if getattr(problem, "_profile", None) is not None:
        entry = problem._profile.setdefault(
            "jacobian_local_refresh_blocks", {"time_s": 0.0, "count": 0}
        )
        entry["count"] += int(requested.size)
    return refreshed


def _block_tridiag_jacobian_local(fun, x: np.ndarray, problem, eps: float = 1e-5) -> BlockTridiagJacobian:
    """Build the local finite-difference Jacobian as dense block tridiagonal."""
    t_profile = _profile_start(problem)
    x = np.asarray(x, dtype=float)
    n_pts = int(problem.n_points)
    n_sp = int(problem.n_species)
    nv = 2 + n_sp

    lower_blocks = np.zeros((max(n_pts - 1, 0), nv, nv), dtype=float)
    diag_blocks = np.zeros((n_pts, nv, nv), dtype=float)
    upper_blocks = np.zeros((max(n_pts - 1, 0), nv, nv), dtype=float)

    cache = build_local_jacobian_cache(x, problem)
    f0 = fun(x, problem)

    rel_perturb = float(getattr(problem, "jacobian_rel_perturb", eps))
    abs_perturb = float(getattr(problem, "jacobian_abs_perturb", 1e-10))
    threshold = float(getattr(problem, "jacobian_threshold", 0.0))
    precomputed_thermo = None
    if bool(getattr(problem, "precompute_jacobian_thermo", False)):
        try:
            precomputed_thermo = _precompute_block_tridiag_center_thermo(
                x, problem, cache, rel_perturb, abs_perturb, eps
            )
        except Exception as exc:
            problem.last_jacobian_precompute_error = str(exc)
            precomputed_thermo = None

    for j in range(n_pts):
        base = j * nv
        cols = np.arange(base, base + nv, dtype=np.int32)
        xsave = x[cols]
        dx = np.abs(xsave) * rel_perturb + abs_perturb
        dx[dx <= 0.0] = max(abs(rel_perturb), abs(eps), 1e-10)
        dx = np.where(xsave < 0.0, -dx, dx)

        center_thermo = None
        if precomputed_thermo is not None:
            rho_b, cp_b, omega_b, hk_b = precomputed_thermo
            center_thermo = (rho_b[j], cp_b[j], omega_b[:, j, :], hk_b[:, j, :])

        rows, f_batch = residual_local_rows_batch_perturbed(
            x, problem, j, cols, xsave + dx, cache=cache, center_thermo=center_thermo
        )
        delta = f_batch - f0[rows][None, :]
        vals = delta / dx[:, None]

        if _fill_block_tridiag_jacobian_block_numba is not None:
            _fill_block_tridiag_jacobian_block_numba(
                lower_blocks,
                diag_blocks,
                upper_blocks,
                int(nv),
                rows,
                cols,
                np.ascontiguousarray(vals),
                float(threshold),
            )
            continue

        for i, col in enumerate(cols):
            cb = int(col) // nv
            cv = int(col) - cb * nv
            col_rows = rows
            col_vals = vals[i]
            if threshold > 0.0:
                keep = np.abs(col_vals) > threshold
                kdiag = np.where(col_rows == int(col))[0]
                if kdiag.size > 0:
                    keep[int(kdiag[0])] = True
                col_rows = col_rows[keep]
                col_vals = col_vals[keep]
            for row, val in zip(col_rows, col_vals):
                rb = int(row) // nv
                rv = int(row) - rb * nv
                if rb == cb:
                    diag_blocks[rb, rv, cv] = val
                elif rb == cb + 1:
                    lower_blocks[cb, rv, cv] = val
                elif rb + 1 == cb:
                    upper_blocks[rb, rv, cv] = val

    t_blocks = _profile_start(problem)
    jmat = BlockTridiagJacobian(lower_blocks, diag_blocks, upper_blocks)
    jmat.use_compiled_substitution = bool(
        getattr(problem, "use_compiled_block_substitution", True)
    )
    _profile_record(problem, "jacobian_block_assembly", t_blocks)
    return _profile_return(problem, "jacobian_build", t_profile, jmat)


# ---------------------------------------------------------------------------
#  Public Jacobian API
# ---------------------------------------------------------------------------
def banded_jacobian(fun, x: np.ndarray, problem, eps: float = 1e-5):
    """
    Build steady Jacobian.

    Modes (problem.jacobian_mode):
    - "numba_local" / "batched_local" / "native_local" (production path)
    - "cantera_local" (reference path)
    """
    mode = str(getattr(problem, "jacobian_mode", "cantera_local")).strip().lower()
    if mode in ("block_tridiag", "block_tridiagonal", "block_thomas"):
        return _block_tridiag_jacobian_local(fun, x, problem, eps=eps)
    if mode in ("banded_lapack", "lapack_banded", "native_banded"):
        return _banded_jacobian_lapack_local(fun, x, problem, eps=eps)
    if mode in ("numba_local", "batched_local", "native_local"):
        return _banded_jacobian_batched_local(fun, x, problem, eps=eps)
    return _banded_jacobian_cantera_local(fun, x, problem, eps=eps)


def update_transient(j_ss, mask: np.ndarray, rdt: float,
                     inplace: bool = False):
    """
    Apply MultiJac::updateTransient equivalent:
        J_t[n,n] = J_ss[n,n] - mask[n] * rdt
    """
    if rdt <= 0.0:
        return j_ss

    j_t = j_ss if inplace else j_ss.copy()
    d = j_t.diagonal()
    d -= np.asarray(mask, dtype=float) * float(rdt)
    j_t.setdiag(d)
    return j_t


def build_jacobian_steady(fun, x: np.ndarray, problem,
                          eps: float = 1e-5):
    j_ss = banded_jacobian(fun, x, problem, eps=eps)
    ss_diag = j_ss.diagonal().copy()
    return j_ss, ss_diag


# ---------------------------------------------------------------------------
#  Sparse linear solver wrappers
# ---------------------------------------------------------------------------
def factorize(jmat) -> dict:
    problem = getattr(jmat, "_profile_problem", None)
    t_profile = _profile_start(problem) if problem is not None else 0.0
    if isinstance(jmat, BandedJacobian):
        gbtrf = get_lapack_funcs("gbtrf", dtype=np.float64)
        ab = np.array(jmat.ab, dtype=float, order="F", copy=True)
        lu, piv, info = gbtrf(ab, int(jmat.lower), int(jmat.upper), overwrite_ab=True)
        if int(info) < 0:
            raise RuntimeError(f"LAPACK dgbtrf illegal argument {-int(info)}")
        if int(info) > 0:
            raise RuntimeError(f"LAPACK dgbtrf singular at diagonal {int(info)}")
        out = {
            "method": "banded_lapack",
            "solver": "dgbtrf",
            "lu": lu,
            "piv": piv,
            "lower": int(jmat.lower),
            "upper": int(jmat.upper),
        }
        if problem is not None:
            _profile_record(problem, "linear_factorize", t_profile)
        return out

    if isinstance(jmat, BlockTridiagJacobian):
        n = int(jmat.n_blocks)
        nv = int(jmat.block_size)
        lu_blocks: list[tuple[np.ndarray, np.ndarray]] = []
        lu_blocks_array = np.empty((n, nv, nv), dtype=np.float64)
        pivots_array = np.empty((n, nv), dtype=np.int32)
        cprime = np.zeros_like(jmat.upper)
        for i in range(n):
            mat = jmat.diag[i].copy()
            if i > 0:
                mat -= jmat.lower[i - 1] @ cprime[i - 1]
            lu = lu_factor(mat, overwrite_a=True, check_finite=False)
            lu_blocks.append(lu)
            lu_blocks_array[i] = lu[0]
            pivots_array[i] = lu[1]
            if i < n - 1:
                cprime[i] = lu_solve(lu, jmat.upper[i], check_finite=False)
        out = {
            "method": "block_tridiag",
            "solver": "block_thomas",
            "lu_blocks": lu_blocks,
            "lu_blocks_array": np.ascontiguousarray(lu_blocks_array),
            "pivots_array": np.ascontiguousarray(pivots_array),
            "lower_blocks": np.ascontiguousarray(jmat.lower, dtype=np.float64),
            "cprime": np.ascontiguousarray(cprime, dtype=np.float64),
            "n_blocks": n,
            "block_size": nv,
            "compiled_substitution": bool(
                getattr(jmat, "use_compiled_substitution", True)
            ),
        }
        if problem is not None:
            _profile_record(problem, "linear_factorize", t_profile)
        return out

    j_csc = jmat.tocsc()
    permc_spec = str(getattr(problem, "linear_permc_spec", "NATURAL"))
    diag_pivot_thresh = float(getattr(problem, "linear_diag_pivot_thresh", 1.0))
    out = {
        "method": "direct",
        "solver": splu(
            j_csc,
            permc_spec=permc_spec,
            diag_pivot_thresh=diag_pivot_thresh,
        ),
    }
    if problem is not None:
        _profile_record(problem, "linear_factorize", t_profile)
    return out


def solve_linear(linear_state, rhs: np.ndarray) -> np.ndarray:
    problem = linear_state.get("profile_problem") if isinstance(linear_state, dict) else None
    t_profile = _profile_start(problem) if problem is not None else 0.0
    rhs_array = np.asarray(rhs, dtype=float)
    rhs_multiplier = (
        linear_state.get("rhs_multiplier")
        if isinstance(linear_state, dict) else None
    )
    if rhs_multiplier is not None:
        rhs_array = rhs_array * np.asarray(rhs_multiplier, dtype=float)

    def physical_solution(value: np.ndarray) -> np.ndarray:
        solution_multiplier = (
            linear_state.get("solution_multiplier")
            if isinstance(linear_state, dict) else None
        )
        if solution_multiplier is not None:
            return np.asarray(value, dtype=float) * np.asarray(solution_multiplier, dtype=float)
        return np.asarray(value, dtype=float)

    if hasattr(linear_state, "solve") and not isinstance(linear_state, dict):
        return linear_state.solve(rhs_array)

    if isinstance(linear_state, dict) and linear_state.get("method") == "shift_reuse_block":
        """Solve a changed PTC diagonal using the previous exact LU.

        For a fixed steady Jacobian, two pseudo-transient systems differ by
        ``-(rdt_new-rdt_old) M``.  The mass mask ``M`` has high rank, so this
        is not a Sherman--Morrison--Woodbury update.  Instead, the factored
        old system is used as a right-preconditioner in a small number of
        stationary corrections.  The residual is always evaluated with the
        new block matrix.  If it does not meet the requested linear tolerance,
        the state is replaced in-place by an exact factorization of that
        matrix.  Thus the optimization can only alter an accepted inexact
        Newton correction; it can never leave a failed approximate LU active.
        """
        matrix = linear_state.get("matrix")
        reference = linear_state.get("reference_lu")
        if not isinstance(matrix, BlockTridiagJacobian) or not isinstance(reference, dict):
            raise TypeError("Invalid shifted block-LU reuse state")

        reuse_problem = linear_state.get("profile_problem")
        t_reuse = _profile_start(reuse_problem)
        max_corrections = max(0, int(linear_state.get("max_corrections", 0)))
        linear_tol = max(float(linear_state.get("linear_tolerance", 1.0e-8)), 0.0)
        rhs_norm = max(float(np.linalg.norm(rhs_array, ord=np.inf)), 1.0e-300)

        # The reference is guaranteed by the caller to be an exact, unscaled
        # block LU.  Calling this wrapper also retains its normal profiling.
        solution = solve_linear(reference, rhs_array)
        relative_residual = float("inf")
        for correction in range(max_corrections + 1):
            defect = rhs_array - matrix.matvec(solution)
            relative_residual = float(np.linalg.norm(defect, ord=np.inf) / rhs_norm)
            if np.isfinite(relative_residual) and relative_residual <= linear_tol:
                entry = reuse_problem._profile.setdefault(
                    "linear_shift_reuse_accepted", {"time_s": 0.0, "count": 0}
                ) if getattr(reuse_problem, "_profile", None) is not None else None
                if entry is not None:
                    entry["count"] += 1
                _profile_record(reuse_problem, "linear_shift_reuse", t_reuse)
                return physical_solution(solution)
            if correction < max_corrections:
                solution += solve_linear(reference, defect)

        # A failed approximation is not kept for a later RHS: factorize the
        # requested current matrix and mutate this short-lived wrapper into the
        # exact direct state.  Subsequent damping trials use that exact LU.
        exact = factorize(matrix)
        if isinstance(exact, dict):
            exact["profile_problem"] = reuse_problem
        linear_state.clear()
        linear_state.update(exact)
        if getattr(reuse_problem, "_profile", None) is not None:
            entry = reuse_problem._profile.setdefault(
                "linear_shift_reuse_fallback", {"time_s": 0.0, "count": 0}
            )
            entry["count"] += 1
        _profile_record(reuse_problem, "linear_shift_reuse", t_reuse)
        return solve_linear(linear_state, rhs_array)

    if isinstance(linear_state, dict) and linear_state.get("method") == "banded_lapack":
        gbtrs = get_lapack_funcs("gbtrs", dtype=np.float64)
        b = rhs_array.reshape(-1, 1)
        x, info = gbtrs(
            linear_state["lu"],
            int(linear_state["lower"]),
            int(linear_state["upper"]),
            b,
            linear_state["piv"],
            overwrite_b=False,
        )
        if int(info) != 0:
            raise RuntimeError(f"LAPACK dgbtrs failed with info={int(info)}")
        out = physical_solution(x[:, 0])
        if problem is not None:
            _profile_record(problem, "linear_solve", t_profile)
        return out
    if isinstance(linear_state, dict) and linear_state.get("method") == "block_tridiag":
        n = int(linear_state["n_blocks"])
        nv = int(linear_state["block_size"])
        rhs_array = np.ascontiguousarray(rhs_array, dtype=np.float64)
        if (
            _solve_block_tridiag_lu_numba is not None
            and rhs_array.ndim == 1
            and bool(linear_state.get("compiled_substitution", True))
        ):
            out = _solve_block_tridiag_lu_numba(
                linear_state["lu_blocks_array"],
                linear_state["pivots_array"],
                linear_state["lower_blocks"],
                linear_state["cprime"],
                rhs_array,
            )
            out = physical_solution(out)
            if problem is not None:
                _profile_record(problem, "linear_solve", t_profile)
            return out
        rhs_b = rhs_array.reshape(n, nv)
        y = np.empty_like(rhs_b)
        lu_blocks = linear_state["lu_blocks"]
        lower_blocks = linear_state["lower_blocks"]
        cprime = linear_state["cprime"]

        y[0] = lu_solve(lu_blocks[0], rhs_b[0], check_finite=False)
        for i in range(1, n):
            y[i] = lu_solve(
                lu_blocks[i],
                rhs_b[i] - lower_blocks[i - 1] @ y[i - 1],
                check_finite=False,
            )

        x_b = np.empty_like(rhs_b)
        x_b[-1] = y[-1]
        for i in range(n - 2, -1, -1):
            x_b[i] = y[i] - cprime[i] @ x_b[i + 1]
        out = physical_solution(x_b.ravel())
        if problem is not None:
            _profile_record(problem, "linear_solve", t_profile)
        return out
    out = physical_solution(linear_state["solver"].solve(rhs_array))
    if problem is not None:
        _profile_record(problem, "linear_solve", t_profile)
    return out
