"""Residual and Jacobian assembly for the 1-D free-flame solver."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from state import C_T, C_U, C_Y, build_transient_mask, unpack_state


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


def _device_residual_enabled(problem) -> bool:
    backend = getattr(problem, "backend", None)
    return (
        backend is not None
        and bool(getattr(backend, "use_gpu", False))
        and hasattr(backend, "eval_grid_device")
        and hasattr(backend, "eval_faces_device")
        and hasattr(backend, "_to_host")
    )


def _corrected_flux_device(xp, Y_L, Y_R, rho_f, D_f, dz, W, W_mix_f, basis: str):
    if basis in ("molar", "mole"):
        W_mix_L = 1.0 / xp.sum(Y_L / W[:, None], axis=0)
        W_mix_R = 1.0 / xp.sum(Y_R / W[:, None], axis=0)
        X_L = Y_L * (W_mix_L[None, :] / W[:, None])
        X_R = Y_R * (W_mix_R[None, :] / W[:, None])
        dphi = (X_R - X_L) / dz[None, :]
        J_star = -rho_f[None, :] * (W[:, None] / W_mix_f[None, :]) * D_f * dphi
    else:
        dphi = (Y_R - Y_L) / dz[None, :]
        J_star = -rho_f[None, :] * D_f * dphi
    return J_star - Y_L * xp.sum(J_star, axis=0, keepdims=True)


def _residual_device(
    x: np.ndarray,
    problem,
    rdt: float = 0.0,
    x_old: np.ndarray | None = None,
    x_older: np.ndarray | None = None,
    transient_order: int = 1,
) -> np.ndarray:
    backend = problem.backend
    xp = backend.xp
    n_pts = int(problem.n_points)
    n_sp = int(problem.n_species)
    nv = 2 + n_sp
    basis = str(getattr(problem, "flux_gradient_basis", "molar")).lower()

    x_dev = xp.asarray(x, dtype=float)
    x_r = x_dev.reshape(n_pts, nv)
    u = x_r[:, C_U]
    T = x_r[:, C_T]
    Y = x_r[:, C_Y:].T
    z = xp.asarray(problem.z, dtype=float)
    W = backend.W_dev
    invW = backend.invW_dev

    rho, _Dm_nodes, cp_n, lam_n, omega, hk_n = backend.eval_grid_device(T, Y)

    T_face = 0.5 * (T[:-1] + T[1:])
    Y_face = 0.5 * (Y[:, :-1] + Y[:, 1:])
    rho_face, D_face, lam_face, W_mix_face = backend.eval_faces_device(T_face, Y_face)
    dz_face = z[1:] - z[:-1]
    flux = _corrected_flux_device(xp, Y[:, :-1], Y[:, 1:], rho_face, D_face, dz_face, W, W_mix_face, basis)

    F = xp.empty((n_pts, nv), dtype=float)

    dz0 = z[1] - z[0]
    F[0, C_U] = -(rho[1] * u[1] - rho[0] * u[0]) / dz0
    T_prof = getattr(problem, "T_profile_fixed", None)
    solve_energy = bool(problem.solve_energy)
    if solve_energy:
        F[0, C_T] = T[0] - float(problem.T_in)
    else:
        T_prof_dev = xp.asarray(T_prof, dtype=float) if T_prof is not None else None
        F[0, C_T] = T[0] - (T_prof_dev[0] if T_prof_dev is not None else float(problem.T_in))

    mdot_in = rho[0] * u[0]
    Y_in = xp.asarray(problem.Y_in, dtype=float)
    left_species = (-(flux[:, 0] + mdot_in * Y[:, 0]) + mdot_in * Y_in)
    k_idx = xp.arange(n_sp)
    k_exc = xp.argmax(Y[:, 0])
    F[0, C_Y:C_Y + n_sp] = xp.where(k_idx == k_exc, 1.0 - xp.sum(Y[:, 0]), left_species)

    if n_pts > 2:
        j = xp.arange(1, n_pts - 1)
        dzm = z[1:-1] - z[:-2]
        dzp = z[2:] - z[1:-1]
        dz2 = z[2:] - z[:-2]
        fm = flux[:, :-1]
        fp = flux[:, 1:]

        j_fixed = problem.j_fixed
        cont_forward = -(rho[2:] * u[2:] - rho[1:-1] * u[1:-1]) / dzp
        if j_fixed is None:
            F[1:-1, C_U] = cont_forward
        else:
            cont_backward = -(rho[1:-1] * u[1:-1] - rho[:-2] * u[:-2]) / dzm
            if solve_energy:
                cont_anchor = T[1:-1] - float(problem.T_fixed_point)
            else:
                cont_anchor = rho[1:-1] * u[1:-1] - rho[0] * 0.3
            F[1:-1, C_U] = xp.where(
                j == int(j_fixed),
                cont_anchor,
                xp.where(j > int(j_fixed), cont_backward, cont_forward),
            )

        rho_j = rho[1:-1]
        rho_u_j = rho_j * u[1:-1]
        jloc = xp.where(u[1:-1] > 0.0, j, j + 1)
        jloc_m = jloc - 1
        dz_up = z[jloc] - z[jloc_m]

        if solve_energy:
            dTdz = (T[jloc] - T[jloc_m]) / dz_up
            cond = -2.0 * (
                lam_face[1:] * (T[2:] - T[1:-1]) / dzp
                - lam_face[:-1] * (T[1:-1] - T[:-2]) / dzm
            ) / dz2
            dhk_dz = (hk_n[:, jloc] - hk_n[:, jloc_m]) / dz_up[None, :]
            flx = 0.5 * (fm + fp)
            en_sum = (
                xp.sum(hk_n[:, 1:-1] * omega[:, 1:-1] * invW[:, None], axis=0)
                + xp.sum(flx * dhk_dz * invW[:, None], axis=0)
            )
            F[1:-1, C_T] = (-cp_n[1:-1] * rho_u_j * dTdz - cond - en_sum) / (rho_j * cp_n[1:-1])
        else:
            T_prof_dev = xp.asarray(T_prof, dtype=float) if T_prof is not None else None
            F[1:-1, C_T] = T[1:-1] - (T_prof_dev[1:-1] if T_prof_dev is not None else float(problem.T_in))

        dYdz = (Y[:, jloc] - Y[:, jloc_m]) / dz_up[None, :]
        conv = rho_u_j[None, :] * dYdz
        diff = 2.0 * (fp - fm) / dz2[None, :]
        F[1:-1, C_Y:C_Y + n_sp] = ((omega[:, 1:-1] - conv - diff) / rho_j[None, :]).T

    F[-1, C_U] = rho[-1] * u[-1] - rho[-2] * u[-2]
    if solve_energy:
        F[-1, C_T] = T[-1] - T[-2]
    else:
        T_prof_dev = xp.asarray(T_prof, dtype=float) if T_prof is not None else None
        F[-1, C_T] = T[-1] - (T_prof_dev[-1] if T_prof_dev is not None else T[-2])
    right_species = Y[:, -1] - Y[:, -2]
    k_exc = xp.argmax(Y[:, -1])
    F[-1, C_Y:C_Y + n_sp] = xp.where(k_idx == k_exc, 1.0 - xp.sum(Y[:, -1]), right_species)

    F_flat = F.ravel()
    if rdt > 0.0 and x_old is not None:
        mask = xp.asarray(
            build_transient_mask(n_pts, n_sp, solve_energy=solve_energy),
            dtype=float,
        )
        x_old_dev = xp.asarray(x_old, dtype=float)
        use_bdf2 = int(transient_order) >= 2 and x_older is not None
        if use_bdf2:
            x_older_dev = xp.asarray(x_older, dtype=float)
            F_flat -= mask * (1.5 * rdt * x_dev - 2.0 * rdt * x_old_dev + 0.5 * rdt * x_older_dev)
        else:
            F_flat -= mask * rdt * (x_dev - x_old_dev)

    return np.asarray(backend._to_host(F_flat), dtype=float)

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


# ---------------------------------------------------------------------------
#  Función residual principal
# ---------------------------------------------------------------------------
def residual(
    x: np.ndarray,
    problem,
    rdt: float = 0.0,
    x_old: np.ndarray | None = None,
    x_older: np.ndarray | None = None,
    transient_order: int = 1,
) -> np.ndarray:
    """
    Residual acoplado completo.

    Parametros
    ----------
    x      : vector de estado actual, layout por-punto entrelazado
    problem: FreeFlameProblem
    rdt    : 1/dt  (0.0 = modo estacionario; > 0 = modo transitorio)
    x_old  : solucion en el paso n   (requerida si rdt > 0)
    x_older: solucion en el paso n-1 (requerida para BDF2)
    transient_order:
      1 -> Backward Euler
      2 -> BDF2
    """
    t_profile = _profile_start(problem)
    if _device_residual_enabled(problem):
        try:
            t_device = _profile_start(problem)
            F_dev = _residual_device(
                x, problem, rdt=rdt, x_old=x_old,
                x_older=x_older, transient_order=transient_order,
            )
            _profile_record(problem, "residual_device", t_device)
            if not np.all(np.isfinite(F_dev)):
                F_dev = np.full(np.asarray(x, dtype=float).size, 1.0e20)
            return _profile_return(problem, "residual_full", t_profile, F_dev)
        except Exception as exc:
            problem.last_residual_error = f"GPU residual fallback: {exc}"
            if bool(getattr(problem, "debug_residual_errors", False)):
                print(problem.last_residual_error)

    x = np.asarray(x, dtype=float)
    n_pts = int(problem.n_points)
    n_sp = int(problem.n_species)
    nv = 2 + n_sp
    backend = problem.backend
    z = problem.z
    W = backend.W
    invW = backend.invW
    basis = str(getattr(problem, "flux_gradient_basis", "molar")).lower()

    # ------------------------------------------------------------------
    #  1. Propiedades nodales
    # ------------------------------------------------------------------
    u, T, Y = _state_views(x, n_pts, n_sp)

    rho = np.empty(n_pts)
    cp_n = np.empty(n_pts)
    lam_n = np.empty(n_pts)
    omega = np.empty((n_sp, n_pts))
    hk_n = np.empty((n_sp, n_pts))

    try:
        eval_node_into = backend.eval_node_into
        for j in range(n_pts):
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
    rho_face = np.empty(n_pts - 1)
    lam_face = np.empty(n_pts - 1)

    try:
        eval_face = backend.eval_midpoint_full_transport
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
            dYdz = (Y[k, jloc] - Y[k, jloc - 1]) / dz_up
            conv = rho_u_j * dYdz
            diff = 2.0 * (fp[k] - fm[k]) / dz2
            F[b + C_Y + k] = (omega[k, j] - conv - diff) / rho[j]

    # --- j=N-1 (outlet / right boundary) ---
    _right_bc(F, u, T, Y, rho, flux[:, -1], problem, nv, n_sp, n_pts)

    # ------------------------------------------------------------------
    #  4. Término transitorio  (idéntico a "rsd[n] -= rdt*(x-x_old)" en Cantera)
    # ------------------------------------------------------------------
    if rdt > 0.0 and x_old is not None:
        x_old = np.asarray(x_old, dtype=float)
        mask = build_transient_mask(n_pts, n_sp,
                                    solve_energy=bool(problem.solve_energy))
        use_bdf2 = int(transient_order) >= 2 and x_older is not None

        if use_bdf2:
            x_older = np.asarray(x_older, dtype=float)
            # BDF2:
            #   (3*x - 4*x_n + x_nm1) / (2*dt), con rdt = 1/dt.
            F -= mask * (1.5 * rdt * x - 2.0 * rdt * x_old + 0.5 * rdt * x_older)
        else:
            # Backward Euler:
            #   (x - x_n) / dt
            F -= mask * rdt * (x - x_old)

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

    # Species: zero-gradient for all except excess (normalization)
    k_exc = int(np.argmax(Y[:, -1]))
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
    dTdz = (T[jloc] - T[jloc - 1]) / dz_up

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
def residual_block_report(x: np.ndarray, problem) -> dict:
    F = residual(x, problem)
    nv = 2 + problem.n_species
    n_pts = problem.n_points

    R_left = F[:nv]
    R_right = F[-nv:]
    norms_int = [float(np.linalg.norm(F[j*nv:(j+1)*nv], ord=np.inf))
                 for j in range(1, n_pts - 1)]

    return {
        "left_inf":      float(np.linalg.norm(R_left, ord=np.inf)),
        "right_inf":     float(np.linalg.norm(R_right, ord=np.inf)),
        "interior_max":  float(max(norms_int)) if norms_int else 0.0,
        "interior_mean": float(sum(norms_int) / len(norms_int)) if norms_int else 0.0,
        "total_inf":     float(np.linalg.norm(F, ord=np.inf)),
    }




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

    backend = problem.backend
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

    if hasattr(backend, "eval_grid_into"):
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
    eval_node_into = problem.backend.eval_node_into
    j = j_center
    if n0 <= j <= n1:
        jl = j - n0
        om = np.empty(n_sp, dtype=float)
        hk = np.empty(n_sp, dtype=float)
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
                dTdz = (T[jloc] - T[jloc - 1]) / dz_up

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
            dYdz = (Y[:, jloc] - Y[:, jloc - 1]) / dz_up
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
        if hasattr(problem.backend, "eval_grid_into"):
            rho_j, _, cp_j, _ = problem.backend.eval_grid_into(
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
            for ib in range(n_batch):
                om = np.empty(n_sp, dtype=float)
                hk = np.empty(n_sp, dtype=float)
                rhoj, _, cpj, _ = problem.backend.eval_node_into(T[ib, j_center], Y[ib, :, j_center], om, hk)
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
                dTdz = (T[batch_idx, jloc] - T[batch_idx, jloc_m]) / dz_up
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
            dYdz = (Y_jloc - Y_jloc_m) / dz_up[:, None]
            conv = rho_u_j[:, None] * dYdz
            diff = 2.0 * (fp - fm) / dz2
            block[:, C_Y:C_Y + n_sp] = (om_j - conv - diff) / rho_j[:, None]

        out_i += nv

    return _profile_return(problem, "residual_local_batch", t_profile, (rows_out, vals_out))


# --- FIN DE RESIDUAL.PY, INICIO DE JACOBIAN.PY ---

"""
jacobian.py - Sparse Jacobian builders and transient diagonal update.

Default assembly follows the Cantera OneDim::evalJacobian pattern:
- perturb one state variable at a time
- evaluate residual only on local rows (j-1, j, j+1)
- fill a sparse Jacobian column from local finite differences
"""




# ---------------------------------------------------------------------------
#  Cantera-style local finite-difference Jacobian
# ---------------------------------------------------------------------------
def _banded_jacobian_cantera_local(fun, x: np.ndarray, problem, eps: float = 1e-5) -> sparse.csr_matrix:
    """
    Build steady Jacobian with Cantera-like local perturbations.

    This mirrors OneDim::evalJacobian:
    - base residual at x
    - perturb one variable x[col]
    - evaluate residual only for point neighborhood of col's grid point
    - write local rows into Jacobian    # Las funciones residual_local_rows y build_local_jacobian_cache están en este mismo archivo.
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


def _banded_jacobian_gpu_local(fun, x: np.ndarray, problem, eps: float = 1e-5) -> sparse.csr_matrix:
    """
    Cantera-style local Jacobian with batched perturbations per grid point.

    This keeps transport frozen like _banded_jacobian_cantera_local, but reduces
    per-column GPU launch/copy overhead by evaluating all variables at a point
    in one backend batch.
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

        x_batch = np.broadcast_to(x, (nv, n_total)).copy()
        x_batch[np.arange(nv), cols] = xsave + dx
        rows, f_batch = residual_local_rows_batch(x_batch, problem, j, cache=cache)
        delta = f_batch - f0[rows][None, :]
        vals = delta / dx[:, None]

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


def _residual_device_batch_steady(x_batch: np.ndarray, problem) -> np.ndarray:
    backend = problem.backend
    xp = backend.xp
    x_dev = xp.asarray(x_batch, dtype=float)
    n_batch = int(x_dev.shape[0])
    n_pts = int(problem.n_points)
    n_sp = int(problem.n_species)
    nv = 2 + n_sp
    basis = str(getattr(problem, "flux_gradient_basis", "molar")).lower()

    x_r = x_dev.reshape(n_batch, n_pts, nv)
    u = x_r[:, :, C_U]
    T = x_r[:, :, C_T]
    Y = x_r[:, :, C_Y:].transpose(0, 2, 1)
    z = xp.asarray(problem.z, dtype=float)
    W = backend.W_dev
    invW = backend.invW_dev

    T_flat = T.reshape(n_batch * n_pts)
    Y_flat = Y.transpose(1, 0, 2).reshape(n_sp, n_batch * n_pts)
    rho_flt, _Dm_nodes, cp_flt, lam_flt, omega_flt, hk_flt = backend.eval_grid_device(T_flat, Y_flat)
    rho = rho_flt.reshape(n_batch, n_pts)
    cp_n = cp_flt.reshape(n_batch, n_pts)
    omega = omega_flt.reshape(n_sp, n_batch, n_pts).transpose(1, 0, 2)
    hk_n = hk_flt.reshape(n_sp, n_batch, n_pts).transpose(1, 0, 2)

    n_faces = n_pts - 1
    T_face = 0.5 * (T[:, :-1] + T[:, 1:])
    Y_face = 0.5 * (Y[:, :, :-1] + Y[:, :, 1:])
    T_face_flat = T_face.reshape(n_batch * n_faces)
    Y_face_flat = Y_face.transpose(1, 0, 2).reshape(n_sp, n_batch * n_faces)
    rho_face_flt, D_face_flt, lam_face_flt, Wmix_face_flt = backend.eval_faces_device(T_face_flat, Y_face_flat)
    rho_face = rho_face_flt.reshape(n_batch, n_faces)
    D_face = D_face_flt.reshape(n_sp, n_batch, n_faces).transpose(1, 0, 2)
    lam_face = lam_face_flt.reshape(n_batch, n_faces)
    Wmix_face = Wmix_face_flt.reshape(n_batch, n_faces)

    YL = Y[:, :, :-1]
    YR = Y[:, :, 1:]
    dz_face = z[1:] - z[:-1]
    if basis in ("molar", "mole"):
        Wb = W[None, :, None]
        Wmix_L = 1.0 / xp.sum(YL / Wb, axis=1)
        Wmix_R = 1.0 / xp.sum(YR / Wb, axis=1)
        X_L = YL * (Wmix_L[:, None, :] / Wb)
        X_R = YR * (Wmix_R[:, None, :] / Wb)
        dphi = (X_R - X_L) / dz_face[None, None, :]
        flux_star = (
            -rho_face[:, None, :]
            * (Wb / Wmix_face[:, None, :])
            * D_face
            * dphi
        )
    else:
        dphi = (YR - YL) / dz_face[None, None, :]
        flux_star = -rho_face[:, None, :] * D_face * dphi
    flux = flux_star - YL * xp.sum(flux_star, axis=1, keepdims=True)

    F = xp.empty((n_batch, n_pts, nv), dtype=float)
    solve_energy = bool(problem.solve_energy)
    T_prof = getattr(problem, "T_profile_fixed", None)
    T_prof_dev = xp.asarray(T_prof, dtype=float) if T_prof is not None else None

    dz0 = z[1] - z[0]
    F[:, 0, C_U] = -(rho[:, 1] * u[:, 1] - rho[:, 0] * u[:, 0]) / dz0
    if solve_energy:
        F[:, 0, C_T] = T[:, 0] - float(problem.T_in)
    else:
        F[:, 0, C_T] = T[:, 0] - (T_prof_dev[0] if T_prof_dev is not None else float(problem.T_in))

    Y_in = xp.asarray(problem.Y_in, dtype=float)
    mdot_in = rho[:, 0] * u[:, 0]
    left_species = (-(flux[:, :, 0] + mdot_in[:, None] * Y[:, :, 0]) + mdot_in[:, None] * Y_in[None, :])
    k_idx = xp.arange(n_sp)[None, :]
    k_exc = xp.argmax(Y[:, :, 0], axis=1)[:, None]
    F[:, 0, C_Y:C_Y + n_sp] = xp.where(k_idx == k_exc, 1.0 - xp.sum(Y[:, :, 0], axis=1)[:, None], left_species)

    if n_pts > 2:
        j = xp.arange(1, n_pts - 1)
        dzm = z[1:-1] - z[:-2]
        dzp = z[2:] - z[1:-1]
        dz2 = z[2:] - z[:-2]
        fm = flux[:, :, :-1]
        fp = flux[:, :, 1:]

        cont_forward = -(rho[:, 2:] * u[:, 2:] - rho[:, 1:-1] * u[:, 1:-1]) / dzp[None, :]
        j_fixed = problem.j_fixed
        if j_fixed is None:
            F[:, 1:-1, C_U] = cont_forward
        else:
            cont_backward = -(rho[:, 1:-1] * u[:, 1:-1] - rho[:, :-2] * u[:, :-2]) / dzm[None, :]
            if solve_energy:
                cont_anchor = T[:, 1:-1] - float(problem.T_fixed_point)
            else:
                cont_anchor = rho[:, 1:-1] * u[:, 1:-1] - rho[:, 0:1] * 0.3
            F[:, 1:-1, C_U] = xp.where(
                j[None, :] == int(j_fixed),
                cont_anchor,
                xp.where(j[None, :] > int(j_fixed), cont_backward, cont_forward),
            )

        rho_j = rho[:, 1:-1]
        rho_u_j = rho_j * u[:, 1:-1]
        jloc = xp.where(u[:, 1:-1] > 0.0, j[None, :], j[None, :] + 1)
        jloc_m = jloc - 1
        dz_up = z[jloc] - z[jloc_m]

        if solve_energy:
            T_jloc = xp.take_along_axis(T, jloc, axis=1)
            T_jloc_m = xp.take_along_axis(T, jloc_m, axis=1)
            dTdz = (T_jloc - T_jloc_m) / dz_up
            cond = -2.0 * (
                lam_face[:, 1:] * (T[:, 2:] - T[:, 1:-1]) / dzp[None, :]
                - lam_face[:, :-1] * (T[:, 1:-1] - T[:, :-2]) / dzm[None, :]
            ) / dz2[None, :]
            hk_jloc = xp.take_along_axis(hk_n, jloc[:, None, :], axis=2)
            hk_jloc_m = xp.take_along_axis(hk_n, jloc_m[:, None, :], axis=2)
            dhk_dz = (hk_jloc - hk_jloc_m) / dz_up[:, None, :]
            flx = 0.5 * (fm + fp)
            en_sum = (
                xp.sum(hk_n[:, :, 1:-1] * omega[:, :, 1:-1] * invW[None, :, None], axis=1)
                + xp.sum(flx * dhk_dz * invW[None, :, None], axis=1)
            )
            F[:, 1:-1, C_T] = (
                -cp_n[:, 1:-1] * rho_u_j * dTdz - cond - en_sum
            ) / (rho_j * cp_n[:, 1:-1])
        else:
            F[:, 1:-1, C_T] = T[:, 1:-1] - (
                T_prof_dev[None, 1:-1] if T_prof_dev is not None else float(problem.T_in)
            )

        Y_jloc = xp.take_along_axis(Y, jloc[:, None, :], axis=2)
        Y_jloc_m = xp.take_along_axis(Y, jloc_m[:, None, :], axis=2)
        dYdz = (Y_jloc - Y_jloc_m) / dz_up[:, None, :]
        conv = rho_u_j[:, None, :] * dYdz
        diff = 2.0 * (fp - fm) / dz2[None, None, :]
        F[:, 1:-1, C_Y:C_Y + n_sp] = (
            (omega[:, :, 1:-1] - conv - diff) / rho_j[:, None, :]
        ).transpose(0, 2, 1)

    F[:, -1, C_U] = rho[:, -1] * u[:, -1] - rho[:, -2] * u[:, -2]
    if solve_energy:
        F[:, -1, C_T] = T[:, -1] - T[:, -2]
    else:
        F[:, -1, C_T] = T[:, -1] - (T_prof_dev[-1] if T_prof_dev is not None else T[:, -2])
    right_species = Y[:, :, -1] - Y[:, :, -2]
    k_exc = xp.argmax(Y[:, :, -1], axis=1)[:, None]
    F[:, -1, C_Y:C_Y + n_sp] = xp.where(k_idx == k_exc, 1.0 - xp.sum(Y[:, :, -1], axis=1)[:, None], right_species)

    return np.asarray(backend._to_host(F.reshape(n_batch, n_pts * nv)), dtype=float)


def _banded_jacobian_gpu_batch(fun, x: np.ndarray, problem, eps: float = 1e-5) -> sparse.csr_matrix:
    t_profile = _profile_start(problem)
    x = np.asarray(x, dtype=float)
    n_pts = int(problem.n_points)
    n_sp = int(problem.n_species)
    nv = 2 + n_sp
    n_total = n_pts * nv

    f0 = fun(x, problem)
    rel_perturb = float(getattr(problem, "jacobian_rel_perturb", eps))
    abs_perturb = float(getattr(problem, "jacobian_abs_perturb", 1e-10))
    threshold = float(getattr(problem, "jacobian_threshold", 0.0))

    row_offsets = np.arange(nv, dtype=np.int32)
    local_rows = []
    for j in range(n_pts):
        p0 = max(0, j - 1)
        p1 = min(n_pts - 1, j + 1)
        blocks = np.arange(p0, p1 + 1, dtype=np.int32)[:, None] * nv
        local_rows.append((blocks + row_offsets[None, :]).ravel())

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

        x_batch = np.broadcast_to(x, (nv, n_total)).copy()
        x_batch[np.arange(nv), cols] = xsave + dx
        f_batch = _residual_device_batch_steady(x_batch, problem)

        rows = local_rows[j]
        delta = f_batch[:, rows] - f0[rows][None, :]
        vals = delta / dx[:, None]

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


# ---------------------------------------------------------------------------
#  Legacy 3-coloring Jacobian (fallback / debug)
# ---------------------------------------------------------------------------
def _banded_jacobian_coloring(fun, x: np.ndarray, problem, eps: float = 1e-8) -> sparse.csr_matrix:
    x = np.asarray(x, dtype=float)
    n_pts = int(problem.n_points)
    n_sp = int(problem.n_species)
    nv = 2 + n_sp
    n_total = n_pts * nv

    f0 = fun(x, problem)
    point_rows = [
        np.arange(max(0, j - 1) * nv, (min(n_pts - 1, j + 1) + 1) * nv, dtype=np.int32)
        for j in range(n_pts)
    ]

    rows_list: list[int] = []
    cols_list: list[int] = []
    vals_list: list[float] = []
    xp = x.copy()

    for v_idx in range(nv):
        for color in range(3):
            pts = list(range(color, n_pts, 3))
            xp[:] = x
            dx_vals: list[float] = []
            cols: list[int] = []

            for j in pts:
                col_idx = j * nv + v_idx
                xval = x[col_idx]
                rel_pert = float(getattr(problem, "jacobian_rel_perturb", eps))
                abs_pert = float(getattr(problem, "jacobian_abs_perturb", 1e-10))
                dx = abs(xval) * rel_pert + abs_pert
                if xval < 0:
                    dx = -dx
                xp[col_idx] += dx
                dx_vals.append(dx)
                cols.append(col_idx)

            fp = fun(xp, problem)
            df = fp - f0

            for idx, j in enumerate(pts):
                rows = point_rows[j]
                vals = df[rows] / dx_vals[idx]
                nz = np.abs(vals) > 1e-30
                if not np.any(nz):
                    continue
                rows_nz = rows[nz]
                vals_nz = vals[nz]
                nnz = int(rows_nz.size)
                rows_list.extend(rows_nz.tolist())
                cols_list.extend([cols[idx]] * nnz)
                vals_list.extend(vals_nz.tolist())

    jmat = sparse.coo_matrix((vals_list, (rows_list, cols_list)), shape=(n_total, n_total)).tocsr()
    return jmat


# ---------------------------------------------------------------------------
#  Public Jacobian API
# ---------------------------------------------------------------------------
def banded_jacobian(fun, x: np.ndarray, problem, eps: float = 1e-5) -> sparse.csr_matrix:
    """
    Build steady Jacobian.

    Modes (problem.jacobian_mode):
    - "cantera_local" (default)
    - "gpu_local"
    - "gpu_batch"
    - "coloring"
    """
    mode = str(getattr(problem, "jacobian_mode", "cantera_local")).strip().lower()
    if mode in ("gpu_local", "device_local"):
        if not _device_residual_enabled(problem):
            raise RuntimeError("jacobian_mode='gpu_local' requiere backend nativo con CuPy activo.")
        return _banded_jacobian_gpu_local(fun, x, problem, eps=eps)
    if mode in ("gpu", "gpu_batch", "device_batch"):
        if not _device_residual_enabled(problem):
            raise RuntimeError("jacobian_mode='gpu_batch' requiere backend nativo con CuPy activo.")
        if not bool(getattr(problem, "allow_experimental_gpu_jacobian", False)):
            raise RuntimeError(
                "jacobian_mode='gpu_batch' es experimental; active "
                "SolveOptions(experimental_gpu_jacobian=True) solo para pruebas."
            )
        return _banded_jacobian_gpu_batch(fun, x, problem, eps=eps)
    if mode in ("coloring", "3color", "legacy"):
        return _banded_jacobian_coloring(fun, x, problem, eps=eps)
    return _banded_jacobian_cantera_local(fun, x, problem, eps=eps)


def update_transient(j_ss: sparse.csr_matrix, mask: np.ndarray, rdt: float,
                     inplace: bool = False) -> sparse.csr_matrix:
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
                          eps: float = 1e-5) -> tuple[sparse.csr_matrix, np.ndarray]:
    j_ss = banded_jacobian(fun, x, problem, eps=eps)
    ss_diag = j_ss.diagonal().copy()
    return j_ss, ss_diag


def build_jacobian_transient(fun, x: np.ndarray, problem, rdt: float,
                             eps: float = 1e-5) -> sparse.csr_matrix:
    j_ss, _ = build_jacobian_steady(fun, x, problem, eps=eps)
    mask = build_transient_mask(problem.n_points, problem.n_species,
                                solve_energy=bool(problem.solve_energy))
    return update_transient(j_ss, mask, rdt)


# ---------------------------------------------------------------------------
#  Sparse linear solver wrappers
# ---------------------------------------------------------------------------
def factorize(jmat: sparse.csr_matrix) -> dict:
    problem = getattr(jmat, "_profile_problem", None)
    t_profile = _profile_start(problem) if problem is not None else 0.0
    j_csc = jmat.tocsc()
    out = {
        "method": "direct",
        "solver": splu(j_csc, permc_spec="COLAMD"),
    }
    if problem is not None:
        _profile_record(problem, "linear_factorize", t_profile)
    return out


def solve_linear(linear_state, rhs: np.ndarray) -> np.ndarray:
    problem = linear_state.get("profile_problem") if isinstance(linear_state, dict) else None
    t_profile = _profile_start(problem) if problem is not None else 0.0
    if hasattr(linear_state, "solve") and not isinstance(linear_state, dict):
        return linear_state.solve(rhs)
    out = linear_state["solver"].solve(rhs)
    if problem is not None:
        _profile_record(problem, "linear_solve", t_profile)
    return out
