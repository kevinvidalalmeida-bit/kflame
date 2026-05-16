"""
residual.py – Residual acoplado para llama libre premezclada 1-D.

UNA SOLA funcion `residual(...)` para estacionario, Backward Euler y BDF2.
El termino temporal usa rdt = 1/dt y la mascara de variables diferenciales.

Esto replica exactamente el comportamiento de Flow1D::eval() en Cantera:
el mismo método calcula siempre el residual, y si rdt != 0, añade el
término de derivada temporal usando la solución previa almacenada.

Layout del estado y del residual: POR-PUNTO ENTRELAZADO.
  x = [u0, T0, Y0..YK at pt0,  u1, T1, Y0..YK at pt1, ...]
  F = misma estructura
"""
from __future__ import annotations
import numpy as np

from state import unpack_state, build_transient_mask, C_U, C_T, C_Y


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
    u, T, Y = unpack_state(x, n_pts, n_sp)

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
        problem.last_residual_error = str(exc)
        return np.full(x.size, 1.0e20)

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
        problem.last_residual_error = str(exc)
        return np.full(x.size, 1.0e20)

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
            # Free flame: replace continuity with T = T_fixed (anchor)
            F[b + C_U] = T[j] - float(problem.T_fixed_point)
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
        return np.full(x.size, 1.0e20)

    return F


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
#  Reporte de normas por bloque
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


