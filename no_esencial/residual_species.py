"""
residual_species.py – Residual vectorizado de especies (frozen T, mdot).

Modelo resuelto:
----------------
Con T(z) y mdot congelados, para cada especie independiente k:

    R_k = [ mdot * dY_k/dz + dJ_k/dz - omega_k ] / rho

Condiciones de borde (Cantera Flow1D):
--------------------------------------
IZQUIERDA (inlet):
    -(J_k(face0) + mdot * Y_k(0)) + mdot * Y_k,in = 0

DERECHA (outlet):
    Y_k(N-1) - Y_k(N-2) = 0   (gradiente nulo)

FIX:
  * BUG 3 (CRÍTICO): arrays D y omega no estaban inicializados antes del
    loop en _eval_all_properties → NameError en tiempo de ejecución.
    Añadidos np.empty(...) antes del loop.
"""

from __future__ import annotations

import numpy as np

from state import unpack_species


# ---------------------------------------------------------------------------
#  Evaluación batch de propiedades con Cantera
# ---------------------------------------------------------------------------
def _eval_all_properties(T: np.ndarray, Y: np.ndarray, backend, n_pts: int):
    """
    Evalúa rho, D y omega_mass en todos los nodos.

    Retorna
    -------
    rho   : (n_pts,)
    D     : (n_species, n_pts)
    omega : (n_species, n_pts)   [kg/m^3/s]
    """
    nsp = Y.shape[0]

    rho = np.empty(n_pts, dtype=float)
    # FIX BUG 3: inicializar D y omega ANTES del loop (antes causaban NameError)
    D = np.empty((nsp, n_pts), dtype=float)
    omega = np.empty((nsp, n_pts), dtype=float)

    for j in range(n_pts):
        rj, Dj, oj = backend.eval_node_species_only(T[j], Y[:, j])
        rho[j] = rj
        D[:, j] = Dj
        omega[:, j] = oj

    return rho, D, omega


def _eval_face_properties(T: np.ndarray, Y: np.ndarray, backend, n_pts: int):
    """
    Evalúa rho y D en todas las caras internas usando midpoint simple.

    Retorna
    -------
    rho_f : (n_faces,)
    D_f   : (n_species, n_faces)
    """
    nsp = Y.shape[0]
    n_faces = n_pts - 1

    rho_f = np.empty(n_faces, dtype=float)
    D_f = np.empty((nsp, n_faces), dtype=float)
    W_mix_face = np.empty(n_faces, dtype=float)

    for jf in range(n_faces):
        Tm = 0.5 * (T[jf] + T[jf + 1])
        Ym = 0.5 * (Y[:, jf] + Y[:, jf + 1])
        rj, Dj, _, W_mix_f = backend.eval_node_species_only(Tm, Ym, return_W_mix=True)
        rho_f[jf] = rj
        D_f[:, jf] = Dj
        W_mix_face[jf] = W_mix_f

    return rho_f, D_f, W_mix_face


# ---------------------------------------------------------------------------
#  Flujos difusivos corregidos
# ---------------------------------------------------------------------------
def _compute_all_fluxes(
    Y: np.ndarray,
    z: np.ndarray,
    rho_f: np.ndarray,
    D_f: np.ndarray,
    W: np.ndarray,
    W_mix_face: np.ndarray,
) -> np.ndarray:
    """
    Calcula los flujos difusivos corregidos en todas las caras usando gradientes molares.
    """
    dz = np.diff(z)  # (n_faces,)
    
    # Calcular W_mix en nodos
    W_mix_node = 1.0 / np.sum(Y / W[:, np.newaxis], axis=0) # (n_pts,)
    X = Y * W_mix_node[np.newaxis, :] / W[:, np.newaxis] # (n_species, n_pts)
    
    dXdz = np.diff(X, axis=1) / dz[np.newaxis, :]  # (n_species, n_faces)

    J_star = -rho_f[np.newaxis, :] * (W[:, np.newaxis] / W_mix_face[np.newaxis, :]) * D_f * dXdz
    sum_star = np.sum(J_star, axis=0, keepdims=True)

    flux = J_star - Y[:, :-1] * sum_star
    return flux


# ---------------------------------------------------------------------------
#  Derivada convectiva upwind para mdot escalar fijo
# ---------------------------------------------------------------------------
def _convective_term(Y_ind: np.ndarray, z: np.ndarray, mdot: float) -> np.ndarray:
    """
    Calcula mdot * dY/dz en los nodos interiores con upwind según el signo de mdot.

    Parámetros
    ----------
    Y_ind : (n_ind, n_pts)
    z     : (n_pts,)
    mdot  : escalar

    Retorna
    -------
    conv : (n_ind, n_pts-2)
    """
    if Y_ind.shape[1] <= 2:
        return np.empty((Y_ind.shape[0], 0), dtype=float)

    if mdot >= 0.0:
        dz_up = z[1:-1] - z[:-2]
        dYdz = (Y_ind[:, 1:-1] - Y_ind[:, :-2]) / dz_up[np.newaxis, :]
    else:
        dz_up = z[2:] - z[1:-1]
        dYdz = (Y_ind[:, 2:] - Y_ind[:, 1:-1]) / dz_up[np.newaxis, :]

    return mdot * dYdz


# ---------------------------------------------------------------------------
#  Bloques de frontera para species-only
# ---------------------------------------------------------------------------
def _left_species_bc(
    Y: np.ndarray,
    flux_0: np.ndarray,
    mdot: float,
    problem,
) -> np.ndarray:
    """
    BC izquierda para especies independientes (Cantera Inlet1D):

        -(J_k + mdot*Y_k(0)) + mdot*Y_k,in = 0
    """
    n_ind = problem.n_ind_species
    return -(flux_0[:n_ind] + mdot * Y[:n_ind, 0]) + mdot * problem.Y_in[:n_ind]


def _right_species_bc(
    Y: np.ndarray,
    flux_last: np.ndarray,
    mdot: float,
    problem,
) -> np.ndarray:
    """
    BC derecha: gradiente nulo (Cantera Outlet1D).
        Y_k(N-1) - Y_k(N-2) = 0
    """
    n_ind = problem.n_ind_species
    return Y[:n_ind, -1] - Y[:n_ind, -2]


# ---------------------------------------------------------------------------
#  Residual vectorizado completo
# ---------------------------------------------------------------------------
def _species_residual_vectorized(Y: np.ndarray, problem) -> np.ndarray:
    """
    Calcula el residual completo de especies con T y mdot congelados.

    Retorna
    -------
    Fm : (n_ind_species, n_points)
        Residual por especie independiente y por nodo.
    """
    n_ind = int(problem.n_ind_species)
    n_pts = int(problem.n_points)

    if n_pts < 2:
        raise ValueError("Se requieren al menos 2 puntos de malla.")

    z = np.asarray(problem.z, dtype=float)
    T = np.asarray(problem.T_fixed, dtype=float)
    mdot = float(problem.mdot_fixed)
    backend = problem.backend

    if backend is None:
        raise ValueError("problem.backend no está configurado.")
    if T.shape != (n_pts,):
        raise ValueError("problem.T_fixed debe tener shape (n_points,).")

    Fm = np.empty((n_ind, n_pts), dtype=float)

    # ------------------------------------------------------------
    # Propiedades nodales y en caras
    # ------------------------------------------------------------
    rho_node, D_node, omega_node = _eval_all_properties(T, Y, backend, n_pts)
    rho_face, D_face, W_mix_face = _eval_face_properties(T, Y, backend, n_pts)

    # Flujos difusivos corregidos en caras internas
    flux = _compute_all_fluxes(Y, z, rho_face, D_face, backend.W, W_mix_face)  # (n_species, n_faces)

    # ------------------------------------------------------------
    # Fronteras
    # ------------------------------------------------------------
    Fm[:, 0] = _left_species_bc(Y, flux[:, 0], mdot, problem)
    Fm[:, -1] = _right_species_bc(Y, flux[:, -1], mdot, problem)

    # ------------------------------------------------------------
    # Interior
    # ------------------------------------------------------------
    if n_pts > 2:
        dz_full = z[2:] - z[:-2]  # (n_interior,)

        conv = _convective_term(Y[:n_ind, :], z, mdot)  # (n_ind, n_interior)

        # Divergencia difusiva:
        # para nodo j=1..n-2 -> 2*(flux[j] - flux[j-1]) / (z[j+1]-z[j-1])
        divJ = 2.0 * (flux[:n_ind, 1:] - flux[:n_ind, :-1]) / dz_full[np.newaxis, :]

        source = omega_node[:n_ind, 1:-1]  # [kg/m^3/s]
        rho_int = rho_node[np.newaxis, 1:-1]

        Fm[:, 1:-1] = (conv + divJ - source) / rho_int

    return Fm


# ---------------------------------------------------------------------------
#  Interfaz pública
# ---------------------------------------------------------------------------
def residual_species_frozen(x: np.ndarray, problem) -> np.ndarray:
    """
    Residual estacionario F(x) para especies con T y mdot congelados.
    """
    problem.check_frozen_profiles()

    x = np.asarray(x, dtype=float)
    Y = unpack_species(x, problem.n_points, problem.n_species)

    if not np.all(np.isfinite(Y)):
        return np.full(x.size, 1.0e20)

    try:
        Fm = _species_residual_vectorized(Y, problem)
    except Exception:
        return np.full(x.size, 1.0e20)

    F = Fm.reshape(-1, order="C")
    if not np.all(np.isfinite(F)):
        return np.full(x.size, 1.0e20)
    return F


def residual_species_frozen_transient(
    x_new: np.ndarray,
    x_old: np.ndarray,
    problem,
    rdt: float,
) -> np.ndarray:
    """
    Residual transitorio backward-Euler:

        F_ts = F_ss(x_new) - rdt * (x_new - x_old)

    aplicado solo en nodos interiores, ya que las fronteras son algebraicas.
    """
    if not np.isfinite(rdt) or rdt <= 0.0:
        raise ValueError("rdt debe ser positivo y finito.")

    x_new = np.asarray(x_new, dtype=float)
    x_old = np.asarray(x_old, dtype=float)

    F_ss = residual_species_frozen(x_new, problem)
    if not np.all(np.isfinite(F_ss)):
        return np.full(x_new.size, 1.0e20)

    n_ind = int(problem.n_ind_species)
    n_pts = int(problem.n_points)

    Fm = F_ss.reshape((n_ind, n_pts), order="C").copy()
    Xn = x_new.reshape((n_ind, n_pts), order="C")
    Xo = x_old.reshape((n_ind, n_pts), order="C")

    if n_pts > 2:
        Fm[:, 1:-1] -= rdt * (Xn[:, 1:-1] - Xo[:, 1:-1])

    F = Fm.reshape(-1, order="C")
    if not np.all(np.isfinite(F)):
        return np.full(x_new.size, 1.0e20)
    return F
