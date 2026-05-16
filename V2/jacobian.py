"""
jacobian.py – Jacobiano numérico sparse con actualización transitoria diagonal.

Con el layout por-punto entrelazado (state.py), la actualización transitoria
reproduce exactamente MultiJac::updateTransient de Cantera:

  J_transient[n, n] = J_steady[n, n] - mask[n] * rdt

donde mask[n] = 1 si la variable n es diferencial (T interior, Y interior),
y 0 en caso contrario (U, fronteras, punto de anclaje).

El ancho de banda del sistema con n_sp especies y n_pts puntos es:
  bw = 2 * n_vars   (n_vars = 2 + n_sp)
igual que OneDim::m_bw en Cantera.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from state import build_transient_mask


# ---------------------------------------------------------------------------
#  Jacobiano numérico sparse (3-coloring por puntos)
# ---------------------------------------------------------------------------
def banded_jacobian(fun, x: np.ndarray, problem, eps: float = 1e-8) -> sparse.csr_matrix:
    """
    Jacobiano numérico banded usando 3-coloring por puntos.

    Dado que cada ecuación en el punto j sólo depende del estado en
    los puntos j-1, j, j+1, se pueden perturbar todos los puntos del
    mismo "color" (j % 3 == c) simultáneamente.

    Coste: 3 evaluaciones del residual para construir el Jacobiano completo
    (en lugar de n_vars * n_pts evaluaciones del método denso).
    """
    x = np.asarray(x, dtype=float)
    n_pts = int(problem.n_points)
    n_sp = int(problem.n_species)
    nv = 2 + n_sp
    n_total = n_pts * nv

    F0 = fun(x, problem)
    point_rows = [
        np.arange(max(0, j - 1) * nv, (min(n_pts - 1, j + 1) + 1) * nv, dtype=np.int32)
        for j in range(n_pts)
    ]

    rows_list: list[int] = []
    cols_list: list[int] = []
    vals_list: list[float] = []
    xp = x.copy()

    # Loop: variable first, then 3-color over points.
    # Within a point, all variables affect the same equations, so they CANNOT
    # be perturbed simultaneously. Instead we fix the variable index and
    # exploit that non-adjacent points don't interact (stencil j±1).
    for v_idx in range(nv):
        for color in range(3):
            pts = list(range(color, n_pts, 3))
            xp[:] = x
            dx_vals: list[float] = []
            cols: list[int] = []

            for j in pts:
                col_idx = j * nv + v_idx
                xval = x[col_idx]
                dx = eps * max(abs(xval), 1.0)
                if xval < 0:
                    dx = -dx
                xp[col_idx] += dx
                dx_vals.append(dx)
                cols.append(col_idx)

            Fp = fun(xp, problem)
            dF = Fp - F0

            for idx, j in enumerate(pts):
                rows = point_rows[j]
                vals = dF[rows] / dx_vals[idx]
                nz = np.abs(vals) > 1e-30
                if not np.any(nz):
                    continue
                rows_nz = rows[nz]
                vals_nz = vals[nz]
                n_nz = int(rows_nz.size)
                rows_list.extend(rows_nz.tolist())
                cols_list.extend([cols[idx]] * n_nz)
                vals_list.extend(vals_nz.tolist())

    J = sparse.coo_matrix(
        (vals_list, (rows_list, cols_list)),
        shape=(n_total, n_total),
    ).tocsr()
    return J


# ---------------------------------------------------------------------------
#  Actualización transitoria (MultiJac::updateTransient)
# ---------------------------------------------------------------------------
def update_transient(J_ss: sparse.csr_matrix, mask: np.ndarray,
                     rdt: float, inplace: bool = False) -> sparse.csr_matrix:
    """
    Aplica el término transitorio al Jacobiano estacionario.

    Equivale a MultiJac::updateTransient(rdt, mask):
      J_transient[n, n] = J_ss[n, n] - mask[n] * rdt

    La operación es puramente diagonal porque estado y residual comparten
    el mismo layout por-punto (estado.py).

    Parámetros
    ----------
    J_ss : Jacobiano estacionario (sparse CSR)
    mask : máscara transitoria (output de build_transient_mask)
    rdt  : 1/dt
    """
    if rdt <= 0.0:
        return J_ss

    # Actualizacion diagonal vectorizada (evita bucle Python por indice).
    J_t = J_ss if inplace else J_ss.copy()
    d = J_t.diagonal()
    d -= np.asarray(mask, dtype=float) * float(rdt)
    J_t.setdiag(d)
    return J_t


def build_jacobian_steady(fun, x: np.ndarray, problem,
                          eps: float = 1e-8) -> tuple[sparse.csr_matrix, np.ndarray]:
    """
    Construye el Jacobiano estacionario y devuelve también su diagonal
    (equivalente a m_ssdiag en MultiJac).

    Retorna
    -------
    J  : Jacobiano estacionario (CSR)
    ss_diag : diagonal de J (para updateTransient posterior)
    """
    J = banded_jacobian(fun, x, problem, eps=eps)
    ss_diag = J.diagonal().copy()
    return J, ss_diag


def build_jacobian_transient(fun, x: np.ndarray, problem, rdt: float,
                              eps: float = 1e-8) -> sparse.csr_matrix:
    """
    Construye el Jacobiano transitorio completo en un paso.

    Equivale a la secuencia Cantera:
      1. evalJacobian(x)           → Jacobiano estacionario
      2. jac->updateTransient(rdt, transientMask())
    """
    J_ss, _ = build_jacobian_steady(fun, x, problem, eps=eps)
    mask = build_transient_mask(problem.n_points, problem.n_species,
                                solve_energy=bool(problem.solve_energy))
    return update_transient(J_ss, mask, rdt)


# ---------------------------------------------------------------------------
#  Solver sparse (wrapper conveniente)
# ---------------------------------------------------------------------------
def factorize(J: sparse.csr_matrix) -> dict:
    """
    Construye el resolvedor LU directo para el Jacobiano.
    """
    Jcsc = J.tocsc()
    return {
        "method": "direct",
        "solver": splu(Jcsc, permc_spec="COLAMD"),
    }


def solve_linear(linear_state, rhs: np.ndarray) -> np.ndarray:
    """Resuelve sistema lineal usando LU directo."""
    # Compatibilidad hacia atrás (si llega un objeto LU directo).
    if hasattr(linear_state, "solve") and not isinstance(linear_state, dict):
        return linear_state.solve(rhs)
    return linear_state["solver"].solve(rhs)


