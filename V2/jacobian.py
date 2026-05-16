"""
jacobian.py - Sparse Jacobian builders and transient diagonal update.

Default assembly follows the Cantera OneDim::evalJacobian pattern:
- perturb one state variable at a time
- evaluate residual only on local rows (j-1, j, j+1)
- fill a sparse Jacobian column from local finite differences
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from state import build_transient_mask


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
    - write local rows into Jacobian column
    """
    from residual import residual_local_rows, build_local_jacobian_cache

    x = np.asarray(x, dtype=float)
    n_pts = int(problem.n_points)
    n_sp = int(problem.n_species)
    nv = 2 + n_sp
    n_total = n_pts * nv

    f0 = fun(x, problem)

    rel_perturb = float(getattr(problem, "jacobian_rel_perturb", eps))
    abs_perturb = float(getattr(problem, "jacobian_abs_perturb", 1e-10))
    threshold = float(getattr(problem, "jacobian_threshold", 0.0))

    rows_list: list[int] = []
    cols_list: list[int] = []
    vals_list: list[float] = []

    xp = x.copy()
    local_cache = build_local_jacobian_cache(x, problem)

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

            rows, f_local = residual_local_rows(xp, problem, j, cache=local_cache)
            delta = f_local - f0[rows]

            if threshold > 0.0:
                keep = np.abs(delta) > threshold
            else:
                keep = np.ones(delta.size, dtype=bool)

            # Keep diagonal entry even if tiny (as in Cantera condition).
            kdiag = np.where(rows == col)[0]
            if kdiag.size > 0:
                keep[int(kdiag[0])] = True

            if np.any(keep):
                rows_nz = rows[keep]
                vals_nz = delta[keep] * rdx
                nnz = int(rows_nz.size)
                rows_list.extend(rows_nz.tolist())
                cols_list.extend([col] * nnz)
                vals_list.extend(vals_nz.tolist())

            xp[col] = xsave

    jmat = sparse.coo_matrix((vals_list, (rows_list, cols_list)), shape=(n_total, n_total)).tocsr()
    return jmat


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
                dx = eps * max(abs(xval), 1.0)
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
    - "coloring"
    """
    mode = str(getattr(problem, "jacobian_mode", "cantera_local")).strip().lower()
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
    j_csc = jmat.tocsc()
    return {
        "method": "direct",
        "solver": splu(j_csc, permc_spec="COLAMD"),
    }


def solve_linear(linear_state, rhs: np.ndarray) -> np.ndarray:
    if hasattr(linear_state, "solve") and not isinstance(linear_state, dict):
        return linear_state.solve(rhs)
    return linear_state["solver"].solve(rhs)
