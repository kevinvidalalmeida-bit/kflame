from __future__ import annotations

from equations import (
    residual,
    build_jacobian_steady,
    update_transient,
    factorize,
    solve_linear
)
from state import build_transient_mask
from dataclasses import dataclass, field
from species_backend import SpeciesBackend
from state import pack_state, unpack_state, interpolate_state
from typing import Any
import numpy as np
import time
"""
solver.py - Consolida la lógica algorítmica de resolución (Newton, Mesh, Solver).
"""
"""
mesh.py – Generación de malla y refinamiento adaptativo tipo Cantera.
"""


# ---------------------------------------------------------------------------
#  Malla inicial (clustering gaussiano)
# ---------------------------------------------------------------------------
def _adaptive_xi(
    n_points: int,
    locs=(0.0, 0.3, 0.5, 1.0),
    cluster_strength: float = 8.0,
    cluster_sigma: float | None = None,
    n_dense: int = 4001,
) -> np.ndarray:
    if n_points < 2:
        raise ValueError("n_points debe ser >= 2")
    x1 = float(locs[1])
    x2 = float(locs[2])
    if not (0.0 <= x1 < x2 <= 1.0):
        return np.linspace(0.0, 1.0, n_points, dtype=float)

    center = 0.5 * (x1 + x2)
    sigma = float(cluster_sigma) if cluster_sigma else 0.5 * (x2 - x1)
    sigma = max(sigma, 1.0e-3)
    strength = max(0.0, float(cluster_strength))

    xi_dense = np.linspace(0.0, 1.0, n_dense, dtype=float)
    monitor = 1.0 + strength * np.exp(-((xi_dense - center) / sigma) ** 2)
    cdf = np.empty_like(xi_dense)
    cdf[0] = 0.0
    dxi = np.diff(xi_dense)
    cdf[1:] = np.cumsum(0.5 * (monitor[1:] + monitor[:-1]) * dxi)
    total = cdf[-1]
    if total <= 0.0 or not np.isfinite(total):
        return np.linspace(0.0, 1.0, n_points, dtype=float)
    cdf /= total
    xi_target = np.linspace(0.0, 1.0, n_points, dtype=float)
    xi = np.interp(xi_target, cdf, xi_dense)
    xi[0] = 0.0
    for j in range(1, xi.size):
        xi[j] = max(xi[j], xi[j - 1] + 1.0e-14)
    xi /= xi[-1]
    return xi


def initial_grid(
    width: float,
    n_points: int = 8,
    locs=(0.0, 0.3, 0.5, 1.0),
    cantera_seed_grid: bool = False,
    adaptive: bool = True,
    cluster_strength: float = 8.0,
    cluster_sigma: float | None = None,
) -> np.ndarray:
    if n_points < 2:
        raise ValueError("n_points debe ser >= 2")
    if width <= 0.0:
        raise ValueError("width debe ser > 0")
    if cantera_seed_grid and n_points == 8:
        # Match FreeFlame(width=...) default seed used by Cantera.
        return width * np.array([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0], dtype=float)
    if not adaptive:
        return np.linspace(0.0, width, n_points, dtype=float)
    xi = _adaptive_xi(
        n_points=n_points,
        locs=locs,
        cluster_strength=cluster_strength,
        cluster_sigma=cluster_sigma,
    )
    return width * xi


def grid_from_reference(npz_path: str, subsample: int | None = None) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    z_ref = np.asarray(data["z"], dtype=float)

    if subsample is not None and subsample > 1:
        idx = np.arange(0, len(z_ref), subsample)
        if idx[-1] != len(z_ref) - 1:
            idx = np.append(idx, len(z_ref) - 1)
        z_ref = z_ref[idx]

    return z_ref


def build_freeflame_refiner_profiles(problem, u: np.ndarray, T: np.ndarray, Y: np.ndarray) -> dict:
    """
    Selección de perfiles para refinamiento en free premixed flame.

    Flow1D activa U, V y T cuando la energía está activa; en este solver
    reducido (sin V) se toma:
      * u si refine_with_u
      * T si solve_energy y refine_with_T
      * Y_k si refine_with_species
    """
    profiles: dict[str, np.ndarray] = {}

    if bool(getattr(problem, "refine_with_u", True)):
        profiles["u"] = np.asarray(u, dtype=float)

    if bool(getattr(problem, "solve_energy", True)) and bool(
        getattr(problem, "refine_with_T", True)
    ):
        profiles["T"] = np.asarray(T, dtype=float)

    if bool(getattr(problem, "refine_with_species", True)):
        names = getattr(problem, "species_names", None)
        for k in range(Y.shape[0]):
            if names is not None and k < len(names):
                key = f"Y_{names[k]}"
            else:
                key = f"Y_{k}"
            profiles[key] = np.asarray(Y[k, :], dtype=float)

    return profiles


class AdaptiveRefiner:
    """
    Refinador de malla adaptativo basado en Cantera refine.cpp.

    Defaults alineados con Refiner::setCriteria:
      ratio=10.0, slope=0.8, curve=0.8, prune=-0.1
    """

    def __init__(self, ratio=10.0, slope=0.8, curve=0.8, prune=-0.1,
                 grid_min=1e-10, max_points=1000):
        if ratio < 2.0:
            raise ValueError("ratio debe ser >= 2.0")
        if not (0.0 <= slope <= 1.0):
            raise ValueError("slope debe estar entre 0 y 1")
        if not (0.0 <= curve <= 1.0):
            raise ValueError("curve debe estar entre 0 y 1")
        if prune > curve or prune > slope:
            raise ValueError("prune debe ser menor que curve y slope")

        self.ratio = float(ratio)
        self.slope = float(slope)
        self.curve = float(curve)
        self.prune = float(prune)
        self.grid_min = float(grid_min)
        self.max_points = int(max_points)
        self._thresh = 1e-14
        self._min_range = 0.01
        self._UNSET = 0
        self._KEEP = 1
        self._REMOVE = -1

    def analyze(self, z: np.ndarray, profiles: dict,
                j_fixed: int | None = None) -> tuple:
        z = np.asarray(z, dtype=float)
        n = int(z.size)
        if n < 2 or n >= self.max_points:
            return set(), set()

        dz = np.diff(z)
        insert_after: set[int] = set()
        keep = np.full(n, self._UNSET, dtype=int)
        keep[0] = self._KEEP
        keep[n - 1] = self._KEEP

        if j_fixed is not None and 0 <= j_fixed < n:
            keep[j_fixed] = self._KEEP

        pruning_enabled = self.prune > 0.0

        for _name, vals in profiles.items():
            vals = np.asarray(vals, dtype=float)
            if vals.shape != (n,):
                continue

            slope_arr = np.diff(vals) / dz

            val_min = float(np.min(vals))
            val_max = float(np.max(vals))
            slp_min = float(np.min(slope_arr))
            slp_max = float(np.max(slope_arr))

            val_mag = max(abs(val_max), abs(val_min))
            slp_mag = max(abs(slp_max), abs(slp_min))

            if (val_max - val_min) > self._min_range * max(val_mag, self._thresh):
                max_change = self.slope * (val_max - val_min)
                for j in range(n - 1):
                    ratio = abs(vals[j + 1] - vals[j]) / (max_change + self._thresh)
                    if ratio > 1.0 and dz[j] >= 2.0 * self.grid_min:
                        insert_after.add(j)
                    if pruning_enabled:
                        if ratio >= self.prune:
                            keep[j] = self._KEEP
                            keep[j + 1] = self._KEEP
                        elif keep[j] == self._UNSET:
                            keep[j] = self._REMOVE

            if (slp_max - slp_min) > self._min_range * max(slp_mag, self._thresh):
                max_change = self.curve * (slp_max - slp_min)
                for j in range(n - 2):
                    ratio = abs(slope_arr[j + 1] - slope_arr[j]) / (
                        max_change + self._thresh / dz[j]
                    )
                    if (
                        ratio > 1.0
                        and dz[j] >= 2.0 * self.grid_min
                        and dz[j + 1] >= 2.0 * self.grid_min
                    ):
                        insert_after.add(j)
                        insert_after.add(j + 1)
                    if pruning_enabled:
                        if ratio >= self.prune:
                            keep[j + 1] = self._KEEP
                        elif keep[j + 1] == self._UNSET:
                            keep[j + 1] = self._REMOVE

        for j in range(1, n - 1):
            if dz[j] > self.ratio * dz[j - 1]:
                insert_after.add(j)
                for jj in (j - 1, j, j + 1, j + 2):
                    if 0 <= jj < n:
                        keep[jj] = self._KEEP

            if dz[j - 1] > self.ratio * dz[j]:
                insert_after.add(j - 1)
                for jj in (j - 2, j - 1, j, j + 1):
                    if 0 <= jj < n:
                        keep[jj] = self._KEEP

            if j > 1 and (z[j + 1] - z[j - 1]) > self.ratio * dz[j - 2]:
                keep[j] = self._KEEP

            if j < n - 2 and (z[j + 1] - z[j - 1]) > self.ratio * dz[j + 1]:
                keep[j] = self._KEEP

        if pruning_enabled:
            for j in range(2, n - 1):
                if keep[j] == self._REMOVE and keep[j - 1] == self._REMOVE:
                    keep[j] = self._KEEP

            remove = {
                int(j) for j in range(1, n - 1)
                if keep[j] == self._REMOVE
            }
        else:
            remove = set()

        return insert_after, remove

    def refine(self, z: np.ndarray, profiles: dict,
               all_Y: np.ndarray | None = None,
               j_fixed: int | None = None) -> tuple:
        z = np.asarray(z, dtype=float)
        insert_after, remove = self.analyze(z, profiles, j_fixed)

        if not insert_after and not remove:
            return z, False, 0, 0

        keep_mask = np.ones(z.size, dtype=bool)
        for j in remove:
            if 0 < j < z.size - 1:
                keep_mask[j] = False

        z_new = []
        n_inserted = 0
        for j in range(z.size - 1):
            if keep_mask[j]:
                z_new.append(float(z[j]))
            if j in insert_after and (len(z_new) + 1) < self.max_points:
                z_new.append(0.5 * (float(z[j]) + float(z[j + 1])))
                n_inserted += 1

        if keep_mask[-1]:
            z_new.append(float(z[-1]))

        z_new = np.asarray(z_new, dtype=float)
        z_new = np.unique(z_new)

        n_removed = int(np.sum(~keep_mask[1:-1]))
        changed = bool(z_new.size != z.size)
        return z_new, changed, n_inserted, n_removed

    def interpolate_solution(self, z_old: np.ndarray, z_new: np.ndarray,
                             *arrays: np.ndarray) -> list:
        result = []
        for arr in arrays:
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 1:
                result.append(np.interp(z_new, z_old, arr))
            elif arr.ndim == 2:
                new_arr = np.empty((arr.shape[0], len(z_new)), dtype=float)
                for k in range(arr.shape[0]):
                    new_arr[k, :] = np.interp(z_new, z_old, arr[k, :])
                result.append(new_arr)
            else:
                raise ValueError(f"Array con ndim={arr.ndim} no soportado")
        return result


# --- FIN DE MESH.PY, INICIO DE NEWTON.PY ---

"""
newton.py - Damped Newton solver aligned with Cantera MultiNewton.

Key behaviors mirrored from Cantera:
- Reuse Jacobian for limited age and rebuild when stale.
- Compute undamped Newton step with fixed Jacobian.
- Damped step acceptance based on weighted step norm:
    accept if s1 < 1.0 or s1 < s0
- If damping fails with old Jacobian, force Jacobian rebuild (up to 3 retries).
- On failure, return the original input state unchanged.
"""




# ---------------------------------------------------------------------------
#  Weighted norm (OneDim::weightedNorm analogue)
# ---------------------------------------------------------------------------
def weighted_norm(step: np.ndarray, x: np.ndarray, problem, rdt: float = 0.0) -> float:
    """
    Weighted norm used by Cantera's Newton criterion.

    w = rtol * mean(abs(x_component)) + atol
    norm = sqrt(sum((step / w)^2) / N)

    Uses steady tolerances when rdt == 0 and transient tolerances when rdt > 0.
    """
    n_pts = int(problem.n_points)
    n_sp = int(problem.n_species)
    nv = 2 + n_sp
    n_total = n_pts * nv

    if step.size != n_total or x.size != n_total:
        return float(np.linalg.norm(step) / max(np.linalg.norm(x), 1e-300))

    if rdt > 0.0:
        rtol = float(getattr(problem, "transient_rtol", 1e-4))
        atol = float(getattr(problem, "transient_atol", 1e-11))
    else:
        rtol = float(getattr(problem, "steady_rtol", 1e-4))
        atol = float(getattr(problem, "steady_atol", 1e-9))

    x_r = x.reshape(n_pts, nv)
    step_r = step.reshape(n_pts, nv)
    sumsq = 0.0
    for v in range(nv):
        esum = float(np.sum(np.abs(x_r[:, v])))
        ewt = max(rtol * esum / n_pts + atol, 1e-300)
        fs = step_r[:, v] / ewt
        sumsq += float(np.dot(fs, fs))

    return float(np.sqrt(sumsq / n_total))


# ---------------------------------------------------------------------------
#  Bound step factor (MultiNewton::boundStep analogue)
# ---------------------------------------------------------------------------

def bound_step_debug(x0: np.ndarray, step0: np.ndarray, problem) -> tuple[float, str]:
    n_pts = int(problem.n_points)
    n_sp = int(problem.n_species)
    nv = 2 + n_sp

    t_lo = float(getattr(problem, "T_lower_bound", 200.0))
    t_hi = float(getattr(problem, "T_upper_bound", 2.0 * 3000.0))
    y_lo = float(getattr(problem, "Y_lower_bound", -1e-5))
    y_hi = 1.0e5

    lower = np.full((n_pts, nv), -1e20)
    upper = np.full((n_pts, nv), 1e20)
    lower[:, 1] = t_lo
    upper[:, 1] = t_hi
    lower[:, 2:] = y_lo
    upper[:, 2:] = y_hi

    fbound = 1.0
    x_r = x0.reshape(n_pts, nv)
    s_r = step0.reshape(n_pts, nv)
    reason = ""

    for v in range(nv):
        xv = x_r[:, v]
        sv = s_r[:, v]
        xnv = xv + sv

        l_v = lower[:, v]
        u_v = upper[:, v]

        above = (sv > 0.0) & (xnv > u_v)
        if np.any(above):
            cand = (u_v[above] - xv[above]) / sv[above]
            min_cand = float(np.min(cand))
            if min_cand < fbound:
                fbound = min_cand
                reason = f"var={v} above upper xv={np.min(xv[above])} sv={np.max(sv[above])}"

        below = (sv < 0.0) & (xnv < l_v)
        if np.any(below):
            cand = (xv[below] - l_v[below]) / (-sv[below])
            min_cand = float(np.min(cand))
            if min_cand < fbound:
                fbound = min_cand
                reason = f"var={v} below lower xv={np.min(xv[below])} sv={np.min(sv[below])}"

    return max(0.0, fbound), reason


# ---------------------------------------------------------------------------
#  Reusable Jacobian state
# ---------------------------------------------------------------------------
@dataclass
class JacobianState:
    J: object = None
    lu: object = None
    ss_diag: np.ndarray = field(default_factory=lambda: np.empty(0))
    age: int = 10000
    n_evals: int = 0
    last_rdt: float | None = None

    def is_stale(self, max_age: int) -> bool:
        # Cantera refreshes when age() > maxAge
        return self.J is None or self.lu is None or self.age > max_age


def _transient_alpha(transient_order: int, x_older: np.ndarray | None) -> float:
    use_bdf2 = int(transient_order) >= 2 and x_older is not None
    return 1.5 if use_bdf2 else 1.0


def _build_linear_model(steady_fun, x: np.ndarray, problem,
                        jac_eps: float, mask: np.ndarray,
                        rdt_curr: float) -> tuple[object, object, np.ndarray]:
    problem._current_rdt = rdt_curr
    try:
        j_ss, ss_diag = build_jacobian_steady(steady_fun, x, problem, eps=jac_eps)
    finally:
        if hasattr(problem, '_current_rdt'):
            delattr(problem, '_current_rdt')

    if rdt_curr > 0.0:
        j_t = update_transient(j_ss, mask, rdt_curr, inplace=True)
    else:
        j_t = j_ss
    lu = factorize(j_t)
    return j_t, lu, ss_diag


# ---------------------------------------------------------------------------
#  Main Newton solver
# ---------------------------------------------------------------------------
def newton_solve(
    steady_fun,
    x0: np.ndarray,
    problem,
    rdt: float = 0.0,
    x_old: np.ndarray | None = None,
    x_older: np.ndarray | None = None,
    transient_order: int = 1,
    max_iter: int = 20,
    max_jac_age: int = 5,
    max_damp_iter: int = 7,
    damp_factor: float = float(np.sqrt(2.0)),
    tol: float = 1.0,
    jac_eps: float = 1e-5,
    alpha_min: float = 1e-10,
    verbose: bool = False,
    jac_state: JacobianState | None = None,
) -> tuple[np.ndarray, bool, list[dict], JacobianState]:
    """
    Damped Newton aligned with Cantera MultiNewton.

    Returns
    -------
    x_out, converged, history, jac_state
    """
    x = np.asarray(x0, dtype=float)
    history: list[dict] = []

    if jac_state is None:
        jac_state = JacobianState()

    def full_fun(xv: np.ndarray) -> np.ndarray:
        from equations import residual as _res
        return _res(
            xv,
            problem,
            rdt=rdt,
            x_old=x_old,
            x_older=x_older,
            transient_order=transient_order,
        )

    mask = build_transient_mask(problem.n_points, problem.n_species,
                                solve_energy=bool(problem.solve_energy))

    rdt_curr = float(rdt) * _transient_alpha(transient_order, x_older)
    rdt_changed = (
        jac_state.last_rdt is None
        or not np.isclose(jac_state.last_rdt, rdt_curr, rtol=1e-12, atol=0.0)
    )

    force_new_jac = bool(rdt_changed)
    n_jac_reeval = 0
    status = -1

    for it in range(max_iter):
        # Jacobian refresh logic
        if force_new_jac or jac_state.is_stale(max_jac_age):
            try:
                j_t, lu, ss_diag = _build_linear_model(
                    steady_fun, x, problem, jac_eps, mask, rdt_curr
                )
                jac_state.J = j_t
                jac_state.lu = lu
                jac_state.ss_diag = ss_diag
                jac_state.age = 0
                jac_state.n_evals += 1
                jac_state.last_rdt = rdt_curr
                force_new_jac = False
            except Exception as exc:
                history.append({"iter": it, "status": "jac_fail", "error": str(exc)})
                status = -4
                break

        # MultiNewton::step equivalent
        f = full_fun(x)
        if not np.all(np.isfinite(f)):
            history.append({"iter": it, "status": "nonfinite_F"})
            status = -5
            break

        try:
            step0 = solve_linear(jac_state.lu, -f)
        except Exception as exc:
            history.append({"iter": it, "status": "linear_solve_fail", "error": str(exc)})
            force_new_jac = True
            continue

        if not np.all(np.isfinite(step0)):
            history.append({"iter": it, "status": "nonfinite_step"})
            status = -5
            break

        jac_state.age += 1

        s0 = weighted_norm(step0, x, problem, rdt=rdt)
        normf = float(np.linalg.norm(f, ord=np.inf))

        # Cantera's MultiNewton::boundStep: compute a scalar factor to keep
        # x + alpha*step inside bounds.
        fbound, _fbound_reason = bound_step_debug(x, step0, problem)
        if fbound < alpha_min:
            history.append({
                "iter": it,
                "status": "bound_fail",
                "normF": normf,
                "s0": s0,
                "fbound": fbound,
                "reason": _fbound_reason,
                "jac_age": jac_state.age,
            })

            # MultiNewton: try fresh Jacobian if previous one was aged (>1)
            if jac_state.age > 1:
                force_new_jac = True
                if verbose:
                    print(f"  Newton it={it:3d} bound failure -> force new Jacobian")
                if n_jac_reeval > 3:
                    status = -3
                    break
                n_jac_reeval += 1
                continue

            status = -3
            break

        alpha = min(fbound, 1.0)

        # MultiNewton::dampStep equivalent
        damp_ok = False
        x1 = x.copy()
        s1 = float("inf")

        for _ in range(max_damp_iter):
            if alpha < alpha_min:
                break

            x_try = x + alpha * step0
            f_try = full_fun(x_try)
            if not np.all(np.isfinite(f_try)):
                alpha /= damp_factor
                continue

            try:
                step1 = solve_linear(jac_state.lu, -f_try)
            except Exception:
                alpha /= damp_factor
                continue

            if not np.all(np.isfinite(step1)):
                alpha /= damp_factor
                continue

            s1_try = weighted_norm(step1, x_try, problem, rdt=rdt)
            if s1_try < 1.0 or s1_try < s0:
                damp_ok = True
                x1 = x_try
                s1 = s1_try
                break

            alpha /= damp_factor

        if damp_ok:
            x = x1
            converged = bool(s1 < tol)
            history.append({
                "iter": it,
                "status": "ok" if converged else "step",
                "normF": normf,
                "s0": s0,
                "s1": s1,
                "alpha": alpha,
                "jac_age": jac_state.age,
            })

            if verbose:
                print(
                    f"  Newton it={it:3d} ||F||inf={normf:.4e} "
                    f"s0={s0:.3e} s1={s1:.3e} a={alpha:.3e} age={jac_state.age}"
                )
            if converged:
                # Cantera resets Jacobian age after steady convergence
                # (keeps it fresh for follow-up operations).
                if rdt == 0.0:
                    jac_state.age = 0
                status = 1
                break
            status = 0

        else:
            history.append({
                "iter": it,
                "status": "no_damp",
                "normF": normf,
                "s0": s0,
                "jac_age": jac_state.age,
            })

            # MultiNewton: try fresh Jacobian if previous one was aged (>1)
            if jac_state.age > 1:
                force_new_jac = True
                if verbose:
                    print(f"  Newton it={it:3d} no damping -> force new Jacobian")
                if n_jac_reeval > 3:
                    status = -2
                    break
                n_jac_reeval += 1
                continue

            status = -2
            break

    if status != 1:
        # Match MultiNewton.cpp: on failure, return the last accepted iterate
        # (which is x0 only if no successful damped step was taken).
        x = np.asarray(x, dtype=float).copy()

    return x, bool(status == 1), history, jac_state


# --- FIN DE NEWTON.PY, INICIO DE SOLVER_CANTERA_AUTO.PY ---

"""
solver_cantera_auto.py – Solver híbrido Newton + time-stepping.

Replica la estructura de Sim1D::solve() de Cantera:

  while new_points > 0:
      SteadyStateSystem::solve()   ← hybrid_newton_solve()
      new_points = refine()        ← AdaptiveRefiner

SteadyStateSystem::solve() (en numerics/SteadyStateSystem.cpp) implementa:
  for each cycle:
      intento Newton estacionario   -> newton_solve(rdt=0)
      si falla -> time steps: BE de arranque y luego BDF2
      si falla -> reduce dt y reintenta
  hasta converger o agotar max_steps

La novedad respecto al código anterior:
  * No hay `transient_jacobian_pairs`; el update diagonal funciona
    porque estado y residual tienen el mismo layout por-punto.
  * Refine usa la solución convergida (igual que Sim1D::refine), no la
    solución del intento en curso.
  * Se guarda m_xlast_ss (último steady) para restaurar si refine falla.
"""





# ---------------------------------------------------------------------------
#  Opciones
# ---------------------------------------------------------------------------
@dataclass
class SolveOptions:
    # Conjetura inicial
    u_left_guess: float = 1.00

    # Newton estacionario
    steady_max_iter: int = 50
    max_jac_age: int = 5
    steady_max_jac_age: int | None = None
    transient_max_jac_age: int | None = None
    max_damp_iter: int = 7
    tol: float = 1.0          # norma ponderada del paso (igual que Cantera)
    jac_eps: float = 1e-5
    jac_abs_perturb: float = 1e-10
    jac_threshold: float = 0.0
    jacobian_mode: str = "cantera_local"
    alpha_min: float = 1e-10

    # Time-stepping (híbrido)
    time_step: float = 1e-5
    time_step_grow: float = 1.5
    time_step_shrink: float = 0.5
    time_step_factor: float | None = None
    min_time_step: float = 1e-16
    max_time_step: float = 1e8
    transient_steps_per_cycle: int = 10 # compatibilidad
    time_step_sequence: tuple[int, ...] = (10,)
    transient_max_iter: int = 50
    max_time_step_count: int = 500
    transient_scheme: str = "be"
    reset_bad_after_failures: int = 3

    # Refinamiento (defaults de Refiner::setCriteria)
    refine_grid: bool = True
    max_refine_passes: int = 6
    refine_ratio: float = 10.0
    refine_slope: float = 0.8
    refine_curve: float = 0.8
    refine_prune: float = -0.1
    refine_grid_min: float = 1e-10
    refine_max_points: int = 500

    # Auto-expansion of domain (mimics FreeFlame.solve(auto=True))
    domain_auto_expand: bool = True
    max_domain_expansions: int = 12
    domain_expand_factor: float = 2.0
    domain_edge_slope_tol: float = 0.02

    # Auto bootstrap on fixed grids (mirrors Cantera _onedim auto path)
    auto_bootstrap_grids: bool = True
    bootstrap_grid_points: tuple[int, ...] = (12, 24, 48)
    bootstrap_max_grid_points: int = 1000

    max_total_time_s: float = 300.0
    verbose: bool = True


class DomainTooNarrowError(RuntimeError):
    """Raised to mirror Cantera FreeFlame auto-width callback behavior."""

    def __init__(self, x: np.ndarray, metrics: dict[str, float]):
        super().__init__("Domain too narrow for flame thickness.")
        self.x = np.asarray(x, dtype=float).copy()
        self.metrics = dict(metrics)


# ---------------------------------------------------------------------------
#  Función residual para el solver Newton
# ---------------------------------------------------------------------------
def _make_steady_fun(problem):
    """Cierre que llama a residual(x, problem, rdt=0)."""
    def fun(x, prob):
        return residual(x, prob, rdt=0.0, x_old=None)
    return fun


# ---------------------------------------------------------------------------
#  Hybrid Newton (SteadyStateSystem::solve)
# ---------------------------------------------------------------------------
def _hybrid_newton(problem, x0: np.ndarray, opts: SolveOptions,
                   label: str = "", steady_callback=None) -> tuple[np.ndarray, bool, list[dict]]:
    """
    Ciclos steady-Newton / time-step hasta convergencia.
    Si `steady_callback` existe, se ejecuta inmediatamente despues de cada
    Newton estacionario convergido (como el callback steady en Cantera).
    Devuelve (x, converged, history).
    """
    x = np.asarray(x0, dtype=float).copy()
    history: list[dict] = []

    # Jacobian finite-difference settings (Cantera-like).
    problem.jacobian_rel_perturb = float(getattr(opts, "jac_eps", 1e-5))
    problem.jacobian_abs_perturb = float(getattr(opts, "jac_abs_perturb", 1e-10))
    problem.jacobian_threshold = float(getattr(opts, "jac_threshold", 0.0))
    problem.jacobian_mode = str(getattr(opts, "jacobian_mode", "coloring"))

    steady_fun = _make_steady_fun(problem)
    jac: JacobianState | None = None
    dt = float(opts.time_step)

    # Cantera-equivalent Jacobian age controls (steady/transient)
    ss_jac_age = (
        int(opts.steady_max_jac_age)
        if opts.steady_max_jac_age is not None
        else int(opts.max_jac_age)
    )
    ts_jac_age = (
        int(opts.transient_max_jac_age)
        if opts.transient_max_jac_age is not None
        else int(opts.max_jac_age)
    )

    # Time-step schedule m_steps; if exhausted, repeat last entry.
    raw_steps = tuple(int(v) for v in getattr(opts, "time_step_sequence", (10,)))
    if not raw_steps:
        raw_steps = (int(getattr(opts, "transient_steps_per_cycle", 10)),)
    step_sequence = tuple(max(1, int(v)) for v in raw_steps)
    step_index = 0
    nsteps = int(step_sequence[step_index])

    # SteadyStateSystem global successful-time-step counter.
    nsteps_total = 0
    nsteps_max = int(max(1, getattr(opts, "max_time_step_count", 500)))
    tfactor = float(
        opts.time_step_factor if opts.time_step_factor is not None else opts.time_step_shrink
    )

    attempt = 0

    if opts.verbose:
        print(f"\n{'-'*60}\n{label or 'Newton hibrido'}\n{'-'*60}")

    while True:
        # ---- Intentar Newton estacionario ----
        x_ss, ok_ss, hist_ss, jac = newton_solve(
            steady_fun, x, problem,
            rdt=0.0, x_old=None,
            max_iter=opts.steady_max_iter,
            max_jac_age=ss_jac_age,
            max_damp_iter=opts.max_damp_iter,
            tol=opts.tol,
            jac_eps=opts.jac_eps,
            alpha_min=opts.alpha_min,
            verbose=opts.verbose,
            jac_state=jac,
        )
        history.append(
            {"cycle": attempt, "phase": "steady", "ok": ok_ss, "steps": len(hist_ss)}
        )

        if ok_ss:
            if steady_callback is not None:
                steady_callback(x_ss)
            if opts.verbose:
                Finf = float(np.linalg.norm(residual(x_ss, problem), ord=np.inf))
                print(f"  [ciclo {attempt}] Estacionario OK  ||F||inf={Finf:.4e}")
            return x_ss, True, history

        if opts.verbose:
            print(f"  [ciclo {attempt}] Estacionario FALLO -> time-stepping")
            print(f"  [ciclo {attempt}] Attempt {nsteps} timesteps.")

        # ---- Time-stepping (SteadyStateSystem::timeStep style) ----
        x_old = x.copy()
        x_older: np.ndarray | None = None
        n_done = 0
        successive_failures = 0

        while n_done < nsteps:
            dt_try = float(dt)
            if dt_try <= 0.0:
                break

            rdt = 1.0 / dt_try
            scheme_mode = str(getattr(opts, "transient_scheme", "be")).strip().lower()
            use_bdf2 = scheme_mode in ("bdf2", "be-bdf2", "mixed") and x_older is not None
            transient_order = 2 if use_bdf2 else 1
            scheme = "BDF2" if use_bdf2 else "BE"
            jac_evals_before = int(jac.n_evals) if jac is not None else 0

            x_ts, ok_ts, hist_ts, jac = newton_solve(
                steady_fun, x_old, problem,
                rdt=rdt, x_old=x_old, x_older=x_older,
                transient_order=transient_order,
                max_iter=opts.transient_max_iter,
                max_jac_age=ts_jac_age,
                max_damp_iter=opts.max_damp_iter,
                tol=opts.tol,
                jac_eps=opts.jac_eps,
                alpha_min=opts.alpha_min,
                verbose=opts.verbose,
                jac_state=jac,
            )
            jac_evals_after = int(jac.n_evals) if jac is not None else jac_evals_before

            history.append(
                {
                    "cycle": attempt,
                    "phase": "transient",
                    "ok": ok_ts,
                    "dt": dt_try,
                    "scheme": scheme,
                    "steps": len(hist_ts),
                    "jac_evals_before": jac_evals_before,
                    "jac_evals_after": jac_evals_after,
                }
            )

            if ok_ts:
                successive_failures = 0
                x_prev = x_old
                x = x_ts
                x_old = x_ts
                x_older = x_prev if scheme_mode in ("bdf2", "be-bdf2", "mixed") else None
                n_done += 1
                nsteps_total += 1

                dt = min(opts.max_time_step, dt_try * opts.time_step_grow)

                if opts.verbose:
                    Finf = float(np.linalg.norm(residual(x, problem), ord=np.inf))
                    print(f"    ts {scheme} OK  dt={dt:.2e}  ||F||inf={Finf:.4e}")

                if nsteps_total >= nsteps_max:
                    history.append(
                        {
                            "cycle": attempt,
                            "phase": "transient_abort",
                            "reason": "max_time_step_count",
                            "nsteps_total": int(nsteps_total),
                            "nsteps_max": int(nsteps_max),
                        }
                    )
                    if opts.verbose:
                        print(
                            "  Hibrido abortado: max_time_step_count alcanzado "
                            f"({nsteps_max})."
                        )
                    return x, False, history
            else:
                successive_failures += 1
                x_older = None  # Fall back to Backward Euler on failure
                reset_bad = False
                reset_after = int(max(1, getattr(opts, "reset_bad_after_failures", 3)))
                if successive_failures >= reset_after:
                    x_old = problem.reset_bad_values(x_old)
                    x = x_old.copy()
                    reset_bad = True
                    successive_failures = 0
                    if jac is not None:
                        jac.age = 10000
                else:
                    dt = dt_try * tfactor

                if opts.verbose:
                    if reset_bad:
                        print(f"    ts {scheme} FAIL  reset_bad_values")
                    else:
                        fail_reason = hist_ts[-1] if hist_ts else "Unknown"
                        print(f"    ts {scheme} FAIL  dt->{dt:.2e}  reason={fail_reason}")

                if dt < opts.min_time_step and not reset_bad:
                    history.append(
                        {
                            "cycle": attempt,
                            "phase": "transient_abort",
                            "reason": "min_time_step",
                            "dt": float(dt),
                            "min_time_step": float(opts.min_time_step),
                        }
                    )
                    if opts.verbose:
                        print(
                            "  Hibrido abortado: min_time_step alcanzado "
                            f"({opts.min_time_step:.2e})."
                        )
                    return x, False, history

        # Repeat the final m_steps value for subsequent failed-steady attempts.
        step_index += 1
        if step_index >= len(step_sequence):
            nsteps = int(step_sequence[-1])
        else:
            nsteps = int(step_sequence[step_index])
        dt = min(dt, opts.max_time_step)
        attempt += 1


# ---------------------------------------------------------------------------
#  Refinamiento (Sim1D::refine)
# ---------------------------------------------------------------------------
def _refine_and_solve(
    problem,
    x_ss: np.ndarray,
    opts: SolveOptions,
    deadline: float | None,
    width_check=None,
) -> tuple[np.ndarray, bool, list[dict]]:
    """
    Refinamiento adaptativo + resolución post-refine.
    Replica Sim1D::refine() + llamada recursiva a SteadyStateSystem::solve().
    """
    refiner = AdaptiveRefiner(
        ratio=opts.refine_ratio,
        slope=opts.refine_slope,
        curve=opts.refine_curve,
        prune=opts.refine_prune,
        grid_min=opts.refine_grid_min,
        max_points=opts.refine_max_points,
    )
    log: list[dict] = []

    # Guardar solución steady convergida (m_xlast_ss en Cantera)
    x_last_ss = x_ss.copy()
    z_last_ss = problem.z.copy()
    n_last_ss = problem.n_points

    for pass_idx in range(opts.max_refine_passes):
        if deadline and time.perf_counter() > deadline:
            log.append({"pass": pass_idx, "stopped": "timeout"})
            break

        u, T, Y = unpack_state(x_ss, problem.n_points, problem.n_species)
        profiles = build_freeflame_refiner_profiles(problem, u, T, Y)

        z_old = problem.z.copy()
        z_new, changed, n_ins, n_rem = refiner.refine(
            z_old, profiles, all_Y=Y, j_fixed=problem.j_fixed)

        info: dict[str, Any] = {
            "pass": pass_idx, "changed": changed,
            "n_old": z_old.size, "n_new": z_new.size,
            "n_ins": n_ins, "n_rem": n_rem,
        }

        if not changed:
            info["reason"] = "grid_converged"
            if opts.verbose:
                print("no new points needed in flame")
            log.append(info)
            return x_ss, True, log

        # Guardar la solución del último steady antes de cambiar la malla
        x_last_ss = x_ss.copy()
        z_last_ss = z_old.copy()
        n_last_ss = z_old.size

        # Interpolar al nuevo grid
        x_new = interpolate_state(x_ss, z_old, z_new, problem.n_species)

        # Actualizar malla y backend
        problem.z = z_new
        problem.n_points = int(z_new.size)
        problem.width = float(z_new[-1] - z_new[0])
        problem.backend = SpeciesBackend(problem)
        x_new = problem.reset_bad_values(x_new)

        _, T_new, Y_new = unpack_state(x_new, problem.n_points, problem.n_species)
        Y_new = problem._sanitize_Y(Y_new)
        u_new, _, _ = unpack_state(x_new, problem.n_points, problem.n_species)
        x_new = pack_state(u_new, T_new, Y_new)

        problem.solve_energy = True
        problem.setup_fixed_temperature(T_profile=T_new)

        x_try, ok, hist_ref = _hybrid_newton(
            problem,
            x_new,
            opts,
            label=f"Post-refine pass {pass_idx}",
            steady_callback=width_check,
        )
        info["solve_ok"] = ok
        info["Finf_after"] = float(np.linalg.norm(
            residual(x_try, problem), ord=np.inf))
        log.append(info)

        if ok:
            x_ss = x_try
        else:
            # Restaurar último steady (como Sim1D hace con m_xlast_ss)
            if opts.verbose:
                print(f"  Refine pass {pass_idx}: solve fallo -> restaurando ultimo steady")
            problem.z = z_last_ss
            problem.n_points = n_last_ss
            problem.width = float(z_last_ss[-1] - z_last_ss[0])
            problem.backend = SpeciesBackend(problem)
            _, T_old, _ = unpack_state(x_last_ss, n_last_ss, problem.n_species)
            problem.setup_fixed_temperature(T_profile=T_old)
            info["restored"] = True
            return x_last_ss, True, log

    return x_ss, True, log


def _solve_auto_stages(
    problem,
    x: np.ndarray,
    opts: SolveOptions,
    deadline: float | None,
    refine_grid: bool | None = None,
    width_check=None,
) -> tuple[np.ndarray, bool, dict[str, Any]]:
    """
    One auto-solve attempt on the current domain:
      A) energy ON
      B) energy OFF (fallback)
      C) energy RE-ON
      refine (optional; disabled by caller in fixed-grid auto path)
    """
    report: dict[str, Any] = {}
    x = np.asarray(x, dtype=float).copy()
    solved = False

    _, T0, _ = unpack_state(x, problem.n_points, problem.n_species)
    problem.setup_fixed_temperature(T_profile=T0)

    # ---- Stage A: energía ON ----
    problem.solve_energy = True
    x_a, ok_a, hist_a = _hybrid_newton(
        problem, x, opts, label="Stage A: energia ON", steady_callback=width_check)
    report["stage_A"] = {"ok": ok_a, "steps": len(hist_a)}

    if ok_a:
        x = x_a
        solved = True
    else:
        # ---- Stage B: energía OFF ----
        problem.solve_energy = False
        _, T_frz, _ = unpack_state(x, problem.n_points, problem.n_species)
        problem.T_profile_fixed = T_frz.copy()
        problem.setup_fixed_temperature(T_profile=T_frz)

        x_b, ok_b, hist_b = _hybrid_newton(
            problem, x, opts, label="Stage B: energia OFF", steady_callback=width_check)
        report["stage_B"] = {"ok": ok_b, "steps": len(hist_b)}

        if ok_b:
            x = x_b
            # ---- Stage C: reactivar energía ----
            problem.solve_energy = True
            _, T_re, _ = unpack_state(x, problem.n_points, problem.n_species)
            problem.setup_fixed_temperature(T_profile=T_re)

            x_c, ok_c, hist_c = _hybrid_newton(
                problem, x, opts, label="Stage C: energia RE-ON", steady_callback=width_check)
            report["stage_C"] = {"ok": ok_c, "steps": len(hist_c)}
            x = x_c
            solved = ok_c
        else:
            solved = False

    # ---- Refinamiento iterativo (Sim1D::solve while loop) ----
    if refine_grid is None:
        refine_grid = bool(opts.refine_grid)
    if solved and refine_grid:
        if deadline and time.perf_counter() > deadline:
            report["refine"] = "timeout_before_refine"
        else:
            x, ok_ref, ref_log = _refine_and_solve(
                problem, x, opts, deadline, width_check=width_check)
            report["refine"] = ref_log
            solved = ok_ref

    return x, bool(solved), report


def _solve_refine_energy_on(
    problem,
    x: np.ndarray,
    opts: SolveOptions,
    deadline: float | None,
    width_check=None,
) -> tuple[np.ndarray, bool, dict[str, Any]]:
    """
    Refine stage aligned with Cantera auto path:
    - Keep energy enabled
    - No ON/OFF/ON fallback sequence
    - Run steady/hybrid solve, then iterative mesh refinement
    """
    report: dict[str, Any] = {}
    x = np.asarray(x, dtype=float).copy()

    _, T0, _ = unpack_state(x, problem.n_points, problem.n_species)
    problem.solve_energy = True
    problem.setup_fixed_temperature(T_profile=T0)

    x_ss, ok_ss, hist_ss = _hybrid_newton(
        problem, x, opts, label="Refine Stage: energia ON", steady_callback=width_check)
    report["refine_stage_steady"] = {"ok": ok_ss, "steps": len(hist_ss)}

    if not ok_ss:
        return x_ss, False, report

    if not bool(opts.refine_grid):
        return x_ss, True, report

    if deadline and time.perf_counter() > deadline:
        report["refine"] = "timeout_before_refine"
        return x_ss, False, report

    x_ref, ok_ref, ref_log = _refine_and_solve(
        problem, x_ss, opts, deadline, width_check=width_check)
    report["refine"] = ref_log
    return x_ref, bool(ok_ref), report


def _domain_too_narrow(problem, x: np.ndarray, slope_tol: float = 0.02) -> tuple[bool, dict[str, float]]:
    """
    Mimic Cantera FreeFlame.solve(auto=True) width check:
      mRef = (T[-1]-T[0]) / (x[-1]-x[0])
      mLeft = (T[1]-T[0]) / (x[1]-x[0]) / mRef
      mRight = (T[-3]-T[-1]) / (x[-3]-x[-1]) / mRef
      too_narrow if mLeft > tol or mRight > tol
    """
    z = np.asarray(problem.z, dtype=float)
    n = int(z.size)
    _, T, _ = unpack_state(x, problem.n_points, problem.n_species)

    metrics = {
        "m_ref": 0.0,
        "m_left": 0.0,
        "m_right": 0.0,
        "n_points": float(n),
        "width_m": float(z[-1] - z[0]) if n >= 2 else 0.0,
    }

    if n < 4:
        return False, metrics

    span = float(z[-1] - z[0])
    if span <= 0.0:
        return False, metrics

    m_ref = float((T[-1] - T[0]) / span)
    metrics["m_ref"] = m_ref
    if abs(m_ref) < 1.0:
        return False, metrics

    m_left = float(abs((T[1] - T[0]) / (z[1] - z[0]) / m_ref))
    m_right = float(abs((T[-3] - T[-1]) / (z[-3] - z[-1]) / m_ref))
    metrics["m_left"] = m_left
    metrics["m_right"] = m_right

    too_narrow = (m_left > slope_tol) or (m_right > slope_tol)
    return bool(too_narrow), metrics


def _expand_domain_and_reseed(problem, x: np.ndarray, factor: float = 2.0) -> np.ndarray:
    """
    Expand current domain and interpolate solution onto the new grid.
    """
    factor = float(factor)
    if factor <= 1.0:
        return np.asarray(x, dtype=float).copy()

    z_old = np.asarray(problem.z, dtype=float).copy()
    z_new = z_old * factor

    # Expanding the domain scales the existing grid coordinates; the state
    # values stay attached to their grid indices so edge slopes decrease.
    x_new = np.asarray(x, dtype=float).copy()
    _assign_grid(problem, z_new)

    x_new = problem.reset_bad_values(x_new)
    _, T_new, _ = unpack_state(x_new, problem.n_points, problem.n_species)
    problem.setup_fixed_temperature(T_profile=T_new)
    return x_new


def _assign_grid(problem, z_new: np.ndarray) -> None:
    """Apply a new grid to the problem and refresh backend bookkeeping."""
    z_new = np.asarray(z_new, dtype=float)
    if z_new.ndim != 1 or z_new.size < 2:
        raise ValueError("Grid invalida: se requieren al menos 2 puntos.")
    problem.z = z_new
    problem.n_points = int(z_new.size)
    problem.width = float(z_new[-1] - z_new[0])
    problem.backend = SpeciesBackend(problem)


def _bootstrap_grid_sequence(problem, opts: SolveOptions, restart_mode: bool) -> list[int]:
    """
    Grid schedule that mirrors Cantera auto mode:
      data/restart -> [current_n]
      no restart   -> [current_n, 12, 24, 48] (deduplicated, bounded)
    """
    n0 = int(problem.n_points)
    if restart_mode or not bool(getattr(opts, "auto_bootstrap_grids", True)):
        return [n0]

    raw = [n0]
    raw.extend(int(v) for v in getattr(opts, "bootstrap_grid_points", (12, 24, 48)))
    max_pts = int(max(2, getattr(opts, "bootstrap_max_grid_points", 1000)))

    out: list[int] = []
    seen: set[int] = set()
    for n in raw:
        if n < 2 or n > max_pts:
            continue
        if n in seen:
            continue
        out.append(n)
        seen.add(n)
    return out or [n0]


def _seed_state_on_fixed_grid(problem, x: np.ndarray | None, n_points: int,
                              opts: SolveOptions, use_initial_guess: bool) -> np.ndarray:
    """
    Prepare state on a fixed grid for auto bootstrap stage.
    """
    n_points = int(max(2, n_points))
    z_old = np.asarray(problem.z, dtype=float).copy()

    if n_points != int(z_old.size):
        z_new = np.linspace(float(z_old[0]), float(z_old[-1]), n_points, dtype=float)
        _assign_grid(problem, z_new)
        if use_initial_guess or x is None:
            x_new = problem.make_initial_guess(u_left_guess=opts.u_left_guess)
        else:
            x_new = interpolate_state(np.asarray(x, dtype=float), z_old, z_new, problem.n_species)
    else:
        if use_initial_guess or x is None:
            x_new = problem.make_initial_guess(u_left_guess=opts.u_left_guess)
        else:
            x_new = np.asarray(x, dtype=float).copy()

    x_new = problem.reset_bad_values(x_new)
    x_new = _apply_fixed_temperature_anchor(problem, x_new, T_target=problem.anchor_T)
    return x_new


def _apply_fixed_temperature_anchor(problem, x: np.ndarray,
                                    T_target: float | None = None) -> np.ndarray:
    """
    Mimic Sim1D::setFixedTemperature for free flames.

    If target temperature lies between two grid points and is not close to
    existing points, insert an extra grid point at the interpolated location.
    """
    x = np.asarray(x, dtype=float).copy()
    n_pts = int(problem.n_points)
    n_sp = int(problem.n_species)
    if n_pts < 2:
        return x

    u, T, Y = unpack_state(x, n_pts, n_sp)
    z = np.asarray(problem.z, dtype=float)
    t_fix = float(problem.anchor_T if T_target is None else T_target)

    j_existing: int | None = None
    m_insert: int | None = None
    z_fixed = None

    for m in range(n_pts - 1):
        t1 = float(T[m])
        t2 = float(T[m + 1])
        z1 = float(z[m])
        z2 = float(z[m + 1])
        thresh = min(1.0, 1.0e-1 * (t2 - t1))

        if abs(t_fix - t1) <= thresh:
            j_existing = m
            z_fixed = z1
            break
        if abs(t2 - t_fix) <= thresh:
            j_existing = m + 1
            z_fixed = z2
            break
        if (t1 < t_fix) and (t_fix < t2):
            m_insert = m
            z_fixed = (z1 - z2) / (t1 - t2) * (t_fix - t2) + z2
            break

    if m_insert is not None and z_fixed is not None:
        m = int(m_insert)
        z1 = float(z[m])
        z2 = float(z[m + 1])
        if abs(z1 - z2) > 0.0:
            w = (float(z_fixed) - z2) / (z1 - z2)
        else:
            w = 0.5

        z_new = np.insert(z, m + 1, float(z_fixed))
        u_ins = w * (u[m] - u[m + 1]) + u[m + 1]
        T_ins = w * (T[m] - T[m + 1]) + T[m + 1]
        Y_ins = w * (Y[:, m] - Y[:, m + 1]) + Y[:, m + 1]

        u_new = np.concatenate((u[:m + 1], np.array([u_ins], dtype=float), u[m + 1:]))
        T_new = np.concatenate((T[:m + 1], np.array([T_ins], dtype=float), T[m + 1:]))
        Y_new = np.concatenate((Y[:, :m + 1], Y_ins[:, None], Y[:, m + 1:]), axis=1)
        Y_new = problem._sanitize_Y(Y_new)

        _assign_grid(problem, z_new)
        x = pack_state(u_new, T_new, Y_new)
        x = problem.reset_bad_values(x)

    _, T_eff, _ = unpack_state(x, problem.n_points, problem.n_species)
    if z_fixed is not None:
        problem.setup_fixed_temperature(T_fixed=t_fix, z_fixed=float(z_fixed), T_profile=T_eff)
    elif j_existing is not None:
        problem.setup_fixed_temperature(T_fixed=t_fix, z_fixed=float(problem.z[j_existing]), T_profile=T_eff)
    else:
        problem.setup_fixed_temperature(T_fixed=t_fix, T_profile=T_eff)

    return x


def _refine_grid_once(problem, x: np.ndarray, opts: SolveOptions) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Single grid-refine operation without solving (mimics flame.refine() call
    done by Cantera immediately after auto-domain expansion).
    """
    info: dict[str, Any] = {}
    refiner = AdaptiveRefiner(
        ratio=opts.refine_ratio,
        slope=opts.refine_slope,
        curve=opts.refine_curve,
        prune=opts.refine_prune,
        grid_min=opts.refine_grid_min,
        max_points=opts.refine_max_points,
    )

    z_old = problem.z.copy()
    u, T, Y = unpack_state(x, problem.n_points, problem.n_species)
    profiles = build_freeflame_refiner_profiles(problem, u, T, Y)
    z_new, changed, n_ins, n_rem = refiner.refine(
        z_old, profiles, all_Y=Y, j_fixed=problem.j_fixed)

    info["changed"] = bool(changed)
    info["n_old"] = int(z_old.size)
    info["n_new"] = int(z_new.size)
    info["n_ins"] = int(n_ins)
    info["n_rem"] = int(n_rem)

    if not changed:
        return x, info

    x_new = interpolate_state(x, z_old, z_new, problem.n_species)
    _assign_grid(problem, z_new)
    x_new = problem.reset_bad_values(x_new)
    _, T_new, _ = unpack_state(x_new, problem.n_points, problem.n_species)
    problem.setup_fixed_temperature(T_profile=T_new)
    return x_new, info


# ---------------------------------------------------------------------------
#  Interfaz principal (Sim1D::solve con auto=True)
# ---------------------------------------------------------------------------
def solve_free_flame(
    problem,
    x0: np.ndarray | None = None,
    options: SolveOptions | None = None,
) -> tuple[np.ndarray, bool, dict[str, Any]]:
    """
    Replica Sim1D::solve(loglevel, refine_grid=True) con estrategia auto:
      1. Intentar con energía ON.
      2. Si falla: congelar T, resolver OFF; luego reactivar energía.
      3. Si convergió: refinar malla y resolver de nuevo (iterado).
      4. Si el dominio queda angosto, expandir y repetir (auto-width check).
    """
    opts = options or SolveOptions()
    report: dict[str, Any] = {"passes": [], "domain_checks": []}
    t_start = time.perf_counter()
    deadline = (t_start + opts.max_total_time_s
                if np.isfinite(opts.max_total_time_s) else None)

    if problem.backend is None:
        problem.backend = SpeciesBackend(problem)

    restart_mode = x0 is not None
    x = None if x0 is None else problem.reset_bad_values(np.asarray(x0, dtype=float))
    solved = False

    n_expand_tries = int(max(1, getattr(opts, "max_domain_expansions", 12)))
    if not bool(getattr(opts, "domain_auto_expand", True)):
        n_expand_tries = 1
    max_exp = n_expand_tries - 1
    slope_tol = float(getattr(opts, "domain_edge_slope_tol", 0.02))

    # Mirror FreeFlame.solve(auto=True): exactly N attempts total (default 12).
    for expand_pass in range(n_expand_tries):
        if deadline and time.perf_counter() > deadline:
            report["timeout_before_solve"] = True
            break

        grid_targets = _bootstrap_grid_sequence(problem, opts, restart_mode)
        report["grid_targets"] = [int(v) for v in grid_targets]
        report.setdefault("outer_passes", []).append(
            {
                "expand_pass": int(expand_pass),
                "grid_targets": [int(v) for v in grid_targets],
                "width_m": float(problem.width),
                "n_points": int(problem.n_points),
            }
        )

        x_work = x

        try:
            for grid_pass, n_grid in enumerate(grid_targets):
                if deadline and time.perf_counter() > deadline:
                    report["timeout_before_solve"] = True
                    break

                # Keep and propagate the latest state unless there is no seed yet.
                # This preserves the post-expansion solution path like Cantera.
                use_initial_guess = (x_work is None)
                x_work = _seed_state_on_fixed_grid(
                    problem, x_work, n_grid, opts, use_initial_guess=use_initial_guess)
                report.setdefault("grid_attempts", []).append(
                    {
                        "expand_pass": int(expand_pass),
                        "grid_pass": int(grid_pass),
                        "grid_points": int(problem.n_points),
                        "width_m": float(problem.width),
                        "restart_mode": bool(restart_mode),
                    }
                )

                def width_check(x_state: np.ndarray, stage_mode: str) -> None:
                    narrow, metrics = _domain_too_narrow(
                        problem, x_state, slope_tol=slope_tol)
                    m = dict(metrics)
                    m["expand_pass"] = int(expand_pass)
                    m["grid_pass"] = int(grid_pass)
                    m["grid_points"] = int(problem.n_points)
                    m["stage_mode"] = str(stage_mode)
                    m["too_narrow"] = bool(narrow)
                    report["domain_checks"].append(m)
                    if narrow:
                        raise DomainTooNarrowError(x_state, m)

                x_work, solved_fixed, step_report = _solve_auto_stages(
                    problem,
                    x_work,
                    opts,
                    deadline,
                    refine_grid=False,
                    width_check=lambda x_state: width_check(x_state, "fixed_auto"),
                )
                step_report["expand_pass"] = int(expand_pass)
                step_report["grid_pass"] = int(grid_pass)
                step_report["grid_points"] = int(problem.n_points)
                step_report["stage_mode"] = "fixed_auto"
                report["passes"].append(step_report)
                for k in ("stage_A", "stage_B", "stage_C", "refine"):
                    if k in step_report:
                        report[k] = step_report[k]

                if not solved_fixed:
                    if restart_mode:
                        break
                    continue

                if bool(opts.refine_grid):
                    x_work, solved_refine, step_report = _solve_refine_energy_on(
                        problem,
                        x_work,
                        opts,
                        deadline,
                        width_check=lambda x_state: width_check(x_state, "refine_energy"),
                    )
                    step_report["expand_pass"] = int(expand_pass)
                    step_report["grid_pass"] = int(grid_pass)
                    step_report["grid_points"] = int(problem.n_points)
                    step_report["stage_mode"] = "refine_energy"
                    report["passes"].append(step_report)
                    for k in ("stage_A", "stage_B", "stage_C", "refine"):
                        if k in step_report:
                            report[k] = step_report[k]
                    solved = bool(solved_refine)
                else:
                    solved = True

                if solved:
                    x = x_work
                    break

            if solved:
                break

            x = x_work
            if restart_mode:
                break

        except DomainTooNarrowError as exc:
            x = exc.x
            report.setdefault("expansion_events", []).append(
                {
                    "expand_pass": int(expand_pass),
                    "metrics": dict(exc.metrics),
                }
            )

            x = _expand_domain_and_reseed(
                problem, x, factor=float(getattr(opts, "domain_expand_factor", 2.0)))
            if opts.verbose:
                print(
                    "Expanding domain to accommodate flame thickness. "
                    f"New width: {problem.width:.6g} m"
                )

            if bool(opts.refine_grid):
                x, ref_info = _refine_grid_once(problem, x, opts)
                ref_info["expand_pass"] = int(expand_pass)
                ref_info["stage_mode"] = str(exc.metrics.get("stage_mode", "unknown"))
                report.setdefault("post_expand_refine", []).append(ref_info)

            if expand_pass >= max_exp:
                report["domain_expansion_limit_reached"] = True
                break

            continue

    if x is None:
        x = _seed_state_on_fixed_grid(
            problem, None, int(problem.n_points), opts, use_initial_guess=True)

    report["n_points_final"] = int(problem.n_points)
    report["solved"] = bool(solved)
    report["total_time_s"] = float(time.perf_counter() - t_start)
    report["Finf_final"] = float(np.linalg.norm(
        residual(x, problem), ord=np.inf))

    if opts.verbose:
        print(f"\n{'='*60}")
        print(f"  Resuelto: {solved}  n_pts={problem.n_points}"
              f"  ||F||inf={report['Finf_final']:.4e}"
              f"  t={report['total_time_s']:.1f}s")
        print(f"{'='*60}")

    return x, bool(solved), report

