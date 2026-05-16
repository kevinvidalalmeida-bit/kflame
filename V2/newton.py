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
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from jacobian import build_jacobian_steady, update_transient, factorize, solve_linear
from state import build_transient_mask


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
def bound_step(x0: np.ndarray, step0: np.ndarray, problem) -> float:
    """
    Largest alpha in [0,1] such that x0 + alpha * step0 stays within bounds.
    """
    n_pts = int(problem.n_points)
    n_sp = int(problem.n_species)
    nv = 2 + n_sp

    t_lo = float(getattr(problem, "T_lower_bound", 200.0))
    t_hi = float(getattr(problem, "T_upper_bound", 2.0 * 3000.0))
    y_lo = float(getattr(problem, "Y_lower_bound", -1e-7))
    y_hi = 1.0e5

    lower = np.full(nv, -1e20)
    upper = np.full(nv, 1e20)
    lower[1] = t_lo
    upper[1] = t_hi
    lower[2:] = y_lo
    upper[2:] = y_hi

    fbound = 1.0
    x_r = x0.reshape(n_pts, nv)
    s_r = step0.reshape(n_pts, nv)

    for v in range(nv):
        xv = x_r[:, v]
        sv = s_r[:, v]
        xnv = xv + sv

        above = (sv > 0.0) & (xnv > upper[v])
        if np.any(above):
            cand = (upper[v] - xv[above]) / sv[above]
            fbound = min(fbound, float(np.min(cand)))

        below = (sv < 0.0) & (xnv < lower[v])
        if np.any(below):
            cand = (xv[below] - lower[v]) / (-sv[below])
            fbound = min(fbound, float(np.min(cand)))

    return max(0.0, fbound)


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
        return self.J is None or self.lu is None or self.age >= max_age


def _transient_alpha(transient_order: int, x_older: np.ndarray | None) -> float:
    use_bdf2 = int(transient_order) >= 2 and x_older is not None
    return 1.5 if use_bdf2 else 1.0


def _build_linear_model(steady_fun, x: np.ndarray, problem,
                        jac_eps: float, mask: np.ndarray,
                        rdt_curr: float) -> tuple[object, object, np.ndarray]:
    j_ss, ss_diag = build_jacobian_steady(steady_fun, x, problem, eps=jac_eps)
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
        from residual import residual as _res
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
        fbound = bound_step(x, step0, problem)

        if fbound < 1e-10:
            history.append({
                "iter": it,
                "status": "bound_step_too_small",
                "s0": s0,
                "fbound": fbound,
            })
            status = -3
            break

        # MultiNewton::dampStep equivalent
        alpha = fbound
        damp_ok = False
        x1 = x.copy()
        s1 = float("inf")
        normf = float(np.linalg.norm(f, ord=np.inf))

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
                n_jac_reeval += 1
                if verbose:
                    print(f"  Newton it={it:3d} no damping -> force new Jacobian")
                if n_jac_reeval > 3:
                    status = -2
                    break
                continue

            status = -2
            break

    if status != 1:
        # Cantera semantics: return unchanged input state on failure
        x = np.asarray(x0, dtype=float).copy()

    return x, bool(status == 1), history, jac_state
