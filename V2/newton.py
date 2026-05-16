"""
newton.py – Solver Newton amortiguado, fiel a MultiNewton.cpp de Cantera.

Estructura que reproduce MultiNewton::solve():
  1. Verifica edad del Jacobiano; reconstruye si es necesario.
  2. Calcula el paso de Newton sin amortiguamiento (MultiNewton::step).
  3. Aplica updateTransient si rdt > 0.
  4. Calcula damping con MultiNewton::dampStep:
     - boundStep: escala el paso para mantener variables en bounds.
     - Bucle de amortiguamiento: reduce alpha hasta que ||step1|| < ||step0||.
  5. Si dampStep falla con Jacobiano viejo, fuerza recomputación.
  6. Criterio de convergencia: norma ponderada del paso < 1.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field

import numpy as np

from jacobian import (build_jacobian_steady, update_transient,
                      factorize, solve_linear)
from state import build_transient_mask


# ---------------------------------------------------------------------------
#  Norma ponderada (OneDim::weightedNorm)
# ---------------------------------------------------------------------------
def weighted_norm(step: np.ndarray, x: np.ndarray, problem) -> float:
    """
    Norma ponderada de Cantera:
      w_{d,n} = rtol * mean(|x_n|) + atol
      ||step|| = sqrt( sum(step/w)^2 / N )
    """
    n_pts = int(problem.n_points)
    n_sp = int(problem.n_species)
    nv = 2 + n_sp
    n_total = n_pts * nv

    if step.size != n_total or x.size != n_total:
        # Fallback simple
        return float(np.linalg.norm(step) / max(np.linalg.norm(x), 1e-300))

    rtol_ss = float(getattr(problem, "steady_rtol", 1e-4))
    atol_ss = float(getattr(problem, "steady_atol", 1e-9))
    rtol_ts = float(getattr(problem, "transient_rtol", 1e-4))
    atol_ts = float(getattr(problem, "transient_atol", 1e-11))

    x_r = x.reshape(n_pts, nv)
    step_r = step.reshape(n_pts, nv)
    sumsq = 0.0
    for v in range(nv):
        # Tolerancias por componente
        is_Y = v >= 2
        rtol = rtol_ts if is_Y else rtol_ss
        atol = atol_ts if is_Y else atol_ss
        esum = float(np.sum(np.abs(x_r[:, v])))
        ewt = rtol * esum / n_pts + atol
        ewt = max(ewt, 1e-300)
        fs = step_r[:, v] / ewt
        sumsq += float(np.dot(fs, fs))
    return float(np.sqrt(sumsq / n_total))


# ---------------------------------------------------------------------------
#  Cálculo de fbound (MultiNewton::boundStep)
# ---------------------------------------------------------------------------
def bound_step(x0: np.ndarray, step0: np.ndarray, problem) -> float:
    """
    Calcula el factor máximo alpha tal que x0 + alpha*step0 permanece
    dentro de los bounds de cada componente.
    Reproduce MultiNewton::boundStep.
    """
    n_pts = int(problem.n_points)
    n_sp = int(problem.n_species)
    nv = 2 + n_sp

    # Bounds por componente (columna en layout por-punto)
    T_lo = float(getattr(problem, "T_lower_bound", 200.0))
    T_hi = float(getattr(problem, "T_upper_bound", 2.0 * 3000.0))
    Y_lo = float(getattr(problem, "Y_lower_bound", -1e-7))
    Y_hi = 1.0e5

    lower = np.full(nv, -1e20)
    upper = np.full(nv, 1e20)
    lower[1] = T_lo   # T
    upper[1] = T_hi
    lower[2:] = Y_lo  # Y_k
    upper[2:] = Y_hi

    fbound = 1.0
    x_r = x0.reshape(n_pts, nv)
    s_r = step0.reshape(n_pts, nv)

    for v in range(nv):
        xv = x_r[:, v]
        sv = s_r[:, v]
        xnv = xv + sv

        m_above = (sv > 0) & (xnv > upper[v])
        if np.any(m_above):
            cand = (upper[v] - xv[m_above]) / sv[m_above]
            fbound = min(fbound, float(np.min(cand)))

        m_below = (sv < 0) & (xnv < lower[v])
        if np.any(m_below):
            cand = (xv[m_below] - lower[v]) / (-sv[m_below])
            fbound = min(fbound, float(np.min(cand)))

    return max(0.0, fbound)


# ---------------------------------------------------------------------------
#  Estado del Jacobiano (guarda J, LU, diagonal y edad)
# ---------------------------------------------------------------------------
@dataclass
class JacobianState:
    J: object = None        # sparse CSR
    lu: object = None       # factorización LU
    ss_diag: np.ndarray = field(default_factory=lambda: np.empty(0))
    age: int = 10000
    n_evals: int = 0
    last_rdt: float | None = None

    def is_stale(self, max_age: int) -> bool:
        return self.J is None or self.age >= max_age


# ---------------------------------------------------------------------------
#  Newton solver principal
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
    tol: float = 1.0,      # criterio sobre norma ponderada del paso Newton
    jac_eps: float = 1e-8,
    alpha_min: float = 1e-10,
    verbose: bool = False,
    jac_state: JacobianState | None = None,
) -> tuple[np.ndarray, bool, list[dict], JacobianState]:
    """
    Newton amortiguado: reproduce MultiNewton::solve().

    Parámetros
    ----------
    steady_fun : función F(x, problem) del residual ESTACIONARIO
    x0         : solución inicial
    problem    : FreeFlameProblem
    rdt        : 0 = estacionario;  >0 = transitorio (rdt = 1/dt)
    x_old      : solucion en paso n   (requerida si rdt > 0)
    x_older    : solucion en paso n-1 (requerida para BDF2)
    transient_order : 1 (BE) o 2 (BDF2)
    tol        : convergencia cuando weighted_norm(step) < tol (Cantera usa 1.0)
    jac_state  : estado previo del Jacobiano (para reuso entre llamadas)

    Retorna
    -------
    x1        : solución tras la última iteración
    converged : True si weighted_norm del paso < tol
    history   : lista de dicts con diagnóstico por iteración
    jac_state : estado actualizado del Jacobiano
    """
    x = np.asarray(x0, dtype=float)
    n = x.size
    history: list[dict] = []

    if jac_state is None:
        jac_state = JacobianState()

    # Función completa (estacionaria + transitoria)
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

    use_bdf2 = int(transient_order) >= 2 and x_older is not None
    transient_alpha = 1.5 if use_bdf2 else 1.0
    rdt_curr = float(rdt) * transient_alpha
    rdt_changed = (
        jac_state.last_rdt is None
        or not np.isclose(jac_state.last_rdt, rdt_curr, rtol=1e-12, atol=0.0)
    )
    force_new_jac = bool(rdt_changed)
    n_jac_reeval = 0
    converged = False
    status = -1

    for it in range(max_iter):
        # ---- Reconstruir Jacobiano si es necesario ----
        if force_new_jac or jac_state.is_stale(max_jac_age):
            try:
                J_ss, ss_diag = build_jacobian_steady(steady_fun, x, problem,
                                                      eps=jac_eps)
                jac_state.ss_diag = ss_diag
                if rdt_curr > 0.0:
                    J_t = update_transient(J_ss, mask, rdt_curr, inplace=True)
                else:
                    J_t = J_ss
                jac_state.J = J_t
                jac_state.lu = factorize(J_t)
                jac_state.age = 0
                jac_state.n_evals += 1
                jac_state.last_rdt = rdt_curr
            except Exception as exc:
                history.append({"iter": it, "status": "jac_fail", "error": str(exc)})
                status = -4
                break
            force_new_jac = False

        # ---- Paso de Newton (step = -J^-1 * F) ----
        F = full_fun(x)
        if not np.all(np.isfinite(F)):
            history.append({"iter": it, "status": "nonfinite_F"})
            break

        try:
            step = solve_linear(jac_state.lu, -F)
        except Exception:
            force_new_jac = True
            continue

        if not np.all(np.isfinite(step)):
            history.append({"iter": it, "status": "nonfinite_step"})
            break

        jac_state.age += 1

        # ---- Norma del paso sin amortiguar ----
        s0 = weighted_norm(step, x, problem)

        # ---- fbound: mantener dentro de bounds ----
        fbound = bound_step(x, step, problem)
        if fbound < 1e-10:
            history.append({"iter": it, "status": "bound_step_too_small",
                            "s0": s0, "fbound": fbound})
            status = -3
            break

        # ---- Bucle de amortiguamiento (dampStep) ----
        alpha = fbound
        x1 = np.empty_like(x)
        step1 = np.empty_like(step)
        s1 = 1e30
        damp_ok = False

        for _ in range(max_damp_iter):
            if alpha < alpha_min:
                break
            x1 = x + alpha * step
            try:
                F1 = full_fun(x1)
            except Exception:
                alpha /= damp_factor
                continue
            if not np.all(np.isfinite(F1)):
                alpha /= damp_factor
                continue
            try:
                step1 = solve_linear(jac_state.lu, -F1)
            except Exception:
                alpha /= damp_factor
                continue
            s1 = weighted_norm(step1, x1, problem)
            if s1 < 1.0 or s1 < s0:
                damp_ok = True
                break
            alpha /= damp_factor

        normF = float(np.linalg.norm(F, ord=np.inf))

        if damp_ok:
            x = x1
            converged = bool(s1 < tol)
            history.append({
                "iter": it, "status": "ok" if converged else "step",
                "normF": normF, "s0": s0, "s1": s1,
                "alpha": alpha, "jac_age": jac_state.age,
            })
            if verbose:
                print(f"  Newton it={it:3d} ||F||∞={normF:.4e} "
                      f"s0={s0:.3e} s1={s1:.3e} α={alpha:.3e} "
                      f"jac_age={jac_state.age}")
            if converged:
                status = 1
                break
            status = 0
        else:
            history.append({
                "iter": it, "status": "no_damp",
                "normF": normF, "s0": s0, "jac_age": jac_state.age,
            })
            if jac_state.age > 1:
                force_new_jac = True
                n_jac_reeval += 1
                if verbose:
                    print(f"  Newton it={it:3d} no damping → forzando nuevo Jacobiano")
                if n_jac_reeval > 3:
                    status = -2
                    break
                continue
            else:
                status = -2
                break

    if status != 1:
        x = np.asarray(x0, dtype=float).copy()

    return x, bool(status == 1), history, jac_state
