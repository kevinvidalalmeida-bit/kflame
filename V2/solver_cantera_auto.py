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
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from newton import JacobianState
from mesh import AdaptiveRefiner, build_freeflame_refiner_profiles
from newton import newton_solve
from residual import residual
from species_backend import SpeciesBackend
from state import pack_state, unpack_state, interpolate_state


# ---------------------------------------------------------------------------
#  Opciones
# ---------------------------------------------------------------------------
@dataclass
class SolveOptions:
    # Conjetura inicial
    u_left_guess: float = 1.00

    # Newton estacionario
    steady_max_iter: int = 20
    max_jac_age: int = 20
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
    transient_max_iter: int = 20
    max_time_step_count: int = 500

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
    restart_after_expand_failure: bool = False

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
                   label: str = "") -> tuple[np.ndarray, bool, list[dict]]:
    """
    Ciclos steady-Newton / time-step hasta convergencia.
    Devuelve (x, converged, history).
    """
    x = np.asarray(x0, dtype=float).copy()
    history: list[dict] = []

    # Jacobian finite-difference settings (Cantera-like).
    problem.jacobian_rel_perturb = float(getattr(opts, "jac_eps", 1e-5))
    problem.jacobian_abs_perturb = float(getattr(opts, "jac_abs_perturb", 1e-10))
    problem.jacobian_threshold = float(getattr(opts, "jac_threshold", 0.0))
    problem.jacobian_mode = str(getattr(opts, "jacobian_mode", "cantera_local"))

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
            use_bdf2 = x_older is not None
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
                verbose=False,
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
                x_older = x_prev
                n_done += 1
                nsteps_total += 1

                # Increase dt only when no Jacobian re-evaluation occurred.
                if jac_evals_after == jac_evals_before:
                    dt = min(opts.max_time_step, dt_try * opts.time_step_grow)
                else:
                    dt = min(opts.max_time_step, dt_try)

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
                reset_bad = False
                if successive_failures > 2:
                    x_old = problem.reset_bad_values(x_old)
                    if x_older is not None:
                        x_older = problem.reset_bad_values(x_older)
                    successive_failures = 0
                    reset_bad = True
                else:
                    dt = dt_try * tfactor

                if opts.verbose:
                    if reset_bad:
                        print(f"    ts {scheme} FAIL  reset_bad_values")
                    else:
                        print(f"    ts {scheme} FAIL  dt->{dt:.2e}")

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
            problem, x_new, opts, label=f"Post-refine pass {pass_idx}")
        info["solve_ok"] = ok
        info["Finf_after"] = float(np.linalg.norm(
            residual(x_try, problem), ord=np.inf))
        log.append(info)

        if ok:
            x_ss = x_try
            if width_check is not None:
                width_check(x_ss)
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
        problem, x, opts, label="Stage A: energia ON")
    report["stage_A"] = {"ok": ok_a, "steps": len(hist_a)}

    if ok_a:
        x = x_a
        if width_check is not None:
            width_check(x)
        solved = True
    else:
        # ---- Stage B: energía OFF ----
        problem.solve_energy = False
        _, T_frz, _ = unpack_state(x, problem.n_points, problem.n_species)
        problem.T_profile_fixed = T_frz.copy()
        problem.setup_fixed_temperature(T_profile=T_frz)

        x_b, ok_b, hist_b = _hybrid_newton(
            problem, x, opts, label="Stage B: energia OFF")
        report["stage_B"] = {"ok": ok_b, "steps": len(hist_b)}

        if ok_b:
            x = x_b
            if width_check is not None:
                width_check(x)
            # ---- Stage C: reactivar energía ----
            problem.solve_energy = True
            _, T_re, _ = unpack_state(x, problem.n_points, problem.n_species)
            problem.setup_fixed_temperature(T_profile=T_re)

            x_c, ok_c, hist_c = _hybrid_newton(
                problem, x, opts, label="Stage C: energia RE-ON")
            report["stage_C"] = {"ok": ok_c, "steps": len(hist_c)}
            x = x_c
            if ok_c and width_check is not None:
                width_check(x)
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
        problem, x, opts, label="Refine Stage: energia ON")
    report["refine_stage_steady"] = {"ok": ok_ss, "steps": len(hist_ss)}

    if not ok_ss:
        return x_ss, False, report

    if width_check is not None:
        width_check(x_ss)

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
    if abs(m_ref) < 1e-300:
        return False, metrics

    m_left = float((T[1] - T[0]) / (z[1] - z[0]) / m_ref)
    m_right = float((T[-3] - T[-1]) / (z[-3] - z[-1]) / m_ref)
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
    z0 = float(z_old[0])
    z_new = z0 + (z_old - z0) * factor

    x_new = interpolate_state(x, z_old, z_new, problem.n_species)
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

                use_initial_guess = not restart_mode
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

            # Mirror Cantera auto path: each pass starts from a fresh
            # default-profile initial guess when not using restart data.
            x = None

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

            if not restart_mode:
                x = None

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

