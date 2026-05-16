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
    u_left_guess: float = 0.30

    # Newton estacionario
    steady_max_iter: int = 20
    max_jac_age: int = 5
    max_damp_iter: int = 7
    tol: float = 1.0          # norma ponderada del paso (igual que Cantera)
    jac_eps: float = 1e-8
    alpha_min: float = 1e-10

    # Time-stepping (híbrido)
    max_cycles: int = 40               # ciclos steady -> transient
    time_step: float = 1e-6
    time_step_grow: float = 2.0
    time_step_shrink: float = 0.5
    min_time_step: float = 1e-14
    max_time_step: float = 1e-2
    transient_steps_per_cycle: int = 10 # pasos transitorios por ciclo
    transient_max_iter: int = 10       # iters Newton por paso transitorio
    max_transient_failures: int = 20
    max_no_progress_cycles: int = 6

    # Refinamiento (defaults de Refiner::setCriteria)
    refine_grid: bool = True
    max_refine_passes: int = 6
    refine_ratio: float = 10.0
    refine_slope: float = 0.8
    refine_curve: float = 0.8
    refine_prune: float = -0.1
    refine_grid_min: float = 1e-10
    refine_max_points: int = 500

    max_total_time_s: float = 300.0
    verbose: bool = True


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
    x = np.asarray(x0, dtype=float)
    history: list[dict] = []
    steady_fun = _make_steady_fun(problem)
    jac: JacobianState | None = None
    dt = opts.time_step
    t0 = time.perf_counter()
    no_progress_cycles = 0

    if opts.verbose:
        print(f"\n{'─'*60}\n{label or 'Newton híbrido'}\n{'─'*60}")

    for cycle in range(opts.max_cycles):
        # ---- Intentar Newton estacionario ----
        x_ss, ok_ss, hist_ss, jac = newton_solve(
            steady_fun, x, problem,
            rdt=0.0, x_old=None,
            max_iter=opts.steady_max_iter,
            max_jac_age=opts.max_jac_age,
            max_damp_iter=opts.max_damp_iter,
            tol=opts.tol,
            jac_eps=opts.jac_eps,
            alpha_min=opts.alpha_min,
            verbose=opts.verbose,
            jac_state=jac,
        )
        history.append({"cycle": cycle, "phase": "steady",
                        "ok": ok_ss, "steps": len(hist_ss)})
        if ok_ss:
            if opts.verbose:
                Finf = float(np.linalg.norm(
                    residual(x_ss, problem), ord=np.inf))
                print(f"  [ciclo {cycle}] Estacionario OK  ||F||∞={Finf:.4e}")
            return x_ss, True, history
        if opts.verbose:
            print(f"  [ciclo {cycle}] Estacionario FALLÓ → time-stepping")

        # ---- Time-stepping ----
        x_old = x.copy()
        x_older: np.ndarray | None = None
        n_done = 0
        n_fail = 0

        for _ in range(opts.transient_steps_per_cycle):
            dt_try = dt
            rdt = 1.0 / dt_try
            use_bdf2 = x_older is not None
            transient_order = 2 if use_bdf2 else 1
            scheme = "BDF2" if use_bdf2 else "BE"

            x_ts, ok_ts, hist_ts, jac = newton_solve(
                steady_fun, x_old, problem,
                rdt=rdt, x_old=x_old, x_older=x_older,
                transient_order=transient_order,
                max_iter=opts.transient_max_iter,
                max_jac_age=opts.max_jac_age,
                max_damp_iter=opts.max_damp_iter,
                tol=opts.tol,
                jac_eps=opts.jac_eps,
                alpha_min=opts.alpha_min,
                verbose=False,
                jac_state=jac,
            )
            history.append({"cycle": cycle, "phase": "transient",
                            "ok": ok_ts, "dt": dt_try, "scheme": scheme,
                            "steps": len(hist_ts)})

            if ok_ts:
                x_prev = x_old
                x = x_ts
                x_old = x_ts
                x_older = x_prev
                n_done += 1

                dt = min(opts.max_time_step, dt_try * opts.time_step_grow)

                if opts.verbose:
                    Finf = float(np.linalg.norm(
                        residual(x, problem), ord=np.inf))
                    print(
                        f"    ts {scheme} OK  dt={dt:.2e}  ||F||inf={Finf:.4e}"
                    )
            else:
                n_fail += 1
                dt = max(opts.min_time_step, dt_try * opts.time_step_shrink)
                if opts.verbose:
                    print(f"    ts {scheme} FAIL  dt->{dt:.2e} (fallo {n_fail})")
                if n_fail >= opts.max_transient_failures:
                    break
                if dt < opts.min_time_step:
                    break

        if n_done == 0:
            no_progress_cycles += 1
            dt = max(opts.min_time_step, dt * opts.time_step_shrink)
            if opts.verbose:
                print(
                    f"  [ciclo {cycle}] Sin avance en time-stepping "
                    f"(#{no_progress_cycles}); dt={dt:.2e}"
                )
            if no_progress_cycles >= opts.max_no_progress_cycles:
                if opts.verbose:
                    print("  Se alcanzo max_no_progress_cycles. Abortando.")
                break
        else:
            no_progress_cycles = 0

    Finf = float(np.linalg.norm(residual(x, problem), ord=np.inf))
    if opts.verbose:
        print(f"  Híbrido terminado sin convergencia  ||F||∞={Finf:.4e}")
    return x, False, history


# ---------------------------------------------------------------------------
#  Refinamiento (Sim1D::refine)
# ---------------------------------------------------------------------------
def _refine_and_solve(problem, x_ss: np.ndarray, opts: SolveOptions,
                      deadline: float | None) -> tuple[np.ndarray, bool, list[dict]]:
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
        else:
            # Restaurar último steady (como Sim1D hace con m_xlast_ss)
            if opts.verbose:
                print(f"  Refine pass {pass_idx}: solve falló → restaurando último steady")
            problem.z = z_last_ss
            problem.n_points = n_last_ss
            problem.width = float(z_last_ss[-1] - z_last_ss[0])
            problem.backend = SpeciesBackend(problem)
            _, T_old, _ = unpack_state(x_last_ss, n_last_ss, problem.n_species)
            problem.setup_fixed_temperature(T_profile=T_old)
            info["restored"] = True
            return x_last_ss, True, log

    return x_ss, True, log


# ---------------------------------------------------------------------------
#  Interfaz principal (Sim1D::solve con auto=True)
# ---------------------------------------------------------------------------
def solve_free_flame(
    problem,
    x0: np.ndarray | None = None,
    options: SolveOptions | None = None,
) -> tuple[np.ndarray, bool, dict[str, Any]]:
    """
    Replica Sim1D::solve(loglevel, refine_grid=True) con la estrategia auto
    de FreeFlame:
      1. Intentar con energía ON.
      2. Si falla: congelar T, resolver OFF; luego reactivar energía.
      3. Si convergió: refinar malla y resolver de nuevo (iterado).

    Retorna (x, converged, report).
    """
    opts = options or SolveOptions()
    report: dict[str, Any] = {}
    t_start = time.perf_counter()
    deadline = (t_start + opts.max_total_time_s
                if np.isfinite(opts.max_total_time_s) else None)

    if problem.backend is None:
        problem.backend = SpeciesBackend(problem)

    x = (problem.make_initial_guess(u_left_guess=opts.u_left_guess)
         if x0 is None else np.asarray(x0, dtype=float).copy())
    x = problem.reset_bad_values(x)

    _, T0, _ = unpack_state(x, problem.n_points, problem.n_species)
    problem.setup_fixed_temperature(T_profile=T0)

    solved = False

    # ---- Stage A: energía ON ----
    problem.solve_energy = True
    x_a, ok_a, hist_a = _hybrid_newton(
        problem, x, opts, label="Stage A: energía ON")
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
            problem, x, opts, label="Stage B: energía OFF")
        report["stage_B"] = {"ok": ok_b, "steps": len(hist_b)}

        if ok_b:
            x = x_b
            # ---- Stage C: reactivar energía ----
            problem.solve_energy = True
            _, T_re, _ = unpack_state(x, problem.n_points, problem.n_species)
            problem.setup_fixed_temperature(T_profile=T_re)

            x_c, ok_c, hist_c = _hybrid_newton(
                problem, x, opts, label="Stage C: energía RE-ON")
            report["stage_C"] = {"ok": ok_c, "steps": len(hist_c)}
            x = x_c
            solved = ok_c
        else:
            solved = False

    # ---- Refinamiento iterativo (Sim1D::solve while loop) ----
    if solved and opts.refine_grid:
        if deadline and time.perf_counter() > deadline:
            report["refine"] = "timeout_before_refine"
        else:
            x, ok_ref, ref_log = _refine_and_solve(problem, x, opts, deadline)
            report["refine"] = ref_log
            solved = ok_ref

    report["n_points_final"] = int(problem.n_points)
    report["solved"] = bool(solved)
    report["total_time_s"] = float(time.perf_counter() - t_start)
    report["Finf_final"] = float(np.linalg.norm(
        residual(x, problem), ord=np.inf))

    if opts.verbose:
        print(f"\n{'═'*60}")
        print(f"  Resuelto: {solved}  n_pts={problem.n_points}"
              f"  ||F||∞={report['Finf_final']:.4e}"
              f"  t={report['total_time_s']:.1f}s")
        print(f"{'═'*60}")

    return x, bool(solved), report

