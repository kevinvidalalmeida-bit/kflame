"""
main_coupled_test.py - FreeFlame acoplado (u + T + especies) con flujo auto
estilo Cantera, pero con limites duros para evitar ejecuciones interminables.
"""
from __future__ import annotations

import time
import numpy as np

from config import FlameCase
from problem import FreeFlameProblem
from residual import residual_coupled
from V2.solver_cantera_auto import AutoSolveOptions, solve_free_flame_cantera_auto
from species_backend import SpeciesBackend
from state import unpack_state


def main():
    case = FlameCase(
        mech="gri30.yaml",
        fuel="CH4",
        oxidizer="O2:1.0, N2:3.76",
        phi=1.0,
        T_in=300.0,
        P=101325.0,
        width=0.03,
        transport_model="mixture-averaged",
        ratio=3.0,
        slope=0.07,
        curve=0.14,
        prune=0.0,
    )

    # Arranque en malla inicial corta (como en Cantera auto)
    problem = FreeFlameProblem(case, n_points=8)
    problem.backend = SpeciesBackend(problem)

    # Limites fuertes para no quedarse "pensando" si no converge
    opts = AutoSolveOptions(
        n_ladder=(12, 24),
        u_left_guess=0.30,
        steady_max_iter=8,
        transient_max_iter=4,
        max_cycles=3,
        tol=1.0e-6,
        jac_eps=1.0e-8,
        alpha_min=1.0e-8,
        time_step=1.0e-6,
        time_step_sequence=(1, 2, 5, 10),
        time_step_grow=1.5,
        time_step_shrink=0.5,
        min_time_step=1.0e-14,
        max_time_step=1.0e-2,
        max_transient_failures=3,
        refine_grid=True,
        max_refine_passes=2,
        refine_ratio=case.ratio,
        refine_slope=case.slope,
        refine_curve=case.curve,
        refine_prune=case.prune,
        refine_grid_min=1.0e-10,
        refine_max_points=200,
        verbose=True,
    )

    print("=" * 68)
    print("  FREEFLAME ACOPLADO - DESDE CERO (SIN REFERENCIA CANTERA)")
    print("=" * 68)
    print(problem.summary())

    t0 = time.perf_counter()
    x_sol, ok, report = solve_free_flame_cantera_auto(problem, options=opts)
    t_total = float(time.perf_counter() - t0)

    u, T, Y = unpack_state(x_sol, problem.n_points, problem.n_species)
    F = residual_coupled(x_sol, problem)

    print("\n" + "=" * 68)
    print("  RESULTADO FINAL")
    print("=" * 68)
    print(f"converged     = {ok}")
    print(f"n_points      = {problem.n_points}")
    print(f"||F||inf      = {np.linalg.norm(F, ord=np.inf):.6e}")
    print(f"Su (~u[0])    = {u[0]:.6f} m/s")
    print(f"mdot_in       = {problem.backend.density(T[0], Y[:, 0]) * u[0]:.6f}")
    print(f"T min/max     = {T.min():.3f} / {T.max():.3f}")
    print(f"sum(Y) left   = {Y[:, 0].sum():.12f}")
    print(f"sum(Y) right  = {Y[:, -1].sum():.12f}")
    print(f"Y_last min    = {Y[-1, :].min():.6e}")
    print(f"t_total [s]   = {t_total:.3f}")
    if "total_time_s" in report:
        print(f"t_solver [s]  = {report['total_time_s']:.3f}")

    print("\n--- Etapas auto ---")
    for st in report.get("stages", []):
        line = (
            f"N={st['N']:3d}  "
            f"on={st.get('energy_on_ok', False)}"
            f"({st.get('energy_on_steps', 0)} pasos, {st.get('energy_on_time_s', 0.0):.2f}s)"
        )
        if "energy_off_ok" in st:
            line += (
                f"  off={st.get('energy_off_ok', False)}"
                f"({st.get('energy_off_steps', 0)} pasos, {st.get('energy_off_time_s', 0.0):.2f}s)"
            )
        if "energy_reon_ok" in st:
            line += (
                f"  re-on={st.get('energy_reon_ok', False)}"
                f"({st.get('energy_reon_steps', 0)} pasos, {st.get('energy_reon_time_s', 0.0):.2f}s)"
            )
        line += f"  stage={st.get('stage_time_s', 0.0):.2f}s"
        print(line)

    if "refine" in report:
        print("\n--- Refinamiento ---")
        for rr in report["refine"]:
            print(
                f"pass={rr['pass']}  changed={rr['changed']}  "
                f"{rr['n_old']}->{rr['n_new']}  +{rr['n_inserted']} -{rr['n_removed']}  "
                f"solve_ok={rr.get('solve_ok', True)}  "
                f"solve_t={rr.get('solve_time_s', 0.0):.2f}s  pass_t={rr.get('pass_time_s', 0.0):.2f}s"
            )
        if "refine_total_time_s" in report:
            print(f"refine_total [s] = {report['refine_total_time_s']:.3f}")

    print("\nRef Cantera (solo orientativo): Su ~ 0.381 m/s")


if __name__ == "__main__":
    main()
