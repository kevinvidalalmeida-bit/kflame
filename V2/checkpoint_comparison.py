"""
Checkpoint comparison between Cantera and the Python V2 free-flame solver.

By default this script runs only V2, matching the previous quick-check behavior.
Use --with-cantera when a full reference comparison is needed.
"""

from __future__ import annotations

import argparse
import time

import cantera as ct
import numpy as np

from config import FlameCase
from problem import FreeFlameProblem
from solver import SolveOptions, solve_free_flame
from state import unpack_state

try:
    from materiales.species_backend_native import NativeSpeciesBackend as BackendToUse

    backend_name = "native-vectorized"
except ImportError:
    from species_backend import SpeciesBackend as BackendToUse

    backend_name = "cantera-wrapper"


def run_cantera(case: FlameCase):
    print("\n--- Cantera reference ---")
    gas = ct.Solution(case.mech)
    gas.TP = case.T_in, case.P
    gas.set_equivalence_ratio(case.phi, case.fuel, case.oxidizer)

    flame = ct.FreeFlame(gas, width=case.width)
    flame.transport_model = case.transport_model
    flame.set_refine_criteria(
        ratio=case.ratio, slope=case.slope, curve=case.curve, prune=case.prune
    )

    t0 = time.perf_counter()
    flame.solve(loglevel=0, auto=True)
    elapsed = time.perf_counter() - t0

    z = np.asarray(flame.grid, dtype=float)
    T = np.asarray(flame.T, dtype=float)
    u = np.asarray(flame.velocity, dtype=float)
    Y = np.asarray(flame.Y, dtype=float)
    Su = float(u[0])

    print(f"  time          : {elapsed:.3f} s")
    print(f"  grid points   : {len(z)}")
    print(f"  flame speed   : {Su * 100:.2f} cm/s")
    return z, T, u, Y, elapsed, Su


def run_v2(case: FlameCase):
    print(f"\n--- V2 Python solver ({backend_name}) ---")
    problem = FreeFlameProblem(case, n_points=8)
    problem.backend = BackendToUse(problem)

    opts = SolveOptions(
        verbose=True,
        refine_ratio=case.ratio,
        refine_slope=case.slope,
        refine_curve=case.curve,
        refine_prune=case.prune,
        max_total_time_s=300.0,
    )

    t0 = time.perf_counter()
    x_sol, ok, _ = solve_free_flame(problem, options=opts)
    elapsed = time.perf_counter() - t0

    z = np.asarray(problem.z, dtype=float)
    u, T, Y = unpack_state(x_sol, problem.n_points, problem.n_species)
    Su = float(u[0])

    if not ok:
        print("  WARNING: V2 solver did not converge.")

    print(f"  time          : {elapsed:.3f} s")
    print(f"  grid points   : {len(z)}")
    print(f"  flame speed   : {Su * 100:.2f} cm/s")
    return z, T, u, Y, elapsed, Su


def build_case() -> FlameCase:
    return FlameCase(
        mech="gri30.yaml",
        fuel="CH4",
        oxidizer="O2:1.0, N2:3.76",
        phi=1.0,
        T_in=300.0,
        P=101325.0,
        width=0.03,
        transport_model="mixture-averaged",
        ratio=5.0,
        slope=0.2,
        curve=0.2,
        prune=0.02,
    )


def print_summary(v2_result, cantera_result=None) -> None:
    z_v2, T_v2, u_v2, _, t_v2, Su_v2 = v2_result

    print("\n" + "=" * 60)
    print(" CHECKPOINT SUMMARY")
    print("=" * 60)
    print(f" V2 flame speed       : {Su_v2 * 100:6.2f} cm/s")
    print(f" V2 time              : {t_v2:6.3f} s")

    if cantera_result is None:
        print(" Cantera reference    : skipped (use --with-cantera)")
        return

    z_ct, T_ct, u_ct, _, t_ct, Su_ct = cantera_result
    T_v2_interp = np.interp(z_ct, z_v2, T_v2)
    u_v2_interp = np.interp(z_ct, z_v2, u_v2)

    err_Su = abs(Su_v2 - Su_ct) / max(abs(Su_ct), 1e-30) * 100.0
    err_T_max = float(np.max(np.abs(T_ct - T_v2_interp)))
    err_u_max = float(np.max(np.abs(u_ct - u_v2_interp)))
    speedup = t_ct / t_v2 if t_v2 > 0 else float("inf")

    print(f" Cantera flame speed  : {Su_ct * 100:6.2f} cm/s")
    print(f" Relative Su error    : {err_Su:6.3f} %")
    print(f" Cantera time         : {t_ct:6.3f} s")
    print(f" V2 speedup           : {speedup:6.2f}x")
    print(f" Max |dT|             : {err_T_max:6.2f} K")
    print(f" Max |du|             : {err_u_max:6.4f} m/s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-cantera",
        action="store_true",
        help="Run the Cantera reference and print profile/flame-speed errors.",
    )
    args = parser.parse_args()

    case = build_case()
    print("=" * 60)
    print(" CHECKPOINT: CANTERA vs V2 PYTHON")
    print("=" * 60)

    cantera_result = run_cantera(case) if args.with_cantera else None
    v2_result = run_v2(case)
    print_summary(v2_result, cantera_result=cantera_result)


if __name__ == "__main__":
    main()
