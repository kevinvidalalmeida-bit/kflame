"""Reproducible native Soret benchmark; rejected runs are saved too.

By default, construction, solving and output require no Cantera.
The --no-native-only comparison imports Cantera explicitly.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
import hashlib
import platform
import subprocess

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).parent / "materiales"))

import numpy as np

from config import FlameCase
from problem import FreeFlameProblem
from mechanism_data import resolve_mechanism as _resolve_mechanism, ONE_ATM
from solver import SolveOptions, solve_free_flame
from species_backend_native import NativeSpeciesBackend
from state import unpack_state


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def native_solve(case, options, bootstrap=False, bootstrap_mesh_factor=1.0, initial_points=8):
    def problem_for(c):
        p = FreeFlameProblem(c, n_points=initial_points)
        p.assume_finite_y = True
        p.backend_factory = lambda prob: NativeSpeciesBackend(prob)
        p.backend = p.backend_factory(p)
        return p

    start = time.perf_counter()
    p = problem_for(case)
    setup_time = time.perf_counter() - start
    solve_start = time.perf_counter()
    run_options = replace(options, multicomponent_bootstrap=bootstrap,
                          bootstrap_mesh_factor=bootstrap_mesh_factor)
    x, ok, report = solve_free_flame(p, options=run_options)
    stages = []
    if "transport_bootstrap" in report:
        stages.append(report["transport_bootstrap"])
    stages.append({k: v for k, v in report.items() if k != "transport_bootstrap"})
    elapsed = time.perf_counter() - solve_start
    u, temperature, y = unpack_state(x, p.n_points, p.n_species)
    rho = p.P / (8314.46261815324 * temperature * np.sum(y / p.backend.W[:, None], axis=0))
    mass_flow = rho * u
    return dict(
        time_s=elapsed, setup_time_s=setup_time, end_to_end_s=time.perf_counter() - start,
        accepted=bool(ok and report.get("final_accepted") and report.get("grid_converged")
                      and (not bootstrap or len(stages) == 2)),
        bootstrap=bootstrap, Su=float(u[0]), n_points=p.n_points, width=p.width,
        species_sum_error=float(np.max(np.abs(y.sum(axis=0) - 1))),
        mass_flow_error=float(np.max(np.abs(mass_flow - mass_flow.mean())) / abs(mass_flow.mean())),
        Finf=report.get("Finf_final"), stages=stages,
        z=p.z, T=temperature, Y=y, u=u,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fuel", default="H2")
    parser.add_argument("--mechanism", default="gri30.yaml")
    parser.add_argument("--phi", type=float, default=1.0)
    parser.add_argument("--pressure-atm", type=float, default=1.0)
    parser.add_argument("--transport", choices=("mixture-averaged", "multicomponent"), default="multicomponent")
    parser.add_argument("--no-soret", action="store_true")
    parser.add_argument("--width", type=float, default=0.03)
    parser.add_argument("--slope", type=float, default=0.04)
    parser.add_argument("--curve", type=float, default=0.08)
    parser.add_argument("--max-seconds", type=float, default=180)
    parser.add_argument("--initial-points", type=int, default=8)
    parser.add_argument("--initial-speed", type=float, default=SolveOptions().u_left_guess,
                        help="Native cold-start guess in m/s; does not prescribe the flame speed")
    parser.add_argument("--bootstrap-grids", action="store_true",
                        help="Diagnostic recovery on the existing 12/24/48 grid sequence")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--bootstrap-mesh-factor", type=float, default=1.0)
    parser.add_argument("--lag", action="store_true")
    parser.add_argument("--native-only", action=argparse.BooleanOptionalAction, default=True,
                        help="Solo V2 por defecto; --no-native-only activa la comparación con Cantera.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--trace", action="store_true", help="Persist inner Newton/PTC histories (diagnostic, not timing evidence)")
    parser.add_argument("--output", type=Path, default=Path("resultados/soret_native_validation"))
    args = parser.parse_args()
    cantera_version = None
    if not args.native_only:
        import cantera
        from run_saved_comparison import _cantera_solve
        cantera_version = cantera.__version__
    out = args.output / datetime.now().strftime("benchmark_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT {out.resolve()}", flush=True)
    case = FlameCase(mech=_resolve_mechanism(args.mechanism), fuel=args.fuel, oxidizer="O2:1, N2:3.76",
                     phi=args.phi, T_in=300, P=ONE_ATM * args.pressure_atm, width=args.width,
                     transport_model=args.transport, soret_enabled=not args.no_soret, flux_gradient_basis="molar",
                     ratio=2.5, slope=args.slope, curve=args.curve, prune=0.003)
    options = SolveOptions(verbose=args.verbose, profile=True, max_total_time_s=args.max_seconds,
                           max_refine_passes=12, require_grid_convergence=True, refine_max_points=1600,
                           refine_ratio=case.ratio, refine_slope=case.slope, refine_curve=case.curve,
                           refine_prune=case.prune, acceptance_criterion="cantera",
                           residual_guard_inf=1e4, final_Finf_limit=1e4, refine_Finf_limit=1e4,
                           jacobian_mode="block_tridiag", lag_multicomponent_transport=args.lag,
                           u_left_guess=args.initial_speed, auto_bootstrap_grids=args.bootstrap_grids,
                           trace_solver=args.trace)
    effective_options = replace(options, multicomponent_bootstrap=args.bootstrap,
                                bootstrap_mesh_factor=args.bootstrap_mesh_factor)
    source_root = Path(__file__).resolve().parent
    source_paths = sorted(source_root.glob("*.py")) + sorted((source_root / "materiales").glob("*.py"))
    source_paths.append(source_root / "materiales" / "collision_integrals_mm.json")
    summary = dict(case=asdict(case), options=asdict(effective_options), runs=[],
                   versions=dict(cantera=cantera_version, numpy=np.__version__),
                   command=sys.argv, bootstrap=args.bootstrap, bootstrap_mesh_factor=args.bootstrap_mesh_factor,
                   platform=platform.platform(), cpu=platform.processor(), python=sys.version,
                   git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   git_status=subprocess.check_output(["git", "status", "--short"], text=True),
                   mechanism_sha256=hashlib.sha256(Path(case.mech).read_bytes()).hexdigest(),
                   source_sha256={str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in source_paths},
                   threads={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")},
                   profile_cache="none", compiler_cold=False,
                   jit_policy="Fresh process; disk JIT cache retained, not cleared; first pair retained",
                   first_execution_includes_jit=True)
    for repetition in range(args.repeat):
        order = ["native", "cantera"] if repetition % 2 == 0 else ["cantera", "native"]
        for solver in order:
            if solver == "cantera" and args.native_only:
                continue
            try:
                record = native_solve(case, options, args.bootstrap, args.bootstrap_mesh_factor, args.initial_points) if solver == "native" else _cantera_solve(case)
                if solver == "cantera":
                    record["accepted"] = True
                profiles = {k: record.pop(k) for k in ("z", "T", "Y", "u")}
                np.savez_compressed(out / f"{solver}_{repetition}.npz", **profiles)
                record.update(solver=solver, repetition=repetition)
            except Exception as exc:
                import traceback
                record = dict(solver=solver, repetition=repetition, accepted=False,
                              exception=repr(exc), traceback=traceback.format_exc())
            summary["runs"].append(record)
            (out / "summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
            print(json.dumps(json_safe({k: v for k, v in record.items() if k != "stages"})), flush=True)


if __name__ == "__main__":
    main()
