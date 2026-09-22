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
from types import SimpleNamespace

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")

import numpy as np

from kflame.chemistry.backend import NativeSpeciesBackend
from kflame.chemistry.initialization import fresh_mixture
from kflame.chemistry.mechanism import (
    ONE_ATM,
    R_UNIV,
    _ATOMIC_WEIGHTS,
    load_mechanism,
    resolve_mechanism as _resolve_mechanism,
)
from kflame.flame.config import FlameCase
from kflame.flame.equations import _multicomponent_flux
from kflame.flame.problem import FreeFlameProblem
from kflame.flame.solver import SolveOptions, solve_free_flame
from kflame.flame.state import unpack_state


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


def compact_result(result):
    return {key: value for key, value in result.items() if key not in ('z', 'u', 'T', 'Y')}


def benchmark_options(profile=True):
    """Shared certified settings for standalone native performance probes."""
    return SolveOptions(
        verbose=False, profile=profile, max_total_time_s=180,
        max_refine_passes=12, require_grid_convergence=True,
        refine_max_points=1600, refine_ratio=2.5, refine_slope=.04,
        refine_curve=.08, refine_prune=.003, acceptance_criterion='cantera',
        residual_guard_inf=1e4, final_Finf_limit=1e4, refine_Finf_limit=1e4,
        jacobian_mode='block_tridiag', auto_bootstrap_grids=False,
    )


def elemental_flux_diagnostics(case, fields, solver):
    """Reconstruct mesh-sensitive elemental fluxes for saved profiles."""
    mech = load_mechanism(case['mech'])
    atomic_weights = np.array([_ATOMIC_WEIGHTS[element] for element in mech.element_names])
    element_mass = mech.atom_matrix * atomic_weights[:, None] / mech.molecular_weights[None, :]
    fresh_element = element_mass @ fresh_mixture(mech, case['phi'], case['fuel'], case['oxidizer'])
    z, temperature, y, u = (fields[key] for key in ('z', 'T', 'Y', 'u'))
    weights = mech.molecular_weights
    rho = case['P'] / (R_UNIV * temperature * np.sum(y / weights[:, None], axis=0))
    face_temperature = .5 * (temperature[:-1] + temperature[1:])
    face_y = .5 * (y[:, :-1] + y[:, 1:])
    if solver == 'native':
        native = NativeSpeciesBackend(SimpleNamespace(case=SimpleNamespace(**case), P=case['P']))
        face_rho, _conductivity, face_weight, multi, thermal = native.eval_multicomponent_face_transport(
            face_temperature, face_y
        )
    else:
        import cantera as ct  # Explicit reference diagnostic only.
        gas = ct.Solution(case['mech'], transport_model='multicomponent')
        face_rho, face_weight = np.empty(face_temperature.size), np.empty(face_temperature.size)
        multi = np.empty((weights.size, weights.size, face_temperature.size))
        thermal = np.empty((weights.size, face_temperature.size))
        for index in range(face_temperature.size):
            gas.TPY = face_temperature[index], case['P'], face_y[:, index]
            face_rho[index], face_weight[index] = gas.density, gas.mean_molecular_weight
            multi[:, :, index], thermal[:, index] = gas.multi_diff_coeffs, gas.thermal_diff_coeffs
        if not case.get('soret_enabled', False):
            thermal.fill(0.0)
    flux = _multicomponent_flux(
        y[:, :-1], y[:, 1:], temperature[:-1], temperature[1:], face_rho,
        face_weight, multi, thermal, np.diff(z), weights,
    )
    mass_flow = rho * u
    expected = mass_flow[0] * fresh_element
    total = element_mass @ (face_y * (.5 * (mass_flow[:-1] + mass_flow[1:]))[None, :] + flux)
    discrepancy = np.max(np.abs(total - expected[:, None]), axis=1)
    return dict(
        convention='Midpoint reconstructed convective + diffusive elemental flux; mesh-sensitive diagnostic',
        max_diffusive_mass_sum=float(np.max(np.abs(flux.sum(axis=0)))),
        min_mass_fraction=float(y.min()),
        elements={
            mech.element_names[index]: dict(
                inlet_flux=float(expected[index]),
                max_absolute_deviation=float(discrepancy[index]),
                relative_deviation=(float(discrepancy[index] / abs(expected[index]))
                                    if abs(expected[index]) > 1e-12 else None),
            )
            for index in range(len(mech.element_names))
        },
    )


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
                        help="Solo KFLAME por defecto; --no-native-only activa la comparación con Cantera.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--trace", action="store_true", help="Persist inner Newton/PTC histories (diagnostic, not timing evidence)")
    parser.add_argument("--output", type=Path, default=Path('runs/soret'))
    args = parser.parse_args()
    cantera_version = None
    if not args.native_only:
        import cantera
        from kflame.reference.compare import _cantera_solve
        cantera_version = cantera.__version__
    out = args.output / datetime.now().strftime("benchmark_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT {out.resolve()}", flush=True)
    case = FlameCase(mech=_resolve_mechanism(args.mechanism), fuel=args.fuel, oxidizer="O2:1, N2:3.76",
                     phi=args.phi, T_in=300, P=ONE_ATM * args.pressure_atm, width=args.width,
                     transport_model=args.transport, soret_enabled=not args.no_soret, flux_gradient_basis="molar",
                     ratio=2.5, slope=args.slope, curve=args.curve, prune=0.003)
    options = replace(
        benchmark_options(), verbose=args.verbose, max_total_time_s=args.max_seconds,
        refine_ratio=case.ratio, refine_slope=case.slope, refine_curve=case.curve,
        refine_prune=case.prune, lag_multicomponent_transport=args.lag,
        u_left_guess=args.initial_speed, auto_bootstrap_grids=args.bootstrap_grids,
        trace_solver=args.trace,
    )
    effective_options = replace(options, multicomponent_bootstrap=args.bootstrap,
                                bootstrap_mesh_factor=args.bootstrap_mesh_factor)
    source_root = Path(__file__).resolve().parents[1]
    source_paths = sorted(source_root.rglob('*.py'))
    source_paths.append(source_root / 'chemistry/data/collision_integrals_mm.json')
    # Installed wheels can run outside a Git checkout.
    def git_info(*arguments):
        try:
            result = subprocess.run(['git', *arguments], capture_output=True, text=True, check=False)
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    summary = dict(case=asdict(case), options=asdict(effective_options), runs=[],
                   versions=dict(cantera=cantera_version, numpy=np.__version__),
                   command=sys.argv, bootstrap=args.bootstrap, bootstrap_mesh_factor=args.bootstrap_mesh_factor,
                   platform=platform.platform(), cpu=platform.processor(), python=sys.version,
                   git_revision=git_info('rev-parse', 'HEAD'),
                   git_status=git_info('status', '--short'),
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
