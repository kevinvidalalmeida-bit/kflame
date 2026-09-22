"""Diagnose H2/GRI30 startup; timings include diagnostic instrumentation.

Run native and Cantera separately, without concurrent heavy jobs. No Cantera
profiles or properties are supplied to the native solver.
"""
import os
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict
import json
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
from kflame.benchmarks.soret import benchmark_options, native_solve, compact_result, json_safe
from kflame.flame.config import FlameCase


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend', choices=['native', 'cantera'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    case = FlameCase(fuel='H2', P=1013250., transport_model='multicomponent',
                     soret_enabled=True, ratio=2.5, slope=.04, curve=.08, prune=.003)
    out = dict(case=asdict(case), backend=args.backend, diagnostic_only=True)
    if args.backend == 'cantera':
        import cantera as ct
        gas = ct.Solution(case.mech)
        gas.TP = case.T_in, case.P
        gas.set_equivalence_ratio(case.phi, case.fuel, case.oxidizer)
        f = ct.FreeFlame(gas, width=case.width)
        f.transport_model = case.transport_model
        f.soret_enabled = True
        f.flux_gradient_basis = case.flux_gradient_basis
        f.set_refine_criteria(ratio=case.ratio, slope=case.slope, curve=case.curve, prune=case.prune)
        f.set_max_grid_points(f.flame, 1600)
        out['version'] = ct.__version__
        out['default_tolerances'] = dict(steady_rel=f.flame.steady_reltol(),
            steady_abs=f.flame.steady_abstol(), transient_rel=f.flame.transient_reltol(),
            transient_abs=f.flame.transient_abstol())
        out['max_time_step_count'] = f.max_time_step_count
        events = []
        def event(kind, dt):
            events.append(dict(kind=kind, dt=dt, n_points=len(f.grid), energy=f.energy_enabled,
                transport=f.transport_model, soret=f.soret_enabled, width=float(f.grid[-1]),
                Su=float(f.velocity[0])))
            return 0.
        f.set_time_step_callback(lambda dt: event('transient_success', dt))
        f.set_steady_callback(lambda dt: event('steady_success', dt))
        start = time.perf_counter()
        with args.output.with_suffix('.log').open('w', encoding='utf-8') as stream, redirect_stdout(stream):
            try:
                f.solve(loglevel=1, auto=True)
                out['solved'] = True
            except Exception as exc:
                out.update(solved=False, error=str(exc))
            f.show_stats()
        out.update(wall_s=time.perf_counter()-start, events=events, Su=float(f.velocity[0]),
                   n_points=len(f.grid), width=float(f.grid[-1]))
        out['stats'] = {key: list(getattr(f, key)) for key in ('eval_count_stats', 'eval_time_stats',
            'jacobian_count_stats', 'jacobian_time_stats', 'grid_size_stats', 'time_step_stats')}
        np.savez_compressed(args.output.with_suffix('.npz'), z=f.grid, T=f.T, u=f.velocity, Y=f.Y)
    else:
        import kflame.flame.solver as solver
        original = solver._hybrid_newton
        phases = []
        def hybrid(problem, *a, **kw):
            profile = getattr(problem, '_profile', {})
            before = {k:dict(v) for k,v in profile.items()}
            entry = dict(label=kw.get('label'), n_points=problem.n_points, width=problem.width,
                         energy=problem.solve_energy, transport=problem.transport_model)
            phases.append(entry)
            started = time.perf_counter()
            try:
                result = original(problem, *a, **kw)
                entry['ok'] = result[1]
                entry['history'] = result[2]
                return result
            except Exception as exc:
                entry['interrupted'] = type(exc).__name__
                raise
            finally:
                entry['wall_s'] = time.perf_counter()-started
                entry['profile_delta'] = {k:{metric:value-before.get(k,{}).get(metric,0)
                    for metric,value in v.items()} for k,v in getattr(problem,'_profile',{}).items()}
                print(entry['label'], entry['n_points'], round(entry['wall_s'],3), entry.get('ok'),
                      entry.get('interrupted'), flush=True)
        opts = benchmark_options(True)
        out['options'] = asdict(opts)
        with patch.object(solver, '_hybrid_newton', hybrid):
            result = native_solve(case, opts, bootstrap=True, bootstrap_mesh_factor=2.)
        out.update(result=compact_result(result), phases=phases)
        np.savez_compressed(args.output.with_suffix('.npz'), **{k:result[k] for k in ('z','T','u','Y')})
    args.output.write_text(json.dumps(json_safe(out), indent=2), encoding='utf-8')
    print(args.output, flush=True)


if __name__ == '__main__':
    main()
