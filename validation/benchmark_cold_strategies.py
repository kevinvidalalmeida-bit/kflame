"""Evidence for independent cold-start candidates; production stays untouched."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
import platform
import shutil
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'V2'), str(ROOT/'V2/materiales')]
import cantera as ct
import numpy as np
import equations
import solver
import species_backend_native as backend_module
from cold_strategy_kernels import build_candidates
from config import FlameCase
from run_saved_comparison import _resolve_mechanism
from benchmark_soret_native import native_solve, json_safe


@contextmanager
def variant_context(name, kernels, records=None):
    records = [] if records is None else records
    original = equations._precompute_block_tridiag_center_thermo
    if name in ('limited-spatial', 'limited-spatial-adapt'):
        from cold_spatial import spatial_context
        with spatial_context(records, adapt=name.endswith('-adapt')):
            yield
    elif name == 'mkl-blocks':
        from cold_blas import mkl_context
        with mkl_context(records):
            yield
    elif name in ('nonlinear-blocks', 'nonlinear-blocks-global'):
        from cold_nonlinear_blocks import nonlinear_blocks_context
        with nonlinear_blocks_context(records, global_descent=name.endswith('-global')):
            yield
    elif name == 'exact-residual':
        from cold_exact_residual import exact_residual_context
        with exact_residual_context(records):
            yield
    elif name == 'generated-chemistry':
        from cold_generated_chemistry import generated_chemistry_context
        with generated_chemistry_context(records):
            yield
    elif name == 'compiled-lu':
        from cold_compiled_lu import compiled_lu_context
        with compiled_lu_context(records):
            yield
    elif name == 'dependencies':
        from cold_dependencies import dependency_context
        with dependency_context(kernels['thermal'], records):
            yield
    elif name in ('transport-action', 'transport-action-shared'):
        from cold_transport_candidates import action_context
        with action_context(records, shared=name.endswith('-shared')):
            yield
    elif name == 'transport-homotopy':
        from cold_transport_candidates import homotopy_context
        with homotopy_context(records):
            yield
    elif name == 'streamed':
        from cold_streamed_jacobian import build_streamed_jacobian
        candidate = build_streamed_jacobian()
        with patch.object(equations, '_block_tridiag_jacobian_local', candidate):
            yield
    elif name == 'mesh-budget':
        from cold_mesh_budget import mesh_budget_context
        with mesh_budget_context(records):
            yield
    elif name == 'two-grid':
        from cold_two_grid import two_grid_context
        with two_grid_context(records):
            yield
    elif name == 'thermal':
        def thermal_fd(*args, **kwargs):
            with patch.object(backend_module, '_eval_thermo_kinetics_sparse_numba_core', kernels[name]):
                try:
                    value = original(*args, **kwargs)
                except Exception as exc:
                    records.append(dict(kind='thermal_fd', exception=repr(exc)))
                    raise
                records.append(dict(kind='thermal_fd', nodes=int(args[2]['n_pts']), returned=value is not None))
                return value
        with patch.object(equations, '_precompute_block_tridiag_center_thermo', thermal_fd):
            yield
    elif name == 'blocked':
        def blocked(x, problem, cache, *args):
            n, k, nv = (int(cache[key]) for key in ('n_pts', 'n_sp', 'nv'))
            # Bound temporary T/Y perturbation storage, not physical parameters.
            size = max(1, (2*1024*1024)//(8*nv*(k+1)))
            out = (np.empty((n, nv)), np.empty((n, nv)),
                   np.empty((k, n, nv)), np.empty((k, n, nv)))
            for first in range(0, n, size):
                last = min(n, first+size)
                local = dict(cache, n_pts=last-first)
                result = original(x[first*nv:last*nv], problem, local, *args)
                if result is None:
                    return None
                out[0][first:last], out[1][first:last] = result[:2]
                out[2][:, first:last], out[3][:, first:last] = result[2:]
            return out
        with patch.object(equations, '_precompute_block_tridiag_center_thermo', blocked):
            yield
    else:
        with patch.object(backend_module, '_eval_thermo_kinetics_sparse_numba_core', kernels[name]):
            yield


def kernel_checks(kernels):
    results = []
    for mechanism, fuel in [('h2o2.yaml', 'H2'), ('gri30.yaml', 'CH4')]:
        for pressure in (1, 10):
            case = SimpleNamespace(mech=_resolve_mechanism(mechanism),
                                   transport_model='mixture-averaged', soret_enabled=False)
            backend = backend_module.NativeSpeciesBackend(SimpleNamespace(case=case, P=pressure*ct.one_atm))
            gas = ct.Solution(case.mech)
            gas.TP = 300, pressure*ct.one_atm
            gas.set_equivalence_ratio(1, fuel, 'O2:1,N2:3.76')
            fresh = gas.Y.copy()
            gas.equilibrate('HP')
            burned = gas.Y.copy()
            ys, ts = [], []
            for t in (300., 700., 1000., 1000.0001, 1500., 2400.):
                for progress in (0., .3, 1.):
                    y = (1-progress)*fresh + progress*burned
                    for negative in (False, True):
                        yp = y.copy()
                        if negative:
                            yp[gas.species_index('H')] = -1e-8
                        ys.append(yp)
                        ts.append(t)
            y = np.array(ys).T.copy()
            baseline = None
            for name, kernel in kernels.items():
                omega, enthalpy = np.empty_like(y), np.empty_like(y)
                start = time.perf_counter()
                with patch.object(backend_module, '_eval_thermo_kinetics_sparse_numba_core', kernel):
                    rho, cp = backend.eval_grid_thermo_kinetics_into(np.array(ts), y, omega, enthalpy)
                fields = dict(rho=rho, cp=cp, omega=omega, h=enthalpy)
                if baseline is None:
                    baseline = fields
                differences = {}
                for field, value in fields.items():
                    np.testing.assert_allclose(value, baseline[field], rtol=5e-12, atol=1e-10,
                                               err_msg=f'{mechanism}/{pressure}/{name}/{field}')
                    differences[field] = float(np.max(abs(value-baseline[field])))
                results.append(dict(mechanism=mechanism, pressure_atm=pressure, variant=name,
                                    compilation_and_check_s=time.perf_counter()-start,
                                    maximum_absolute_difference=differences))
                print('KERNEL', json.dumps(results[-1]), flush=True)
    return results


def strict_final_gate(run):
    """Check raw final metrics independently of experimental acceptance hooks."""
    final = run['stages'][-1]
    return bool(run['accepted'] and final.get('grid_converged')
                and np.isfinite(run['Finf']) and run['Finf'] <= 1e4
                and np.isfinite(final.get('weighted_step_norm_final', np.nan))
                and final['weighted_step_norm_final'] <= 1.
                and np.isfinite(run['species_sum_error']) and np.isfinite(run['mass_flow_error'])
                and np.isfinite(run['Y']).all() and np.min(run['Y']) >= -1e-7
                and np.isfinite(run['T']).all() and np.min(run['T']) >= 200.)


def strategy_gate(name, trace):
    if name.startswith('limited-spatial'):
        return bool(len(trace)==1 and trace[0].get('assembly_calls',0)>0
                    and not trace[0].get('assembly_errors'))
    return True


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--max-seconds', type=float, default=180.)
    parser.add_argument('--cases', nargs='+', choices=['ch4_1', 'ch4_10', 'h2_1', 'h2_10'],
                        default=['ch4_1', 'h2_1'])
    parser.add_argument('--variants', nargs='+', choices=['baseline', 'thermal', 'blocked', 'powers', 'serial',
                                                        'granularity', 'mesh-budget', 'two-grid', 'streamed', 'dependencies',
                                                        'transport-action', 'transport-action-shared', 'transport-homotopy', 'compiled-lu',
                                                        'exact-residual', 'generated-chemistry', 'nonlinear-blocks', 'nonlinear-blocks-global',
                                                        'mkl-blocks', 'simd-nasa', 'limited-spatial', 'limited-spatial-adapt'],
                        default=['baseline'])
    parser.add_argument('--reproduce-archived', action='store_true',
                        help='Explicit historical reproduction only; no candidate is approved for production.')
    parser.add_argument('--granularity-report', type=Path)
    parser.add_argument('--checks-only', action='store_true')
    args = parser.parse_args(argv)
    if args.variants[0] != 'baseline' or args.repeat < 1 or not np.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error('Require baseline first and a positive repetition count.')
    if any(name not in ('baseline', 'exact-residual', 'nonlinear-blocks', 'nonlinear-blocks-global', 'mkl-blocks', 'simd-nasa', 'limited-spatial', 'limited-spatial-adapt') for name in args.variants) and not args.reproduce_archived:
        parser.error('Candidates are archived. Historical reproduction requires --reproduce-archived.')
    if any(name.startswith('transport-') for name in args.variants) and any(
            not case.startswith('h2') for case in args.cases):
        parser.error('Transport candidates require the multicomponent H2 cases; do not time a no-op on CH4.')
    if 'granularity' in args.variants and args.granularity_report is None:
        parser.error('Granularity requires a measured --granularity-report.')
    return args


def selected_kernels(variants):
    if all(name in ('baseline', 'mkl-blocks', 'simd-nasa', 'limited-spatial', 'limited-spatial-adapt') for name in variants):
        kernels = dict(baseline=backend_module._eval_thermo_kinetics_sparse_numba_core)
        if 'simd-nasa' in variants:
            from cold_simd import build_simd
            kernels['simd-nasa'] = build_simd()
        return kernels
    if all(name in ('baseline', 'exact-residual', 'generated-chemistry', 'nonlinear-blocks', 'nonlinear-blocks-global') for name in variants):
        # Normal checks must neither build nor compile archived kernels.
        return dict(baseline=backend_module._eval_thermo_kinetics_sparse_numba_core)
    return build_candidates()


def main():
    args = parse_arguments()
    destination = ROOT/'resultados/cold_strategies'/datetime.now().strftime('%Y%m%d_%H%M%S')
    destination.mkdir(parents=True, exist_ok=False)
    print('OUTPUT', destination, flush=True)
    kernels = selected_kernels(args.variants)
    if 'granularity' in args.variants:
        calibration = json.loads(args.granularity_report.read_text(encoding='utf-8'))
        if calibration['source_sha256'] != kernels['serial'].audit_sha256:
            raise RuntimeError('Granularity calibration belongs to a different kernel.')
        cutoff = int(calibration['selected_serial_cutoff'])
        kernels['granularity'] = lambda *a: kernels['serial' if a[0].size <= cutoff else 'baseline'](*a)
    result = dict(command=sys.argv, policy='Cold flame, no saved seeds; separate full-stack warmup per variant.',
                  full_stack_warmup=True, reference_kinetics='current production at campaign start',
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), Path(__file__).with_name('cold_strategy_kernels.py'),
                                           ROOT/'V2/equations.py', ROOT/'V2/solver.py',
                                           ROOT/'V2/materiales/species_backend_native.py',
                                           ROOT/'V2/materiales/kinetics_native.py',
                                           ROOT/'validation/cold_mesh_budget.py', ROOT/'validation/cold_two_grid.py']},
                  kernel_sha256={name: getattr(kernel, 'audit_sha256', 'production') for name, kernel in kernels.items()},
                  versions=dict(python=sys.version, cantera=ct.__version__, numpy=np.__version__),
                  platform=platform.platform(), cpu=platform.processor(), logical_cpus=os.cpu_count(),
                  threads={k: os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS']},
                  warmup_runs=[], runs=[])
    if 'granularity' in args.variants:
        result['granularity_calibration'] = calibration
    # Preserve exact sources even when the shared worktree has uncommitted edits.
    paths = list((ROOT/'V2').rglob('*.py')) + list((ROOT/'validation').glob('cold_*.py'))
    paths += [Path(__file__), ROOT/'validation/analyze_cold_strategies.py',
              ROOT/'V2/materiales/collision_integrals_mm.json']
    for source_path in paths:
        relative = source_path.relative_to(ROOT)
        snapshot = destination/'sources'/relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, snapshot)
        result['source_sha256'][str(relative)] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    def save():
        (destination/'summary.json').write_text(json.dumps(json_safe(result), indent=2, allow_nan=False), encoding='utf-8')
    result['kernel_checks'] = kernel_checks(kernels)
    if 'simd-nasa' in kernels:
        from cold_simd import vector_evidence
        result['simd_evidence'] = vector_evidence(kernels['simd-nasa'])
        (destination/'simd_source.py').write_text(kernels['simd-nasa'].audit_source, encoding='utf-8')
    save()
    if args.checks_only:
        return
    for label in args.cases:
        multi = label.startswith('h2')
        pressure = int(label.split('_')[1])
        slope, curve = (.01, .02) if pressure == 10 else (.04, .08)
        case = FlameCase(mech=_resolve_mechanism('gri30.yaml'), fuel='H2' if multi else 'CH4',
                         oxidizer='O2:1,N2:3.76', phi=1., T_in=300., P=pressure*ct.one_atm,
                         width=.03, transport_model='multicomponent' if multi else 'mixture-averaged',
                         soret_enabled=multi, ratio=2.5, slope=slope, curve=curve, prune=.003)
        opts = solver.SolveOptions(verbose=False, profile=True, max_total_time_s=args.max_seconds,
                                   max_refine_passes=12, require_grid_convergence=True, refine_max_points=1600,
                                   refine_ratio=2.5, refine_slope=slope, refine_curve=curve, refine_prune=.003,
                                   acceptance_criterion='cantera', residual_guard_inf=1e4,
                                   final_Finf_limit=1e4, refine_Finf_limit=1e4,
                                   jacobian_mode='block_tridiag', lag_multicomponent_transport=multi,
                                   multicomponent_bootstrap=multi, bootstrap_mesh_factor=2.)
        valid = []
        for name in args.variants:
            trace = []
            with variant_context(name, kernels, trace):
                run = native_solve(case, opts, bootstrap=multi, bootstrap_mesh_factor=2.)
            run['internal_accepted'] = run['accepted']
            run['accepted'] = strict_final_gate(run) and strategy_gate(name,trace)
            warm_profile = destination/f'{label}_{name}_warmup.npz'
            np.savez_compressed(warm_profile, **{key:run[key] for key in ('z','T','Y','u')})
            result['warmup_runs'].append(dict(case=label, variant=name, accepted=run['accepted'],
                                              profile_sha256=hashlib.sha256(warm_profile.read_bytes()).hexdigest(),
                                              internal_accepted=run['internal_accepted'],
                                              time_s=run['time_s'], strategy_trace=trace,
                                              Finf=run['Finf'], species_sum_error=run['species_sum_error'],
                                              mass_flow_error=run['mass_flow_error'],
                                              final_weighted_norm=run['stages'][-1].get('weighted_step_norm_final')))
            print('WARMUP', json.dumps({k: v for k, v in result['warmup_runs'][-1].items() if k != 'strategy_trace'}), flush=True)
            if run['accepted']:
                valid.append(name)
            save()
            if name == 'baseline' and not run['accepted']:
                raise RuntimeError('Baseline failed; no candidate comparison is valid.')
        reference = None
        for repetition in range(args.repeat):
            for name in (valid if repetition % 2 == 0 else valid[::-1]):
                trace = []
                with variant_context(name, kernels, trace):
                    run = native_solve(case, opts, bootstrap=multi, bootstrap_mesh_factor=2.)
                run['internal_accepted'] = run['accepted']
                run['accepted'] = strict_final_gate(run) and strategy_gate(name,trace)
                fields = {key: run.pop(key) for key in ('z', 'T', 'Y', 'u')}
                if reference is None:
                    reference = fields
                profile = destination/f'{label}_{name}_{repetition}.npz'
                np.savez_compressed(profile, **fields)
                same = np.array_equal(fields['z'], reference['z'])
                run.update(case=label, variant=name, repetition=repetition, configuration=asdict(case), options=asdict(opts),
                           mechanism_sha256=hashlib.sha256(Path(case.mech).read_bytes()).hexdigest(),
                           strategy_trace=trace,
                           same_grid_as_baseline=same, profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
                           differences_on_identical_grid={key: float(np.max(abs(fields[key]-reference[key])))
                                                          for key in fields} if same else None)
                result['runs'].append(run)
                save()
                print('RUN', json.dumps({k: run[k] for k in ['case', 'variant', 'repetition', 'accepted', 'time_s',
                                                            'Su', 'n_points', 'Finf', 'same_grid_as_baseline']}), flush=True)


if __name__ == '__main__':
    main()
