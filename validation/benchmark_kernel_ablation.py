"""Paired cold-flame screening of exact LU dispatch and mass-action products.

Reference arithmetic is reconstructed in memory for this audit only, without
adding switches or legacy kernels to the production solver.
"""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import textwrap
import time
from unittest.mock import patch

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'V2'))
sys.path.insert(0, str(ROOT / 'V2' / 'materiales'))
import cantera as ct
import numpy as np
from numba import njit
import equations
import solver
import species_backend_native as backend_module
from kinetics_native import _negative_mass_action_factor
from benchmark_soret_native import native_solve, json_safe
from benchmark_lu_layout import wrapper_reference
from config import FlameCase
from run_saved_comparison import _resolve_mechanism


def build_reference_kernel():
    current = backend_module._eval_thermo_kinetics_sparse_numba_core
    source = textwrap.dedent(inspect.getsource(current.py_func))
    source = source[source.index('def '):]
    # Reproduce the old order of operations, not merely a helper returning
    # an equivalent product: loop placement can itself affect performance.
    marker = '            delta_g = 0.0\n'
    if source.count(marker) != 1:
        raise RuntimeError('Cannot restore original kinetics loop unambiguously.')
    source = source.replace(marker, '''            rf_exp = 0.0
            for ii in range(r_count[r]):
                k = r_idx[r, ii]
                rf_exp += r_nu[r, ii] * logC[k]

            rr_exp = 0.0
            if is_reversible[r]:
                for ii in range(p_count[r]):
                    k = p_idx[r, ii]
                    rr_exp += p_nu[r, ii] * logC[k]

''' + marker)
    begin = source.index('            Rf = _mass_action_product(')
    end = source.index('            M = c_total', begin)
    source = source[:begin] + '''            Rf = math.exp(rf_exp) * kf
            Rr = 0.0
            if is_reversible[r]:
                Rr = math.exp(rr_exp) * kr

            if has_negative:
                Rf *= _negative_mass_action_factor(concentrations, r_idx[r], r_nu[r], r_count[r])
                Rr *= _negative_mass_action_factor(concentrations, p_idx[r], p_nu[r], p_count[r])

''' + source[end:]
    namespace = dict(vars(backend_module))
    namespace['_negative_mass_action_factor'] = _negative_mass_action_factor
    exec(compile(source, '<log-mass-action-reference>', 'exec'), namespace)
    reference = njit(parallel=True)(namespace[current.py_func.__name__])
    reference._audit_source_sha256 = hashlib.sha256(source.encode('utf-8')).hexdigest()
    return reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--cases', nargs='+', default=['ch4_1', 'h2_1', 'h2_10'])
    parser.add_argument('--warmup-flames', action='store_true',
                        help='Predeclared full-stack warm-up per variant, discarded as seeds and timed separately')
    parser.add_argument('--variants', nargs='+', choices=['baseline', 'lapack', 'products'],
                        default=['baseline', 'lapack', 'products'])
    args = parser.parse_args()
    destination = ROOT / 'resultados' / 'kernel_ablation' / datetime.now().strftime('%Y%m%d_%H%M%S')
    destination.mkdir(parents=True, exist_ok=False)
    print('OUTPUT', destination, flush=True)
    native_kernel = backend_module._eval_thermo_kinetics_sparse_numba_core
    reference_kernel = build_reference_kernel()
    baseline_lu = wrapper_reference()
    current_lu = equations.factorize
    variants = {'baseline': (baseline_lu, reference_kernel),
                'lapack': (current_lu, reference_kernel),
                'products': (current_lu, native_kernel)}
    if not args.variants or args.variants[0] != 'baseline':
        raise ValueError('The baseline must be first for the saved-profile reference.')
    variants = {name: variants[name] for name in args.variants}
    result = dict(command=sys.argv, versions=dict(python=sys.version, cantera=ct.__version__, numpy=np.__version__),
                  threads={k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS')},
                  policy='No saved flame seed; kernels warmed separately; setup excluded from solve time.',
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), ROOT/'V2/equations.py',
                                           ROOT/'V2/materiales/kinetics_native.py',
                                           ROOT/'V2/materiales/species_backend_native.py']},
                  reference_kinetics='original loop placement and signed mass-action arithmetic',
                  reference_kernel_sha256=reference_kernel._audit_source_sha256,
                  full_stack_warmup=bool(args.warmup_flames), warmup_runs=[], runs=[])
    for label in args.cases:
        if label not in ('ch4_1', 'h2_1', 'h2_10'):
            raise ValueError('Unknown case: ' + label)
        multi = label != 'ch4_1'
        pressure = 10 if label == 'h2_10' else 1
        slope, curve = (.01, .02) if pressure == 10 else (.04, .08)
        case = FlameCase(mech=_resolve_mechanism('gri30.yaml'), fuel='H2' if multi else 'CH4',
                         oxidizer='O2:1,N2:3.76', phi=1., T_in=300., P=pressure*ct.one_atm,
                         width=.03, transport_model='multicomponent' if multi else 'mixture-averaged',
                         soret_enabled=multi, ratio=2.5, slope=slope, curve=curve, prune=.003)
        opts = solver.SolveOptions(verbose=False, profile=True, max_total_time_s=180.,
                                   max_refine_passes=12, require_grid_convergence=True, refine_max_points=1600,
                                   refine_ratio=2.5, refine_slope=slope, refine_curve=curve, refine_prune=.003,
                                   acceptance_criterion='cantera', residual_guard_inf=1e4,
                                   final_Finf_limit=1e4, refine_Finf_limit=1e4,
                                   jacobian_mode='block_tridiag', lag_multicomponent_transport=multi,
                                   multicomponent_bootstrap=multi, bootstrap_mesh_factor=2.)
        # Compile both kernels on representative nodal and FD-batch layouts,
        # excluded from measurements. Neither warm-up yields a flame seed.
        backend = backend_module.NativeSpeciesBackend(type('Problem', (), {'case': case, 'P': case.P})())
        gas = ct.Solution(case.mech)
        gas.set_equivalence_ratio(1., case.fuel, case.oxidizer)
        warm_times = {}
        for name, kernel in (('baseline', reference_kernel), ('products', native_kernel)):
            with patch.object(backend_module, '_eval_thermo_kinetics_sparse_numba_core', kernel):
                start = time.perf_counter()
                for n in (9, 55):
                    y = np.repeat(gas.Y[:, None], n, axis=1)
                    backend.eval_grid_thermo_kinetics_into(np.linspace(300., 2400., n), y,
                                                          np.empty_like(y), np.empty_like(y))
                warm_times[name] = time.perf_counter() - start
        print(json.dumps(dict(case=label, warmup_s=warm_times)), flush=True)
        if args.warmup_flames:
            for name, (lu, kernel) in variants.items():
                with ExitStack() as stack:
                    stack.enter_context(patch.object(equations, 'factorize', lu))
                    stack.enter_context(patch.object(solver, 'factorize', lu))
                    stack.enter_context(patch.object(backend_module, '_eval_thermo_kinetics_sparse_numba_core', kernel))
                    warm = native_solve(case, opts, bootstrap=multi, bootstrap_mesh_factor=2.)
                result['warmup_runs'].append(dict(case=label, variant=name, accepted=warm['accepted'],
                                                 time_s=warm['time_s']))
                (destination/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
                print(json.dumps(result['warmup_runs'][-1] | {'phase': 'warmup'}), flush=True)
                if not warm['accepted']:
                    raise RuntimeError('Rejected warm-up; not silently discarded.')
                del warm  # never passed to any measured solve
        reference_fields = None
        for repetition in range(args.repeat):
            names = list(variants)
            if repetition % 2:
                names.reverse()
            for name in names:
                lu, kernel = variants[name]
                with ExitStack() as stack:
                    stack.enter_context(patch.object(equations, 'factorize', lu))
                    stack.enter_context(patch.object(solver, 'factorize', lu))
                    stack.enter_context(patch.object(backend_module, '_eval_thermo_kinetics_sparse_numba_core', kernel))
                    run = native_solve(case, opts, bootstrap=multi, bootstrap_mesh_factor=2.)
                fields = {key: run.pop(key) for key in ('z', 'T', 'Y', 'u')}
                profile_file = destination / f'{label}_{name}_{repetition}.npz'
                np.savez_compressed(profile_file, **fields)
                if reference_fields is None:
                    reference_fields = fields
                same_grid = np.array_equal(fields['z'], reference_fields['z'])
                run.update(case=label, variant=name, repetition=repetition, configuration=asdict(case),
                           mechanism_sha256=hashlib.sha256(Path(case.mech).read_bytes()).hexdigest(),
                           options=asdict(opts), warmup_s=warm_times,
                           same_grid_as_baseline=same_grid,
                           differences_on_identical_grid={k: float(np.max(abs(fields[k]-reference_fields[k])))
                                                          for k in fields} if same_grid else None,
                           profile_sha256=hashlib.sha256(profile_file.read_bytes()).hexdigest())
                result['runs'].append(json_safe(run))
                (destination/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
                print(json.dumps({k: run[k] for k in ('case', 'variant', 'repetition', 'accepted', 'time_s',
                                                     'Su', 'n_points', 'Finf', 'same_grid_as_baseline')}), flush=True)
                if not run['accepted']:
                    raise RuntimeError('Rejected candidate retained in evidence; stop before promotion.')


if __name__ == '__main__':
    main()
