"""Paired full-flame benchmark of native thermal reuse.

No production edits; native-only. Timings exclude per-mode warmups and include
setup; mechanism/fit caches stay warm. The profile seed is cold for every run.
"""
from pathlib import Path
import argparse
from dataclasses import asdict
import hashlib
import json
import os
import platform
import time

from profile_native_stages import ROOT, native_solve, json_safe, options, compact, FlameCase
import numpy as np
import numba
from kava.flame.problem import FreeFlameProblem
from kava.chemistry.backend import NativeSpeciesBackend
def local_check():
    results = []
    for mechanism in ('gri30.yaml', 'h2o2.yaml'):
        for pressure in (101325., 1013250.):
            case = FlameCase(mech=mechanism, fuel='H2', P=pressure)
            problem = FreeFlameProblem(case, n_points=8)
            problem.assume_finite_y = True
            backend = NativeSpeciesBackend(problem)
            ns = problem.n_species
            nv = ns + 2
            base_t = np.array([300., 999.9999, 1000., 1000.0001, 1800., 2500.])
            rng = np.random.default_rng(1920)
            base_y = rng.uniform(.01, 1., (ns, base_t.size))
            base_y /= base_y.sum(axis=0)
            base_y[0, 1] = -1e-8
            base_y[1, 1] = 0.
            temperature = np.repeat(base_t, nv)
            y = np.repeat(base_y, nv, axis=1)
            for j in range(base_t.size):
                temperature[j * nv + 1] += abs(base_t[j]) * 1e-5 + 1e-10
                for k in range(ns):
                    y[k, j * nv + k + 2] += abs(base_y[k, j]) * 1e-5 + 1e-10
            values = []
            for method in (backend.eval_grid_thermo_kinetics_into,
                           backend.eval_jacobian_thermo_kinetics_into):
                omega = np.empty_like(y)
                enthalpy = np.empty_like(y)
                rho, cp = method(temperature, y, omega, enthalpy)
                values.append((rho, cp, omega, enthalpy))
            diffs = {name: float(np.max(np.abs(a-b))) for name, a, b in
                     zip(('rho', 'cp', 'omega', 'h'), values[0], values[1])}
            identical = all(np.array_equal(a,b) for a,b in zip(*values))
            results.append(dict(mechanism=mechanism, pressure=pressure, identical=identical, max_abs=diffs))
            print('local', json.dumps(results[-1]), flush=True)
            if not identical:
                raise AssertionError('The candidate must first reproduce local values bit for bit')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--cases', nargs='+', default=['CH4_1', 'CH4_10', 'H2_1', 'H2_10'])
    args = parser.parse_args()
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except ImportError:
        pools = None
    output = dict(protocol=__doc__, options=asdict(options(False)), local=local_check(), cases={},
                  mode='production', platform=platform.platform(),
                  threads=dict(numba=numba.get_num_threads(), blas=pools,
                               env={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'MKL_NUM_THREADS')}),
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (ROOT / 'src/kava/flame/equations.py', ROOT / 'src/kava/chemistry/backend.py',
                                           ROOT / 'src/kava/chemistry/jacobian.py')})
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(json_safe(output), indent=2), encoding='utf-8')

    for name in args.cases:
        fuel, pressure = name.split('_')
        multi = fuel == 'H2'
        case = FlameCase(fuel=fuel, P=101325. * float(pressure),
                         transport_model='multicomponent' if multi else 'mixture-averaged',
                         soret_enabled=multi, ratio=2.5, slope=.04, curve=.08, prune=.003)
        record = output['cases'][name] = dict(case=asdict(case), warmups={}, pairs=[])

        def run(mode):
            original_method = NativeSpeciesBackend.eval_jacobian_thermo_kinetics_into
            if mode == 'baseline':
                NativeSpeciesBackend.eval_jacobian_thermo_kinetics_into = NativeSpeciesBackend.eval_grid_thermo_kinetics_into
            try:
                start = time.perf_counter()
                result = native_solve(case, options(False), bootstrap=multi, bootstrap_mesh_factor=2.)
                wall = time.perf_counter() - start
            finally:
                NativeSpeciesBackend.eval_jacobian_thermo_kinetics_into = original_method
            summary = compact(result)
            summary.pop('stages')
            summary['wall_s'] = wall
            print(name, mode, json.dumps(summary), flush=True)
            return summary, {k:result[k] for k in ('z','u','T','Y')}

        for mode in ('baseline', 'grouped'):
            record['warmups'][mode], _ = run(mode)
        for i in range(args.pairs):
            pair, fields = {}, {}
            for mode in (('baseline', 'grouped') if i % 2 == 0 else ('grouped', 'baseline')):
                pair[mode], fields[mode] = run(mode)
            pair['identical_fields'] = all(np.array_equal(fields['baseline'][k], fields['grouped'][k]) for k in fields['baseline'])
            if not pair['identical_fields']:
                pair['max_abs_differences'] = {
                    k: float(np.max(np.abs(fields['baseline'][k]-fields['grouped'][k])))
                    if fields['baseline'][k].shape == fields['grouped'][k].shape else None
                    for k in fields['baseline']}
            record['pairs'].append(pair)
            save()
        times = {mode: float(np.median([p[mode]['wall_s'] for p in record['pairs']]))
                 for mode in ('baseline', 'grouped')}
        record['median_wall_s'] = times
        record['reduction_percent'] = 100 * (1-times['grouped']/times['baseline'])
        record['all_accepted'] = all(p[m]['accepted'] for p in record['pairs'] for m in times)
        record['all_identical_fields'] = all(p['identical_fields'] for p in record['pairs'])
        print(name, 'summary', json.dumps({k:v for k,v in record.items() if k not in ('pairs','case','warmups')}), flush=True)
        save()
        if not record['all_accepted'] or not record['all_identical_fields']:
            raise AssertionError('Thermal reuse failed full-flame acceptance or exact-field equivalence')


if __name__ == '__main__':
    main()
