"""Alternating full cold-profile flames; warmups excluded, identical settings."""
import os
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
from pathlib import Path
import argparse
from dataclasses import asdict
import hashlib
import json
import platform
import time
import numpy as np
import numba
from kflame.benchmarks.soret import benchmark_options, compact_result, native_solve, json_safe
from kflame.flame.config import FlameCase


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--pairs', type=int, default=3)
    ap.add_argument('--cases', nargs='+', default=['CH4_1', 'CH4_10', 'H2_1', 'H2_10'])
    ap.add_argument('--baseline', choices=['fd', 'analytic'], default='fd')
    ap.add_argument('--candidate', choices=['analytic', 'spatial'], default='analytic')
    args = ap.parse_args()
    modes = (args.baseline, args.candidate)
    if modes[0] == modes[1]:
        ap.error('Choose different modes')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    out = dict(modes=modes, protocol=__doc__, platform=platform.platform(), numba_threads=numba.get_num_threads(),
               options=asdict(benchmark_options(True)), cases={}, source_sha256={
                   str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [root/'src/kflame/chemistry/analytic.py', root/'src/kflame/flame/equations.py',
                             root/'src/kflame/flame/analytic_jacobian.py', root/'src/kflame/flame/solver.py']})

    def save():
        args.output.write_text(json.dumps(json_safe(out), indent=2), encoding='utf-8')

    for name in args.cases:
        fuel, pressure = name.split('_')
        multi = fuel == 'H2'
        case = FlameCase(fuel=fuel, P=101325.*float(pressure),
                         transport_model='multicomponent' if multi else 'mixture-averaged',
                         soret_enabled=multi, ratio=2.5, slope=.04, curve=.08, prune=.003)
        record = out['cases'][name] = dict(case=asdict(case), warmups={}, pairs=[])

        def run(mode, label):
            opts = benchmark_options(True)
            opts.analytic_chemistry = mode == 'analytic'
            opts.analytic_spatial = mode == 'spatial'
            start = time.perf_counter()
            result = native_solve(case, opts, bootstrap=multi, bootstrap_mesh_factor=2.)
            summary = compact_result(result)
            summary['wall_s'] = time.perf_counter()-start
            np.savez_compressed(args.output.parent/f'{name}_{label}_{mode}.npz',
                                **{k:result[k] for k in ('z', 'u', 'T', 'Y')})
            print(name, label, mode, {k:summary[k] for k in ('wall_s','accepted','Su','n_points','Finf')}, flush=True)
            return summary, result

        for mode in modes:
            record['warmups'][mode], _ = run(mode, 'warmup')
            save()
        for i in range(args.pairs):
            pair, fields = {}, {}
            for mode in (modes if i%2 == 0 else modes[::-1]):
                pair[mode], fields[mode] = run(mode, str(i))
            a, b = fields[modes[0]], fields[modes[1]]
            pair['speed_relative_difference'] = abs(a['Su']-b['Su'])/abs(a['Su'])
            pair['temperature_interpolated_max_K'] = float(np.max(abs(
                a['T']-np.interp(a['z'], b['z'], b['T']))))
            pair['species_interpolated_max'] = float(max(np.max(abs(
                ya-np.interp(a['z'], b['z'], yb))) for ya,yb in zip(a['Y'],b['Y'])))
            record['pairs'].append(pair)
            save()
        record['median_wall_s'] = {m:float(np.median([p[m]['wall_s'] for p in record['pairs']]))
                                   for m in modes}
        record['reduction_percent'] = 100*(1-record['median_wall_s'][modes[1]]/record['median_wall_s'][modes[0]])
        record['all_accepted'] = all(p[m]['accepted'] for p in record['pairs'] for m in modes)
        save()
        print(name, record['median_wall_s'], record['reduction_percent'], flush=True)


if __name__ == '__main__':
    main()
