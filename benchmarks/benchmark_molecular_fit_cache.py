"""Paired cold-flame ablation of molecular-fit reuse, with identical physics.

Run from the repository root. The two warm-up solves are excluded from timing.
Every measured solve starts without a flame seed and with an empty fit cache.
"""
from pathlib import Path
import argparse
import json
import time
from dataclasses import asdict, replace

from kflame.benchmarks.soret import benchmark_options, native_solve
from kflame.flame.config import FlameCase
import kflame.chemistry.collision_integrals as collision
import kflame.chemistry.multicomponent as multi
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--pressure-atm', type=float, default=1.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    case = FlameCase(fuel='H2', P=101325. * args.pressure_atm,
                     transport_model='multicomponent', soret_enabled=True,
                     ratio=2.5, slope=.04, curve=.08, prune=.003)
    opts = replace(benchmark_options(False), max_total_time_s=120)
    normal = multi.native_transport_fits
    def uncached(mech, base):
        fits = collision._build_transport_fits(mech, base)
        for array in fits:
            array.setflags(write=False)
        return fits

    def run(mode):
        multi.native_transport_fits = normal if mode == 'cached' else uncached
        collision._FIT_CACHE.clear()
        start = time.perf_counter()
        result = native_solve(case, opts, bootstrap=True, bootstrap_mesh_factor=2.)
        elapsed = time.perf_counter() - start
        fields = {name: result.pop(name) for name in ('z', 'T', 'Y', 'u')}
        result.pop('stages')
        result['wall_s'] = elapsed
        result['mode'] = mode
        return result, fields

    output = dict(case=asdict(case), options=asdict(opts), warmups=[], pairs=[],
                  protocol='Alternating pairs; no profile seed; empty fit cache before each solve; two excluded warmups')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for mode in ('uncached', 'cached'):
            result, _ = run(mode)
            output['warmups'].append(result)
            print('warmup', json.dumps(result), flush=True)
        for i in range(args.pairs):
            pair = {}
            fields = {}
            for mode in (('uncached', 'cached') if i % 2 == 0 else ('cached', 'uncached')):
                pair[mode], fields[mode] = run(mode)
                print('measured', i, json.dumps(pair[mode]), flush=True)
            pair['identical_fields'] = all(np.array_equal(fields['cached'][k], fields['uncached'][k]) for k in fields['cached'])
            pair['max_abs_field_difference'] = {
                k: float(np.max(np.abs(fields['cached'][k] - fields['uncached'][k])))
                if fields['cached'][k].shape == fields['uncached'][k].shape else None for k in fields['cached']}
            output['pairs'].append(pair)
            args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')
        for field in ('wall_s', 'time_s', 'setup_time_s'):
            medians = {mode: float(np.median([p[mode][field] for p in output['pairs']])) for mode in ('uncached', 'cached')}
            output[field] = dict(medians, reduction_percent=100*(1-medians['cached']/medians['uncached']))
        output['all_accepted'] = all(p[mode]['accepted'] for p in output['pairs'] for mode in ('uncached', 'cached'))
        output['all_identical_fields'] = all(p['identical_fields'] for p in output['pairs'])
        args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')
        print(json.dumps({k:v for k,v in output.items() if k not in ('pairs', 'warmups', 'options')}, indent=2), flush=True)
    finally:
        multi.native_transport_fits = normal


if __name__ == '__main__':
    main()
