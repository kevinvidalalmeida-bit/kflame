"""Ablate new molecular/thermal-fit caches using identical native fits.

This is NOT a comparison against the former exported Cantera coefficients.
All runs are cold flames; compilation is excluded via per-mode warmups.
"""
from pathlib import Path
import argparse
import json
import time
from dataclasses import asdict
from profile_native_stages import native_solve, options, compact, FlameCase, json_safe
import numpy as np
import kava.chemistry.transport as transport
import kava.chemistry.collision_integrals as collision
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=3)
    args = parser.parse_args()
    output = dict(protocol=__doc__, options=asdict(options(False)), cases={})
    original_pair = transport._pair_data
    original_cond = transport.native_conductivity_fits

    def uncached_pair(mech):
        result = transport._build_pair_data(mech)
        for a in result:
            a.setflags(write=False)
        return result

    def uncached_cond(mech, base):
        collision._CONDUCTIVITY_CACHE.clear()
        return original_cond(mech, base)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(json_safe(output), indent=2), encoding='utf-8')

    for fuel in ('CH4', 'H2'):
        multi = fuel == 'H2'
        case = FlameCase(fuel=fuel, transport_model='multicomponent' if multi else 'mixture-averaged',
                         soret_enabled=multi, ratio=2.5, slope=.04, curve=.08, prune=.003)
        record = output['cases'][fuel] = dict(case=asdict(case), warmups={}, pairs=[])

        def run(mode):
            transport._pair_data = original_pair if mode == 'cached' else uncached_pair
            transport.native_conductivity_fits = original_cond if mode == 'cached' else uncached_cond
            # Reset both modes equally; let each cache fill within the flame.
            transport._PAIR_CACHE.clear()
            collision._CONDUCTIVITY_CACHE.clear()
            try:
                start = time.perf_counter()
                result = native_solve(case, options(False), bootstrap=multi, bootstrap_mesh_factor=2.)
                elapsed = time.perf_counter() - start
            finally:
                transport._pair_data = original_pair
                transport.native_conductivity_fits = original_cond
            summary = compact(result)
            summary.pop('stages')
            summary['wall_s'] = elapsed
            print(fuel, mode, json.dumps(summary), flush=True)
            return summary, {k:result[k] for k in ('z','u','T','Y')}

        for mode in ('uncached', 'cached'):
            record['warmups'][mode], _ = run(mode)
        for i in range(args.pairs):
            pair, fields = {}, {}
            for mode in (('uncached', 'cached') if i % 2 == 0 else ('cached', 'uncached')):
                pair[mode], fields[mode] = run(mode)
            pair['identical'] = all(np.array_equal(fields['cached'][k], fields['uncached'][k]) for k in fields['cached'])
            record['pairs'].append(pair)
            save()
            assert pair['identical'] and all(pair[m]['accepted'] for m in ('uncached', 'cached'))
        medians = {mode:float(np.median([p[mode]['wall_s'] for p in record['pairs']])) for mode in ('uncached','cached')}
        record['median_wall_s'] = medians
        record['reduction_percent'] = 100*(1-medians['cached']/medians['uncached'])
        print(fuel, 'medians', medians, record['reduction_percent'], flush=True)
        save()


if __name__ == '__main__':
    main()
