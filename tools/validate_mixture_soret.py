"""Independent complete-flame check of native mixture-averaged Soret.

Requires the optional Cantera reference. Writes every case, including failures;
timings are single observations including native first-use costs, not speed claims.
"""
import argparse
import json
from pathlib import Path
import time

import cantera as ct
import numpy as np

from kflame import solve_flame
from kflame.chemistry.mechanism import resolve_mechanism


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for mechanism in ('h2o2.yaml', 'gri30.yaml'):
        for atm in (1, 10):
            row = dict(mechanism=mechanism, pressure_atm=atm, accepted=False)
            records.append(row)
            try:
                actual = solve_flame(mechanism=mechanism, fuel='H2', pressure=atm * 101325.,
                                     soret=True, species=('H2', 'O2', 'H2O', 'OH'),
                                     output=args.output / f'{Path(mechanism).stem}_{atm}', max_time=180)
                gas = ct.Solution(resolve_mechanism(mechanism), transport_model='mixture-averaged')
                gas.TP = 300., atm * 101325.
                gas.set_equivalence_ratio(1., 'H2', 'O2:1,N2:3.76')
                flame = ct.FreeFlame(gas, width=.03)
                flame.set_refine_criteria(ratio=2.5, slope=.04, curve=.08, prune=.003)
                flame.soret_enabled = True
                started = time.perf_counter()
                flame.solve(loglevel=0, auto=True)
                reference_s = time.perf_counter() - started
                np.savez_compressed(args.output / f'reference_{Path(mechanism).stem}_{atm}.npz',
                                    z=flame.grid, T=flame.T, Y=flame.Y, u=flame.velocity)
                difference = abs(actual['Su'] / flame.velocity[0] - 1)
                row.update(accepted=actual['accepted'], native_Su=actual['Su'],
                           cantera_Su=float(flame.velocity[0]), relative_Su_difference=difference,
                           native_nodes=len(actual['z']), cantera_nodes=len(flame.grid),
                           native_s=actual['runtime_s'], cantera_s=reference_s,
                           native_width=float(actual['z'][-1]), cantera_width=float(flame.grid[-1]),
                           species_sum_error=float(np.max(np.abs(actual['Y'].sum(axis=0)-1))))
                if difference > .02:
                    raise AssertionError('Speed differs by >2%; investigate before promoting this case')
            except (RuntimeError, ValueError, AssertionError, ct.CanteraError) as exc:
                row.update(accepted=False, error=str(exc))
            (args.output / 'summary.json').write_text(json.dumps(dict(cantera=ct.__version__, cases=records), indent=2), encoding='utf-8')
            print(json.dumps(row), flush=True)
    if not all(row['accepted'] for row in records):
        raise SystemExit('At least one mixture-Soret validation failed; inspect summary.json')


if __name__ == '__main__':
    main()
