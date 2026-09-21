"""Paired whole-FGM ablation of vectorized Jacobian batch preparation.

The legacy loop is kept in a test oracle only, never a production CLI mode.
Each run is a new process, no persistent flame seeds, native-only import guard.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=3)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='4',
               PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in
                                         ('tests/no_cantera', 'tests', 'src')))
    runner = """
import sys
import kflame.flame.equations as equations
from test_perturbation_batches import legacy_perturbation_states
mode = sys.argv.pop(1)
if mode == 'legacy':
    equations._center_perturbation_states = legacy_perturbation_states
import kflame.fgm.generate as generate_fgm_tables_native
generate_fgm_tables_native.main()
"""
    result = dict(protocol=__doc__, pairs=[], warmups={}, threads=dict(numba=4, blas=1))
    def run(mode, label):
        name = label + '_' + mode
        command = [sys.executable, '-c', runner, mode, '--phi-values', '0.7,0.9,1,1.1,1.4',
                   '--parallel-workers', '1', '--disable-seed-cache', '--save-raw-profiles',
                   '--output-root', str(args.output_root.resolve()), '--run-name', name]
        start = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        wall = time.perf_counter() - start
        folder = args.output_root / name
        (args.output_root / (name + '.log')).write_text(completed.stdout + completed.stderr, encoding='utf-8')
        if completed.returncode:
            raise RuntimeError(f'{name} failed; see its log')
        meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        assert meta['all_final_accepted'] and meta['cantera_version'] is None
        record = dict(process_s=wall, fgm_s=meta['runtime_s'], accepted=meta['all_final_accepted'])
        print(name, record, flush=True)
        return record, folder
    for mode in ('legacy', 'vectorized'):
        result['warmups'][mode], _ = run(mode, 'warmup')
    for i in range(args.pairs):
        pair, folders = {}, {}
        for mode in (('legacy', 'vectorized') if i % 2 == 0 else ('vectorized', 'legacy')):
            pair[mode], folders[mode] = run(mode, f'pair_{i}')
        with np.load(folders['legacy'] / 'fgm_table.npz', allow_pickle=True) as a, np.load(
                folders['vectorized'] / 'fgm_table.npz', allow_pickle=True) as b:
            fields = ('phi_grid', 'Z_grid', 'c_grid', 'Su', 'n_points', 'T', 'u', 'Y',
                      'rho', 'cp_mass', 'conductivity', 'qdot', 'omega_c', 'beta')
            pair['identical'] = all(np.array_equal(a[k], b[k]) for k in fields)
        assert pair['identical']
        result['pairs'].append(pair)
        (args.output_root / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    result['median_fgm_s'] = {m:float(np.median([p[m]['fgm_s'] for p in result['pairs']]))
                              for m in ('legacy', 'vectorized')}
    result['reduction_percent'] = 100 * (1 - result['median_fgm_s']['vectorized'] / result['median_fgm_s']['legacy'])
    (args.output_root / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
