"""Paired native FGM sweeps in fresh processes with profile seed caches disabled."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--pairs', type=int, default=3)
    ap.add_argument('--baseline', choices=['fd', 'analytic'], default='fd')
    ap.add_argument('--candidate', choices=['analytic', 'spatial'], default='analytic')
    args = ap.parse_args()
    modes = (args.baseline, args.candidate)
    if modes[0] == modes[1]:
        ap.error('Choose different modes')
    root = Path(__file__).resolve().parents[1]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, NUMBA_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONIOENCODING='utf-8',
               PYTHONPATH=str(root/'tests/no_cantera'))
    data = dict(modes=modes, protocol=__doc__, warmups={}, pairs=[])

    def save():
        (args.output/'summary.json').write_text(json.dumps(data, indent=2), encoding='utf-8')

    def run(mode, label):
        folder = args.output/f'{label}_{mode}'
        command = [sys.executable, '-m', 'kflame.fgm.generate',
                   '--phi-values', '0.7,0.9,1,1.1,1.4', '--output-root', str(folder),
                   '--run-name', 'run', '--save-raw-profiles', '--disable-seed-cache',
                   '--parallel-workers', '1', '--numba-kinetics-threads', '4',
                   '--max-flame-time-s', '180',
                   '--analytic-chemistry' if mode == 'analytic' else '--no-analytic-chemistry']
        command.append('--analytic-spatial' if mode == 'spatial' else '--no-analytic-spatial')
        start = time.perf_counter()
        result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True,
                                encoding='utf-8', timeout=900)
        folder.mkdir(exist_ok=True)
        (folder/'console.log').write_text(result.stdout+result.stderr, encoding='utf-8')
        rec = dict(process_s=time.perf_counter()-start, returncode=result.returncode, accepted=False)
        meta_path = folder/'run/metadata.json'
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            rec.update(wall_s=meta['runtime_s'], accepted=bool(meta['all_final_accepted']
                       and meta['table_validation']['valid']), analytic_chemistry=meta['analytic_chemistry'], analytic_spatial=meta['analytic_spatial'])
            with np.load(folder/'run/fgm_table.npz', allow_pickle=True) as table:
                for k in ('Su', 'phi_grid', 'n_points', 'width'):
                    rec[k] = table[k].tolist()
        print(label, mode, rec, flush=True)
        return rec

    for mode in modes:
        data['warmups'][mode] = run(mode, 'warmup')
        save()
    for i in range(args.pairs):
        pair = {}
        for mode in (modes if i%2 == 0 else modes[::-1]):
            pair[mode] = run(mode, str(i))
        if all(pair[m]['accepted'] for m in modes):
            a, b = pair[modes[0]], pair[modes[1]]
            pair['same_phi_grid'] = a['phi_grid'] == b['phi_grid']
            if pair['same_phi_grid']:
                pair['max_speed_relative_difference'] = float(np.max(abs(
                    np.array(a['Su'])-b['Su'])/np.array(a['Su'])))
        data['pairs'].append(pair)
        save()
    data['all_accepted'] = all(p[m]['accepted'] for p in data['pairs'] for m in modes)
    if data['all_accepted']:
        data['median_wall_s'] = {m:float(np.median([p[m]['wall_s'] for p in data['pairs']]))
                                 for m in modes}
        data['reduction_percent'] = 100*(1-data['median_wall_s'][modes[1]]/data['median_wall_s'][modes[0]])
    save()


if __name__ == '__main__':
    main()
