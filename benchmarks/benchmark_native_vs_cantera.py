"""Compare KFLAME and Cantera over matched FGM sweeps and cold flames.

Each timed subprocess uses the public generators, fixed CPU threads and no
saved native profile seed. First-use JIT is recorded separately from pairs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from importlib import metadata

ROOT = Path(__file__).resolve().parents[1]
PHIS = '0.7,0.9,1,1.1,1.4'


def versions():
    result = {'python': sys.version}
    for name in ('numpy', 'scipy', 'numba', 'llvmlite', 'PyYAML', 'cantera', 'matplotlib'):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result
def worker(args):
    import numpy as np
    if args.backend == 'cantera':
        from kflame.reference import generate_fgm as generator
    else:
        from kflame.fgm import generate as generator
    mechanism = ROOT / 'src/kflame/chemistry/data/gri30.yaml'
    common = ['--mech', str(mechanism), '--phi-values', args.phis,
              '--output-root', str(args.output), '--run-name', 'run', '--save-raw-profiles']
    if args.backend != 'cantera':
        common += ['--disable-seed-cache', '--parallel-workers', '1',
                   '--numba-kinetics-threads', '4', '--max-flame-time-s', '180']
    record = {'versions': versions(), 'backend': args.backend, 'scope': args.scope,
              'mechanism_sha256': hashlib.sha256(mechanism.read_bytes()).hexdigest()}
    if args.scope == 'fgm':
        sys.argv = ['benchmark', *common]
        generator.main()
        run = args.output / 'run'
        meta = json.loads((run / 'metadata.json').read_text(encoding='utf-8'))
        record.update(wall_s=meta['runtime_s'], accepted=meta['table_validation']['valid'])
        with np.load(run / 'fgm_table.npz', allow_pickle=True) as table:
            record.update(phi=table['phi_grid'].tolist(), Su=table['Su'].tolist(),
                          nodes=table['n_points'].tolist(), width=table['width'].tolist())
        if args.backend != 'cantera':
            record['accepted'] = bool(meta['all_final_accepted'])
    else:
        inputs = generator.build_argparser().parse_args(common)
        progress = generator.parse_progress_weights(inputs.progress_species)
        start = time.perf_counter()
        if args.backend == 'cantera':
            flame, _ = generator.solve_flame_cantera(float(args.phis), inputs, None, progress)
            accepted = True  # Successful Cantera solve, not a KFLAME residual certificate.
        else:
            mechanism_data = generator.load_mechanism(inputs.mech)
            flame, _, _ = generator.solve_flame_native(
                float(args.phis), inputs, mechanism_data, generator.make_solve_options(inputs), progress)
            accepted = bool(flame.solve_ok and flame.final_accepted)
        record.update(wall_s=time.perf_counter()-start, accepted=accepted,
                      phi=flame.phi, Su=flame.Su_m_per_s, nodes=flame.n_points,
                      width=flame.width_m, reported_inner_s=flame.solve_time_s)
        np.savez_compressed(args.output / 'profile.npz', z=flame.z, T=flame.T, Y=flame.Y,
                            u=flame.u, c=flame.c, qdot=flame.qdot)
    (args.output / 'measurement.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    if not record['accepted']:
        raise RuntimeError('Solver did not accept every flame')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--old-python', type=Path)
    parser.add_argument('--new-python', type=Path)
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--phis', default=PHIS)
    parser.add_argument('--backend', choices=('native', 'cantera'))
    parser.add_argument('--scope', choices=('fgm', 'single'), default='fgm')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.backend:
        worker(args)
        return
    if not args.old_python or not args.new_python:
        parser.error('Specify both --old-python and --new-python')
    variants = [('native311', args.old_python, 'native'),
                ('cantera311', args.old_python, 'cantera'),
                ('native313', args.new_python, 'native'),
                ('cantera313', args.new_python, 'cantera')]
    result = {'protocol': __doc__, 'threads': {'numba': 4, 'blas': 1},
              'warmups': [], 'measurements': []}
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='4',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONIOENCODING='utf-8')

    def run(variant, scope, phis, label):
        name, python, backend = variant
        folder = args.output / f'{label}_{name}_{scope}_{phis.replace(",", "_")}'
        folder.mkdir(exist_ok=False)
        child_env = dict(env)
        child_env['PYTHONPATH'] = str(ROOT / 'tests/no_cantera') if backend != 'cantera' else ''
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                [str(python.resolve()), str(Path(__file__).resolve()), '--backend', backend,
                 '--scope', scope, '--phis', phis, '--output', str(folder)],
                cwd=ROOT, env=child_env, capture_output=True, text=True, encoding='utf-8', timeout=900)
            (folder / 'console.log').write_text(completed.stdout+completed.stderr, encoding='utf-8')
            path = folder / 'measurement.json'
            rec = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'accepted': False}
            rec['returncode'] = completed.returncode
        except subprocess.TimeoutExpired:
            rec = {'accepted': False, 'error': '900 s process limit exceeded'}
        rec.update(process_s=time.perf_counter()-start, variant=name, scope=scope,
                   requested_phi=phis, folder=str(folder.relative_to(args.output)))
        print(label, name, scope, phis, 'accepted=', rec['accepted'],
              'wall_s=', round(rec.get('wall_s', rec['process_s']), 3), flush=True)
        return rec

    # Separate first-use/compile observations from repeated, unseeded measurements.
    for variant in variants:
        result['warmups'].append(run(variant, 'fgm', args.phis, 'warmup'))
        (args.output / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for pair in range(args.pairs):
        order = variants[pair:] + variants[:pair]
        for scope, phis in [('fgm', args.phis), *[('single', p) for p in args.phis.split(',')]]:
            for variant in order:
                result['measurements'].append(run(variant, scope, phis, f'pair{pair}'))
                (args.output / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
