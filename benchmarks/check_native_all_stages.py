"""Summarize native integration outputs without importing Cantera.

Use with tests/no_cantera on PYTHONPATH. Raw profiles remain in the run
directory; this writes compact evidence, not a new solver timing benchmark.
"""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
import numpy as np
from kava.reference.analyze_soret import elemental_flux_diagnostics


def load(path):
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k].copy() for k in data.files}


def differences(actual, reference):
    assert actual.shape == reference.shape
    absolute = float(np.max(np.abs(actual - reference)))
    return dict(max_absolute=absolute,
                relative_to_reference_peak=absolute / max(float(np.max(np.abs(reference))), 1e-300),
                identical=bool(np.array_equal(actual, reference)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--cold-name', default='content_key_cold')
    parser.add_argument('--parallel-name', default='content_key_parallel')
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--soret-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    versions = {}
    for name in ('numpy', 'scipy', 'numba', 'PyYAML', 'cantera'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    result = dict(python=sys.version, versions=versions, integration={}, old_fit_comparison={})
    fields = ('T', 'u', 'Y', 'rho', 'cp_mass', 'conductivity', 'qdot', 'omega_c', 'beta')
    for name in (args.cold_name, args.parallel_name):
        folder = args.run_root / name
        meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        table = load(folder / 'fgm_table.npz')
        assert meta['all_final_accepted'] and meta['cantera_version'] is None
        assert meta['initialization_backend'] == 'native_nasa7_hp'
        assert meta['postprocessing_backend'] == 'native'
        assert all(np.isfinite(table[k]).all() for k in fields)
        result['integration'][name] = {
            k: meta[k] for k in ('all_final_accepted', 'initialization_backend',
                                'postprocessing_backend', 'parallel_used', 'parallel_workers_resolved',
                                'seed_cache_used', 'runtime_s', 'n_phi_table')}
        result['integration'][name].update(Su=table['Su'].tolist(), n_points=table['n_points'].tolist(),
                                          phi=table['phi_grid'].tolist())
    actual = load(args.run_root / args.cold_name / 'fgm_table.npz')
    reference = load(args.baseline / 'fgm_table.npz')
    for k in fields + ('c_grid', 'Z_grid', 'phi_grid', 'Su', 'n_points'):
        result['old_fit_comparison'][k] = differences(actual[k], reference[k])
    profiles = {}
    for path in sorted((args.run_root / args.cold_name / 'raw_profiles').glob('*.npz')):
        current, previous = load(path), load(args.baseline / 'raw_profiles' / path.name)
        profiles[path.name] = {k: differences(current[k], previous[k]) for k in ('z', 'u', 'T', 'Y')}
    result['old_fit_raw_profiles'] = profiles
    summary = json.loads((args.soret_dir / 'summary.json').read_text(encoding='utf-8'))
    run = summary['runs'][0]
    assert run['accepted'] and run['solver'] == 'native'
    result['soret_10atm'] = {k:run[k] for k in ('accepted', 'Su', 'n_points', 'Finf', 'species_sum_error', 'mass_flow_error')}
    result['soret_10atm']['elemental_flux'] = elemental_flux_diagnostics(
        summary['case'], load(args.soret_dir / 'native_0.npz'), 'native')
    result['cantera_imported'] = any(k == 'cantera' or k.startswith('cantera.') for k in sys.modules)
    assert not result['cantera_imported']
    sources = sorted((ROOT / 'src/kava').rglob('*.py'))
    result['source_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
