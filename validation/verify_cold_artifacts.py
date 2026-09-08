"""Independent integrity audit of cold-start profiles and final acceptance.

This checks the raw files rather than trusting the aggregate report's success
booleans. Performance conclusions still require the paired statistical report.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def verify(folder):
    data = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    count = 0
    for run in data['runs']:
        profile = folder/f"{run['case']}_{run['variant']}_{run['repetition']}.npz"
        assert hashlib.sha256(profile.read_bytes()).hexdigest() == run['profile_sha256'], profile
        with np.load(profile) as fields:
            z, t, y, u = (fields[k] for k in ('z', 'T', 'Y', 'u'))
            assert np.all(np.diff(z) > 0)
            assert len(z) == len(t) == len(u) == y.shape[1] == run['n_points']
            assert all(np.isfinite(v).all() for v in (z, t, y, u))
            assert np.min(t) >= 200 and np.min(y) >= -1e-7
            np.testing.assert_allclose(float(np.max(abs(y.sum(axis=0)-1))), run['species_sum_error'], rtol=0, atol=1e-15)
            np.testing.assert_allclose(float(u[0]), run['Su'], rtol=0, atol=0)
            np.testing.assert_allclose(float(z[-1]-z[0]), run['width'], rtol=1e-14, atol=0)
        final = run['stages'][-1]
        assert run['accepted'] and final['grid_converged'] and final['final_accepted']
        assert np.isfinite(run['Finf']) and run['Finf'] <= 1e4
        assert final['weighted_step_norm_final'] <= 1.
        assert run['options']['residual_guard_inf'] == 1e4
        assert run['options']['require_grid_convergence']
        count += 1
    snapshot_count = 0
    warm_count = 0
    for run in data.get('warmup_runs', []):
        if 'profile_sha256' not in run:
            continue
        path = folder/f"{run['case']}_{run['variant']}_warmup.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == run['profile_sha256'], path
        with np.load(path) as fields:
            z,t,y,u=(fields[k] for k in ('z','T','Y','u'))
            assert len(z)==len(t)==len(u)==y.shape[1]
            assert np.all(np.diff(z)>0)
        warm_count += 1
    for relative, digest in data['source_sha256'].items():
        path = folder/'sources'/relative
        if (folder/'sources').exists():
            assert path.exists(), path
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
            snapshot_count += 1
    return dict(directory=str(folder), measured_profiles_verified=count,
                warmup_profiles_verified=warm_count,
                source_snapshots_verified=snapshot_count,
                source_snapshot_available=(folder/'sources').exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = dict(campaigns=[verify(p) for p in args.directories])
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
