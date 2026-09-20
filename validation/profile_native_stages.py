"""Read-only stage audit of native cold starts (no production monkeypatches).

Warm each case before cProfile; measured times include profiling overhead and
must not be presented as uninstrumented production timings. Subtimers overlap.
Run: python validation/profile_native_stages.py --output validation/stages.json
"""
from pathlib import Path
import argparse
import cProfile
from dataclasses import asdict
import hashlib
import importlib.abc
import json
import platform
import pstats
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'V2'), str(ROOT / 'V2/materiales')]


class NoCantera(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'cantera' or fullname.startswith('cantera.'):
            raise ImportError('Cantera is blocked during the native stage audit')


sys.meta_path.insert(0, NoCantera())
from benchmark_soret_native import native_solve, json_safe
from config import FlameCase
from solver import SolveOptions
import numpy as np


def options(profile=True):
    return SolveOptions(
        verbose=False, profile=profile, max_total_time_s=180,
        max_refine_passes=12, require_grid_convergence=True,
        refine_max_points=1600, refine_ratio=2.5, refine_slope=.04,
        refine_curve=.08, refine_prune=.003, acceptance_criterion='cantera',
        residual_guard_inf=1e4, final_Finf_limit=1e4, refine_Finf_limit=1e4,
        jacobian_mode='block_tridiag', auto_bootstrap_grids=False,
    )


def compact(result):
    return {k: v for k, v in result.items() if k not in ('z', 'u', 'T', 'Y')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', default=['CH4_1', 'CH4_10', 'H2_1', 'H2_10'],
                        choices=['CH4_1', 'CH4_10', 'H2_1', 'H2_10'])
    args = parser.parse_args()
    output = dict(
        protocol='One excluded warmup per case, then cProfile; cold flame seed; '
                 'cached mechanism/molecular fits; four Numba and one BLAS thread '
                 'unless environment overrides; inclusive subtimers must not be summed.',
        options=asdict(options()), python=sys.version, platform=platform.platform(),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((ROOT / 'V2').rglob('*.py'))}, cases={},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in args.cases:
        fuel, pressure = name.split('_')
        multi = fuel == 'H2'
        case = FlameCase(fuel=fuel, P=101325. * float(pressure),
                         transport_model='multicomponent' if multi else 'mixture-averaged',
                         soret_enabled=multi, ratio=2.5, slope=.04, curve=.08, prune=.003)
        warm = native_solve(case, options(False), bootstrap=multi, bootstrap_mesh_factor=2.)
        print(name, 'warmup', json.dumps({k:v for k,v in compact(warm).items() if k != 'stages'}), flush=True)
        profiler = cProfile.Profile()
        start = time.perf_counter()
        profiler.enable()
        result = native_solve(case, options(), bootstrap=multi, bootstrap_mesh_factor=2.)
        profiler.disable()
        wall = time.perf_counter() - start
        stats = pstats.Stats(profiler)
        rows = []
        for (filename, line, function), (primitive, calls, own, cumulative, callers) in stats.stats.items():
            rows.append(dict(file=filename, line=line, function=function, calls=calls,
                             primitive_calls=primitive, own_s=own, inclusive_s=cumulative))
        rows.sort(key=lambda row: row['own_s'], reverse=True)
        record = dict(case=asdict(case), warmup=compact(warm), measured=compact(result),
                      wall_s=wall, functions=rows,
                      profiles_identical=all(np.array_equal(warm[k], result[k]) for k in ('z', 'u', 'T', 'Y')))
        output['cases'][name] = record
        args.output.write_text(json.dumps(json_safe(output), indent=2), encoding='utf-8')
        print(name, 'measured', json.dumps({k:v for k,v in compact(result).items() if k != 'stages'}), flush=True)
        print('top own times', json.dumps(rows[:12]), flush=True)
    output['cantera_imported'] = any(k == 'cantera' or k.startswith('cantera.') for k in sys.modules)
    args.output.write_text(json.dumps(json_safe(output), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
