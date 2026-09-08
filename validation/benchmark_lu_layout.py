"""Isolated LAPACK-wrapper ablation, not an alternative production solver."""
import inspect
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'V2'))
import numpy as np
import equations
from scipy.linalg import lu_factor, lu_solve


def wrapper_reference():
    source = inspect.getsource(equations.factorize)
    begin = source.index('            factors, pivots, info = _block_getrf')
    end = source.index('            lu_blocks.append(lu)', begin)
    source = source[:begin] + '            lu = lu_factor(mat, overwrite_a=True, check_finite=False)\n' + source[end:]
    begin = source.index('                solved, info = _block_getrs')
    end = source.index('                cprime[i] = solved', begin) + len('                cprime[i] = solved')
    source = source[:begin] + '                cprime[i] = lu_solve(lu, jmat.upper[i], check_finite=False)' + source[end:]
    namespace = dict(vars(equations))
    namespace.update(lu_factor=lu_factor, lu_solve=lu_solve)
    exec(compile(source, '<SciPy-wrapper-reference>', 'exec'), namespace)
    return namespace['factorize']


def main():
    functions = {'wrapper_reference': wrapper_reference(), 'lapack_candidate': equations.factorize}
    rng = np.random.default_rng(20260905)
    results = []
    for n in (9, 36, 261, 854):
        nv = 55
        diag = rng.normal(size=(n, nv, nv)) + 100 * np.eye(nv)
        matrix = equations.BlockTridiagJacobian(
            rng.normal(size=(n-1, nv, nv)), diag,
            rng.normal(size=(n-1, nv, nv)))
        refs = {name: fn(matrix) for name, fn in functions.items()}
        for field in ('lu_blocks_array', 'pivots_array', 'cprime'):
            np.testing.assert_array_equal(refs['wrapper_reference'][field], refs['lapack_candidate'][field])
        times = {name: [] for name in functions}
        batch = max(1, 854 // n)
        for repetition in range(9):
            order = list(functions) if repetition % 2 == 0 else list(reversed(functions))
            for name in order:
                start = time.perf_counter()
                for _ in range(batch):
                    functions[name](matrix)
                times[name].append((time.perf_counter()-start) / batch)
        results.append(dict(nodes=n, block_size=nv, raw_seconds=times,
                            median_seconds={name: float(np.median(t)) for name, t in times.items()},
                            factors_bitwise_equal=True))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
