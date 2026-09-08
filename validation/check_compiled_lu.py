"""Local pivoting, singularity, layout and throughput checks for compiled LU."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
from pathlib import Path
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'V2')]
import json
import time
import numpy as np
import equations
from cold_compiled_lu import factorize


def matrix(n, nv, seed=71):
    rng = np.random.default_rng(seed)
    diagonal = rng.normal(size=(n, nv, nv)) + (nv+2)*np.eye(nv)[None]
    # Force actual row pivoting, not just a diagonal-dominant smoke test.
    diagonal[:, [0, -1]] = diagonal[:, [-1, 0]]
    return equations.BlockTridiagJacobian(rng.normal(size=(n-1, nv, nv))*.1,
                                        diagonal, rng.normal(size=(n-1, nv, nv))*.1)


def main():
    report = dict(local=[], timing=[])
    for n, nv in ((1, 2), (4, 5), (8, 12), (9, 55)):
        j = matrix(n, nv)
        baseline = equations.factorize(j)
        candidate = factorize(j, equations.factorize, [])
        for key in ('lu_blocks_array', 'cprime'):
            np.testing.assert_allclose(candidate[key], baseline[key], rtol=2e-12, atol=1e-12)
        np.testing.assert_array_equal(candidate['pivots_array'], baseline['pivots_array'])
        rhs = np.random.default_rng(n).normal(size=n*nv)
        sol = equations.solve_linear(candidate, rhs)
        np.testing.assert_allclose(j.matvec(sol), rhs, rtol=5e-12, atol=5e-12)
        candidate['compiled_substitution'] = False
        np.testing.assert_allclose(equations.solve_linear(candidate, rhs), sol, rtol=2e-12, atol=1e-12)
        report['local'].append(dict(n=n, nv=nv,
            factor_bitwise_equal=np.array_equal(candidate['lu_blocks_array'], baseline['lu_blocks_array']),
            relative_linear_residual=float(np.linalg.norm(j.matvec(sol)-rhs)/np.linalg.norm(rhs))))
    singular = equations.BlockTridiagJacobian(np.zeros((0, 2, 2)), np.zeros((1, 2, 2)), np.zeros((0, 2, 2)))
    try:
        factorize(singular, equations.factorize, [])
        raise AssertionError('Singular factorization accepted')
    except RuntimeError:
        report['singularity_rejected'] = True
    for n in (261, 854):
        j = matrix(n, 55)
        values = {'baseline': [], 'compiled': []}
        for repetition in range(9):
            for name in (list(values) if repetition % 2 == 0 else list(values)[::-1]):
                started = time.perf_counter()
                value = equations.factorize(j) if name == 'baseline' else factorize(j, equations.factorize, [])
                values[name].append(time.perf_counter()-started)
        report['timing'].append(dict(n=n, raw_s=values,
            median_s={key: float(np.median(val)) for key, val in values.items()}))
    Path(__file__).with_name('compiled_lu_local_checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
