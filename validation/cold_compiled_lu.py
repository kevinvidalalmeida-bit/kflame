"""Experimental compiled block-LU traversal using the public SciPy C API.

No cached raw pointers: Cython addresses are resolved in each process. The
LP64 signatures are validated before using them; other ABIs fail closed.
"""
from contextlib import contextmanager
import ctypes
from functools import lru_cache
from unittest.mock import patch
import numpy as np
from numba import njit
from numba.extending import get_cython_function_address
import scipy.linalg.cython_lapack as lapack
import equations
import solver


@lru_cache(maxsize=1)
def compiled_factorizer():
    signatures = {
        'dgetrf': 'void (int *, int *, __pyx_t_5scipy_6linalg_13cython_lapack_d *, int *, int *, int *)',
        'dgetrs': 'void (char *, int *, int *, __pyx_t_5scipy_6linalg_13cython_lapack_d *, int *, int *, __pyx_t_5scipy_6linalg_13cython_lapack_d *, int *, int *)',
    }
    capsule_name = ctypes.pythonapi.PyCapsule_GetName
    capsule_name.restype = ctypes.c_char_p
    capsule_name.argtypes = [ctypes.py_object]
    functions = {}
    for name, signature in signatures.items():
        if capsule_name(lapack.__pyx_capi__[name]).decode() != signature:
            raise RuntimeError('Unsupported Cython LAPACK ABI: '+name)
        count = 6 if name == 'dgetrf' else 9
        functions[name] = ctypes.CFUNCTYPE(None, *([ctypes.c_void_p]*count))(
            get_cython_function_address('scipy.linalg.cython_lapack', name))
    getrf, getrs = functions['dgetrf'], functions['dgetrs']

    @njit(cache=False)
    def core(lower, diagonal, upper):
        n, nv = diagonal.shape[:2]
        factors = np.empty((n, nv, nv))
        pivots = np.empty((n, nv), dtype=np.int32)
        cprime = np.zeros_like(upper)
        dimension = np.array([nv], dtype=np.int32)
        info = np.zeros(1, dtype=np.int32)
        trans = np.array([78], dtype=np.uint8)  # ASCII N, no transposition
        work_pivots = np.empty(nv, dtype=np.int32)
        for i in range(n):
            mat = diagonal[i].copy()
            if i > 0:
                mat -= lower[i-1] @ cprime[i-1]
            # A C-contiguous transpose stores the original matrix in Fortran
            # layout. Keep the existing C-order Schur update before this copy.
            mat_f = mat.T.copy()
            getrf(dimension.ctypes, dimension.ctypes, mat_f.ctypes,
                  dimension.ctypes, work_pivots.ctypes, info.ctypes)
            if info[0] != 0:
                return factors, pivots, cprime, info[0], i
            factors[i] = mat_f.T
            pivots[i] = work_pivots-1  # SciPy/Python convention is zero-based.
            if i < n-1:
                rhs_f = upper[i].T.copy()
                getrs(trans.ctypes, dimension.ctypes, dimension.ctypes,
                      mat_f.ctypes, dimension.ctypes, work_pivots.ctypes,
                      rhs_f.ctypes, dimension.ctypes, info.ctypes)
                if info[0] != 0:
                    return factors, pivots, cprime, info[0], i
                cprime[i] = rhs_f.T
        return factors, pivots, cprime, 0, -1
    return core


def factorize(jmat, original, records):
    if not isinstance(jmat, equations.BlockTridiagJacobian):
        return original(jmat)
    n, nv = jmat.n_blocks, jmat.block_size
    if n < 1 or nv < 1 or nv > np.iinfo(np.int32).max:
        raise ValueError('Invalid block dimensions for LP64 factorization')
    if jmat.diag.shape != (n, nv, nv) or any(
            a.shape != (n-1, nv, nv) for a in (jmat.lower, jmat.upper)):
        raise ValueError('Inconsistent block shapes')
    problem = getattr(jmat, '_profile_problem', None)
    start = equations._profile_start(problem)
    lower, diagonal, upper = (np.ascontiguousarray(a, dtype=np.float64)
                              for a in (jmat.lower, jmat.diag, jmat.upper))
    factors, pivots, cprime, info, block = compiled_factorizer()(lower, diagonal, upper)
    if info:
        raise RuntimeError(f'Block {block}: compiled LAPACK failed with info={info}')
    records.append(dict(kind='compiled_lu', nodes=n, block_size=nv))
    out = dict(method='block_tridiag', solver='compiled_block_thomas',
        lu_blocks=[(factors[i], pivots[i]) for i in range(n)],
        lu_blocks_array=factors, pivots_array=pivots, lower_blocks=lower,
        cprime=cprime, n_blocks=n, block_size=nv,
        compiled_substitution=bool(getattr(jmat, 'use_compiled_substitution', True)))
    if problem is not None:
        equations._profile_record(problem, 'linear_factorize', start)
    return out


@contextmanager
def compiled_lu_context(records):
    original = equations.factorize
    def candidate(jmat):
        try:
            return factorize(jmat, original, records)
        except Exception as exc:
            records.append(dict(kind='compiled_lu', exception=repr(exc)))
            raise
    with patch.object(equations, 'factorize', candidate), patch.object(solver, 'factorize', candidate):
        yield
