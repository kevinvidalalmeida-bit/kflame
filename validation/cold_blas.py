"""MKL LAPACKE substitution ONLY for V2 block GETRF/GETRS.

NumPy matmul and Soret's Numba linalg retain their existing providers.
LP64 C API, column-major storage, SciPy zero-based pivot contract.
No global installation or replacement of SciPy binaries is performed.
"""
from contextlib import contextmanager
import ctypes as c
import hashlib
import os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import equations


class MKLBlocks:
    def __init__(self, directory=None):
        self.directory = Path(directory or Path(__file__).resolve().parents[1] /
                              'resultados/blas_env_20260907/Library/bin').resolve()
        self.handle = os.add_dll_directory(str(self.directory))
        os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
        self.dll = c.CDLL(str(self.directory/'mkl_rt.2.dll'))
        self.dll.MKL_Set_Num_Threads.argtypes = [c.c_int]
        self.dll.MKL_Set_Num_Threads(1)
        self.dll.MKL_Set_Dynamic.argtypes = [c.c_int]
        self.dll.MKL_Set_Dynamic(0)
        self.rf = self.dll.LAPACKE_dgetrf
        self.rf.argtypes = [c.c_int, c.c_int, c.c_int, c.c_void_p, c.c_int, c.c_void_p]
        self.rf.restype = c.c_int
        self.rs = self.dll.LAPACKE_dgetrs
        self.rs.argtypes = [c.c_int, c.c_char, c.c_int, c.c_int, c.c_void_p,
                           c.c_int, c.c_void_p, c.c_void_p, c.c_int]
        self.rs.restype = c.c_int
        self.calls = dict(getrf=0, getrs=0)

    def getrf(self, matrix, overwrite_a=False):
        a = np.array(matrix, dtype=np.float64, order='F', copy=True)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError('Only square V2 blocks supported')
        n = len(a)
        piv = np.empty(n, dtype=np.int32)
        info = self.rf(102, n, n, a.ctypes.data, n, piv.ctypes.data)
        self.calls['getrf'] += 1
        return a, piv - 1, info

    def getrs(self, factors, pivots, rhs, trans=0, overwrite_b=False):
        a = np.asarray(factors, dtype=np.float64, order='F')
        b = np.array(rhs, dtype=np.float64, order='F', copy=True)
        n = len(a)
        if b.ndim not in (1, 2) or b.shape[0] != n:
            raise ValueError('RHS incompatible with block')
        piv = np.asarray(pivots, dtype=np.int32) + 1
        info = self.rs(102, [b'N', b'T', b'C'][trans], n,
                       1 if b.ndim == 1 else b.shape[1], a.ctypes.data,
                       n, piv.ctypes.data, b.ctypes.data, n)
        self.calls['getrs'] += 1
        return b, info


@contextmanager
def mkl_context(records):
    provider = MKLBlocks()
    with patch.object(equations, '_block_getrf', provider.getrf), \
         patch.object(equations, '_block_getrs', provider.getrs):
        try:
            yield
        finally:
            records.append(dict(kind='mkl_blocks', dll=str(provider.directory/'mkl_rt.2.dll'),
                                dll_sha256=hashlib.sha256((provider.directory/'mkl_rt.2.dll').read_bytes()).hexdigest(),
                                scope='block GETRF/GETRS only; other BLAS unchanged',
                                **provider.calls))
