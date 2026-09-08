import sys
from pathlib import Path
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'V2'), str(ROOT/'V2/materiales'), str(ROOT/'validation')]
from cold_blas import MKLBlocks
from benchmark_cold_strategies import parse_arguments, selected_kernels


class CandidateTests(unittest.TestCase):
    def test_opt_in(self):
        self.assertEqual(parse_arguments([]).variants, ['baseline'])
        self.assertEqual(list(selected_kernels(['baseline', 'mkl-blocks'])), ['baseline'])

    @unittest.skipUnless((ROOT/'resultados/blas_env_20260907/Library/bin/mkl_rt.2.dll').exists(),
                         'isolated MKL runtime not installed')
    def test_mkl_pivots_and_rhs(self):
        mkl = MKLBlocks()
        rng = np.random.default_rng(72)
        for n in (3, 12, 55):
            a = rng.normal(size=(n,n))
            a[0,0] = 0.
            original = a.copy()
            lu, piv, info = mkl.getrf(a)
            self.assertEqual(info, 0)
            self.assertTrue(np.all((piv >= 0) & (piv < n)))
            for rhs in (rng.normal(size=n), rng.normal(size=(n,4))):
                for trans in (0,1):
                    x, info = mkl.getrs(lu,piv,rhs,trans=trans)
                    self.assertEqual(info, 0)
                    np.testing.assert_allclose((a if trans==0 else a.T)@x,rhs,rtol=1e-11,atol=1e-11)
            np.testing.assert_array_equal(a,original)
        _, _, info = mkl.getrf(np.zeros((3,3)))
        self.assertGreater(info,0)


if __name__ == '__main__':
    unittest.main()
