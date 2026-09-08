"""Small regression contracts for the isolated compiled LAPACK experiment."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'validation'), str(ROOT/'V2')]
import equations
import solver
from cold_compiled_lu import compiled_lu_context, factorize
from check_compiled_lu import matrix


class CompiledLUGuards(unittest.TestCase):
    def test_pivots_solution_and_input_preserved(self):
        j = matrix(4, 5)
        saved = [a.copy() for a in (j.lower, j.diag, j.upper)]
        baseline = equations.factorize(j)
        candidate = factorize(j, equations.factorize, [])
        np.testing.assert_array_equal(candidate['pivots_array'], baseline['pivots_array'])
        rhs = np.arange(20, dtype=float)
        x = equations.solve_linear(candidate, rhs)
        np.testing.assert_allclose(j.matvec(x), rhs, atol=1e-12, rtol=1e-12)
        for before, after in zip(saved, (j.lower, j.diag, j.upper)):
            np.testing.assert_array_equal(before, after)

    def test_singularity_raises_and_context_restores(self):
        j = equations.BlockTridiagJacobian(np.zeros((0, 2, 2)),
            np.zeros((1, 2, 2)), np.zeros((0, 2, 2)))
        original, solver_original = equations.factorize, solver.factorize
        trace = []
        with self.assertRaises(RuntimeError):
            with compiled_lu_context(trace):
                solver.factorize(j)
        self.assertIs(equations.factorize, original)
        self.assertIs(solver.factorize, solver_original)
        self.assertIn('exception', trace[-1])

    def test_nonblock_uses_original(self):
        marker = object()
        self.assertIs(factorize(marker, lambda value: value, []), marker)


if __name__ == '__main__':
    unittest.main()
