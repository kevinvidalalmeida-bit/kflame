from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "V2"
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from equations import BlockTridiagJacobian, factorize, solve_linear


class BlockDiagonalUpdateTests(unittest.TestCase):
    def test_vectorized_diagonal_round_trip_preserves_off_diagonals(self) -> None:
        rng = np.random.default_rng(42)
        diag_blocks = rng.normal(size=(4, 3, 3))
        lower = rng.normal(size=(3, 3, 3))
        upper = rng.normal(size=(3, 3, 3))
        matrix = BlockTridiagJacobian(lower, diag_blocks.copy(), upper)

        off_diagonal_before = matrix.diag.copy()
        for j in range(4):
            np.fill_diagonal(off_diagonal_before[j], 0.0)

        replacement = np.arange(12, dtype=float) + 100.0
        matrix.setdiag(replacement)

        np.testing.assert_array_equal(matrix.diagonal(), replacement)
        off_diagonal_after = matrix.diag.copy()
        for j in range(4):
            np.fill_diagonal(off_diagonal_after[j], 0.0)
        np.testing.assert_allclose(off_diagonal_after, off_diagonal_before)


class CompiledBlockSubstitutionTests(unittest.TestCase):
    def test_matches_dense_solution_with_pivoting(self) -> None:
        rng = np.random.default_rng(7)
        n_blocks = 4
        block_size = 3
        lower = rng.normal(scale=0.04, size=(n_blocks - 1, block_size, block_size))
        upper = rng.normal(scale=0.04, size=(n_blocks - 1, block_size, block_size))
        diag = rng.normal(scale=0.2, size=(n_blocks, block_size, block_size))
        diag += 2.0 * np.eye(block_size)[None, :, :]
        # Force a non-trivial pivot in one block while preserving conditioning.
        diag[1, 0, 0] = 1.0e-8
        diag[1, 2, 0] = 1.5

        jacobian = BlockTridiagJacobian(lower, diag, upper)
        state = factorize(jacobian)
        rhs = rng.normal(size=n_blocks * block_size)
        actual = solve_linear(state, rhs)

        fallback_jacobian = BlockTridiagJacobian(lower, diag, upper)
        fallback_jacobian.use_compiled_substitution = False
        fallback = solve_linear(factorize(fallback_jacobian), rhs)

        dense = np.zeros((rhs.size, rhs.size), dtype=float)
        for i in range(n_blocks):
            row = slice(i * block_size, (i + 1) * block_size)
            dense[row, row] = diag[i]
            if i < n_blocks - 1:
                nxt = slice((i + 1) * block_size, (i + 2) * block_size)
                dense[row, nxt] = upper[i]
                dense[nxt, row] = lower[i]
        expected = np.linalg.solve(dense, rhs)

        np.testing.assert_allclose(actual, expected, rtol=2.0e-12, atol=2.0e-12)
        np.testing.assert_allclose(actual, fallback, rtol=2.0e-12, atol=2.0e-12)


if __name__ == "__main__":
    unittest.main()
