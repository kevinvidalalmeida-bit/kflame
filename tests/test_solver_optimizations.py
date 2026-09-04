from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "V2"
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from equations import BlockTridiagJacobian, factorize, solve_linear
from solver import JacobianState, _equilibrate_block_tridiag, _remember_continuation_linearization


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


class PhysicalLinearScalingTests(unittest.TestCase):
    def test_scaled_block_system_recovers_the_unscaled_newton_correction(self) -> None:
        rng = np.random.default_rng(83)
        n_blocks, block_size = 5, 4
        lower = rng.normal(scale=0.03, size=(n_blocks - 1, block_size, block_size))
        upper = rng.normal(scale=0.03, size=(n_blocks - 1, block_size, block_size))
        diag = rng.normal(scale=0.15, size=(n_blocks, block_size, block_size))
        diag += 2.5 * np.eye(block_size)[None, :, :]
        raw = BlockTridiagJacobian(lower, diag, upper)
        rhs = rng.normal(size=n_blocks * block_size)
        column_scale = np.tile(
            np.array([1.0e-3, 1.0e-1, 1.0e-6, 1.0e-4]), n_blocks
        )

        scaled, row_inverse = _equilibrate_block_tridiag(raw, column_scale)
        state = factorize(scaled)
        state["rhs_multiplier"] = row_inverse
        state["solution_multiplier"] = column_scale

        actual = solve_linear(state, rhs)
        expected = solve_linear(factorize(raw), rhs)
        np.testing.assert_allclose(actual, expected, rtol=2.0e-11, atol=2.0e-11)


class ContinuationLinearizationTests(unittest.TestCase):
    def test_keeps_only_an_exact_stationary_block_lu(self) -> None:
        n_blocks, block_size = 3, 2
        matrix = BlockTridiagJacobian(
            np.zeros((n_blocks - 1, block_size, block_size)),
            np.tile(np.eye(block_size)[None, :, :], (n_blocks, 1, 1)),
            np.zeros((n_blocks - 1, block_size, block_size)),
        )
        lu = factorize(matrix)
        lu["profile_problem"] = object()
        state = JacobianState(
            J=matrix,
            lu=lu,
            age=2,
            n_evals=5,
            last_rdt=0.0,
        )
        problem = SimpleNamespace(n_points=n_blocks, n_species=0)
        source_state = np.linspace(0.0, 1.0, n_blocks * block_size)

        _remember_continuation_linearization(problem, source_state, state)

        handoff = problem._continuation_linearization
        self.assertEqual(handoff["n_points"], n_blocks)
        self.assertEqual(handoff["n_species"], 0)
        self.assertNotIn("profile_problem", handoff["lu"])
        np.testing.assert_allclose(
            solve_linear(handoff["lu"], source_state), source_state
        )

    def test_rejects_a_pseudo_transient_factorization(self) -> None:
        matrix = BlockTridiagJacobian(
            np.zeros((1, 2, 2)),
            np.tile(np.eye(2)[None, :, :], (2, 1, 1)),
            np.zeros((1, 2, 2)),
        )
        problem = SimpleNamespace(n_points=2, n_species=0)
        state = JacobianState(J=matrix, lu=factorize(matrix), last_rdt=10.0)

        _remember_continuation_linearization(problem, np.zeros(4), state)

        self.assertFalse(hasattr(problem, "_continuation_linearization"))


class ShiftedLUReuseTests(unittest.TestCase):
    def _matrix(self, diagonal_shift: float = 0.0) -> BlockTridiagJacobian:
        rng = np.random.default_rng(303)
        n_blocks, block_size = 5, 3
        lower = rng.normal(scale=0.02, size=(n_blocks - 1, block_size, block_size))
        upper = rng.normal(scale=0.02, size=(n_blocks - 1, block_size, block_size))
        diag = rng.normal(scale=0.05, size=(n_blocks, block_size, block_size))
        diag += 2.0 * np.eye(block_size)[None, :, :]
        diag[:, np.arange(block_size), np.arange(block_size)] += diagonal_shift
        return BlockTridiagJacobian(lower, diag, upper)

    def test_shifted_lu_corrections_match_exact_block_solution(self) -> None:
        previous = self._matrix(0.0)
        current = self._matrix(-0.025)
        rng = np.random.default_rng(304)
        rhs = rng.normal(size=current.shape[0])
        state = {
            "method": "shift_reuse_block",
            "matrix": current,
            "reference_lu": factorize(previous),
            "max_corrections": 6,
            "linear_tolerance": 1.0e-11,
        }

        actual = solve_linear(state, rhs)
        expected = solve_linear(factorize(current), rhs)

        self.assertEqual(state["method"], "shift_reuse_block")
        np.testing.assert_allclose(actual, expected, rtol=2.0e-11, atol=2.0e-11)

    def test_shifted_lu_falls_back_to_exact_factorization(self) -> None:
        previous = self._matrix(0.0)
        current = self._matrix(-1.2)
        rng = np.random.default_rng(305)
        rhs = rng.normal(size=current.shape[0])
        state = {
            "method": "shift_reuse_block",
            "matrix": current,
            "reference_lu": factorize(previous),
            "max_corrections": 0,
            "linear_tolerance": 1.0e-14,
        }

        actual = solve_linear(state, rhs)
        expected = solve_linear(factorize(current), rhs)

        self.assertEqual(state["method"], "block_tridiag")
        np.testing.assert_allclose(actual, expected, rtol=2.0e-12, atol=2.0e-12)


if __name__ == "__main__":
    unittest.main()
