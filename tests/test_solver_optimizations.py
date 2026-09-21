from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

import kflame.flame.equations as equations
from kflame.flame.equations import BlockTridiagJacobian, factorize, refresh_block_tridiag_jacobian_columns, solve_linear
from kflame.flame.solver import (
    _local_linearisation_defect_blocks,
)


class BlockDiagonalUpdateTests(unittest.TestCase):
    def test_singular_block_is_reported_before_substitution(self) -> None:
        matrix = BlockTridiagJacobian(np.empty((0, 2, 2)),
                                      np.zeros((1, 2, 2)), np.empty((0, 2, 2)))
        with self.assertRaisesRegex(RuntimeError, 'Block 0: LAPACK dgetrf'):
            factorize(matrix)

    def test_low_level_lapack_failure_is_not_ignored(self) -> None:
        matrix = BlockTridiagJacobian(np.zeros((1, 2, 2)),
                                      np.tile(np.eye(2), (2, 1, 1)), np.zeros((1, 2, 2)))
        with patch.object(equations, '_block_getrs', return_value=(np.zeros((2, 2)), -1)):
            with self.assertRaisesRegex(RuntimeError, 'dgetrs failed'):
                factorize(matrix)

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


class LocalJacobianRefreshTests(unittest.TestCase):
    def test_defect_selects_local_stencil_and_rejects_global_selection(self) -> None:
        matrix = BlockTridiagJacobian(
            np.zeros((4, 2, 2)),
            np.tile(np.eye(2)[None, :, :], (5, 1, 1)),
            np.zeros((4, 2, 2)),
        )
        f0 = np.ones(10)
        step = np.zeros(10)
        f_trial = f0.copy()
        # The nonlinear defect is localized at node 2, so columns 1--3 must
        # be refreshed to cover the tridiagonal residual stencil.
        f_trial[4] += 1.0
        selected, info = _local_linearisation_defect_blocks(
            f0, f_trial, matrix, step, 1.0,
            defect_threshold=0.5, max_fraction=0.8,
        )
        np.testing.assert_array_equal(selected, np.array([1, 2, 3]))
        self.assertEqual(info["defect_selected_blocks"], 3.0)

        rejected, rejected_info = _local_linearisation_defect_blocks(
            f0, f_trial, matrix, step, 1.0,
            defect_threshold=0.5, max_fraction=0.4,
        )
        self.assertEqual(rejected.size, 0)
        self.assertEqual(rejected_info["defect_rejected_global"], 1.0)

    def test_refresh_replaces_only_requested_column_block(self) -> None:
        n_blocks, block_size = 3, 2
        reference = BlockTridiagJacobian(
            np.full((2, 2, 2), -3.0),
            np.full((3, 2, 2), -2.0),
            np.full((2, 2, 2), -1.0),
        )
        before = reference.copy()
        problem = SimpleNamespace(
            n_points=n_blocks,
            n_species=0,
            jacobian_rel_perturb=1.0e-5,
            jacobian_abs_perturb=1.0e-10,
            jacobian_threshold=0.0,
            _profile=None,
        )
        target_values = np.array(
            [[10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
             [20.0, 21.0, 22.0, 23.0, 24.0, 25.0]]
        )

        def fake_rows(_x, _problem, _j, _cols, xpert, **_kwargs):
            # x is zero, so xpert contains the finite-difference increments.
            batch = np.vstack((
                target_values[0] * xpert[0],
                target_values[1] * xpert[1],
            ))
            return np.arange(6, dtype=np.int32), batch

        with patch.object(equations, "build_local_jacobian_cache", return_value={}), patch.object(
            equations, "residual_local_rows_batch_perturbed", side_effect=fake_rows
        ), patch.object(equations, "_fill_block_tridiag_jacobian_block_numba", None):
            refreshed = refresh_block_tridiag_jacobian_columns(
                lambda x, _problem: np.zeros_like(x),
                np.zeros(6),
                problem,
                reference,
                [1],
            )

        # Non-adjacent column blocks are untouched.
        np.testing.assert_allclose(refreshed.diag[0], before.diag[0])
        np.testing.assert_allclose(refreshed.diag[2], before.diag[2])
        np.testing.assert_allclose(refreshed.lower[0], before.lower[0])
        np.testing.assert_allclose(refreshed.upper[1], before.upper[1])
        # Column block 1 owns diag[1], upper[0], and lower[1].
        np.testing.assert_allclose(
            refreshed.upper[0], np.array([[10.0, 20.0], [11.0, 21.0]])
        )
        np.testing.assert_allclose(
            refreshed.diag[1], np.array([[12.0, 22.0], [13.0, 23.0]])
        )
        np.testing.assert_allclose(
            refreshed.lower[1], np.array([[14.0, 24.0], [15.0, 25.0]])
        )


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
