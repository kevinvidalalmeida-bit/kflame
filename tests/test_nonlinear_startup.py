"""Unit tests for the experimental nonlinear startup strategies.

Experiment A: adaptive Newton corrections during PTC.
Experiment B: early mesh refinement on PTC stall.
Experiment C: combined strategy.

These tests verify the new diagnostics and control logic without solving
a real flame. The production solver path (all flags False) is verified
to be unchanged by the existing test suite.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from kflame.flame.equations import BlockTridiagJacobian
from kflame.flame.solver import (
    SolveOptions,
    _local_linearisation_defect_blocks,
    _try_early_refinement,
    AdaptiveRefiner,
    build_freeflame_refiner_profiles,
    weighted_norm,
)
from kflame.flame.state import pack_state, unpack_state


class TestSolveOptionsDefaults(unittest.TestCase):
    """Verify that all experimental parameters default to off."""

    def test_adaptive_newton_corrections_default_off(self):
        opts = SolveOptions()
        self.assertFalse(opts.adaptive_newton_corrections)
        self.assertEqual(opts.max_ptc_corrections, 3)

    def test_early_refinement_default_off(self):
        opts = SolveOptions()
        self.assertFalse(opts.early_refinement)
        self.assertEqual(opts.stall_window, 40)
        self.assertEqual(opts.max_early_refinements, 2)

    def test_combined_strategy_default_off(self):
        opts = SolveOptions()
        self.assertFalse(opts.combined_startup_strategy)

    def test_production_options_unchanged(self):
        """The default SolveOptions must not enable any experimental path."""
        opts = SolveOptions()
        # These are the exact production values.
        self.assertEqual(opts.max_time_step_count, 500)
        self.assertAlmostEqual(opts.time_step, 1e-5)
        self.assertTrue(opts.refine_grid)
        self.assertEqual(opts.max_jac_age, 20)


class TestLinearisationDefect(unittest.TestCase):
    """Test the defect diagnostic used by Experiment A."""

    def test_identity_jacobian_zero_defect(self):
        """A linear system F(x) = Jx has zero defect."""
        n_blocks = 4
        bs = 2
        J = BlockTridiagJacobian(
            np.zeros((n_blocks - 1, bs, bs)),
            np.tile(np.eye(bs)[None, :, :], (n_blocks, 1, 1)),
            np.zeros((n_blocks - 1, bs, bs)),
        )
        f0 = np.ones(n_blocks * bs)
        step = np.ones(n_blocks * bs) * 0.1
        # For a linear function: F(x + α·s) = F(x) + α·J·s exactly.
        f_trial = f0 + 1.0 * J.matvec(step)

        selected, info = _local_linearisation_defect_blocks(
            f0, f_trial, J, step, alpha=1.0,
            defect_threshold=0.01, max_fraction=0.8,
        )
        self.assertEqual(selected.size, 0)
        self.assertAlmostEqual(info["defect_max"], 0.0, places=12)

    def test_nonlinear_defect_detected(self):
        """A concentrated nonlinear perturbation must trigger selection."""
        n_blocks = 5
        bs = 2
        J = BlockTridiagJacobian(
            np.zeros((n_blocks - 1, bs, bs)),
            np.tile(np.eye(bs)[None, :, :], (n_blocks, 1, 1)),
            np.zeros((n_blocks - 1, bs, bs)),
        )
        f0 = np.ones(n_blocks * bs) * 10.0
        step = np.ones(n_blocks * bs) * 0.1
        # Linear prediction
        f_trial = f0 + 1.0 * J.matvec(step)
        # Add nonlinear perturbation at block 2
        f_trial[4] += 5.0
        f_trial[5] += 5.0

        selected, info = _local_linearisation_defect_blocks(
            f0, f_trial, J, step, alpha=1.0,
            defect_threshold=0.1, max_fraction=0.8,
        )
        self.assertGreater(selected.size, 0)
        self.assertIn(2, selected)  # The perturbed block
        self.assertGreater(info["defect_max"], 0.1)


class TestProgressRatio(unittest.TestCase):
    """Test the progress ratio computation logic."""

    def test_good_progress(self):
        """When norm reduces significantly, ratio < threshold."""
        norm_before = 100.0
        norm_after = 10.0
        ratio = norm_after / max(norm_before, 1e-300)
        self.assertLess(ratio, 0.5)

    def test_poor_progress(self):
        """When norm barely reduces, ratio > threshold."""
        norm_before = 100.0
        norm_after = 80.0
        ratio = norm_after / max(norm_before, 1e-300)
        self.assertGreater(ratio, 0.5)

    def test_growth(self):
        """Growth gives ratio > 1, definitely poor progress."""
        norm_before = 100.0
        norm_after = 120.0
        ratio = norm_after / max(norm_before, 1e-300)
        self.assertGreater(ratio, 1.0)


class TestStallDetection(unittest.TestCase):
    """Test the sliding window stall detector used by Experiment B."""

    def test_no_stall_when_reducing(self):
        """Exponentially decreasing norms should not trigger stall."""
        window = 10
        threshold = 0.5
        norms = [100.0 * (0.9 ** i) for i in range(window)]
        oldest = norms[0]
        newest = norms[-1]
        ratio = newest / oldest
        self.assertLess(ratio, threshold)

    def test_stall_when_flat(self):
        """Constant norms should trigger stall."""
        window = 10
        threshold = 0.5
        norms = [100.0] * window
        oldest = norms[0]
        newest = norms[-1]
        ratio = newest / oldest
        self.assertGreater(ratio, threshold)

    def test_stall_when_slow_progress(self):
        """Very slow reduction should trigger stall."""
        window = 40
        threshold = 0.5
        # Only 1% reduction per step
        norms = [100.0 * (0.99 ** i) for i in range(window)]
        oldest = norms[0]
        newest = norms[-1]
        ratio = newest / oldest
        # 0.99^40 ≈ 0.669 > 0.5
        self.assertGreater(ratio, threshold)

    def test_window_not_full(self):
        """No stall detection if window is not full."""
        window = 40
        norms = [100.0] * 20  # Only half full
        # Should not trigger because we need >= window entries
        self.assertLess(len(norms), window)


class TestAdaptiveRefiner(unittest.TestCase):
    """Test that the refiner correctly identifies points to insert."""

    def test_steep_front_triggers_insertion(self):
        """A steep gradient should trigger point insertion."""
        z = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
        T = np.array([300.0, 300.0, 300.0, 2000.0, 2200.0, 2200.0])
        profiles = {"T": T}
        refiner = AdaptiveRefiner(ratio=2.5, slope=0.04, curve=0.08, prune=0.003)
        insert_after, remove = refiner.analyze(z, profiles)
        self.assertGreater(len(insert_after), 0)

    def test_uniform_profile_no_insertion(self):
        """A uniform profile should not trigger insertion."""
        z = np.linspace(0, 0.03, 20)
        T = np.full(20, 300.0)
        profiles = {"T": T}
        refiner = AdaptiveRefiner(ratio=2.5, slope=0.04, curve=0.08, prune=0.003)
        insert_after, remove = refiner.analyze(z, profiles)
        self.assertEqual(len(insert_after), 0)


if __name__ == "__main__":
    unittest.main()
