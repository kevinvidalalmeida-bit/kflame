from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "FGM" / "scripts"
V2 = ROOT / "V2"
for path in (SCRIPTS, V2):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fgm_common import (
    build_adaptive_c_grid,
    build_global_indicator,
    validate_fgm_table,
)
from generate_fgm_tables_native import _is_local_phi_step
from fgm_continuation import AdaptiveContinuationController, ContinuationConfig
from run_paper_campaign import _pressure_prediction_defect, _pressure_secant_seed


class ContinuationTrustRegionTests(unittest.TestCase):
    def test_accepts_local_multiplicative_steps(self) -> None:
        self.assertTrue(_is_local_phi_step(0.9, 1.0, 1.15))
        self.assertTrue(_is_local_phi_step(1.0, 1.1, 1.15))

    def test_rejects_nonlocal_steps(self) -> None:
        self.assertFalse(_is_local_phi_step(0.7, 0.9, 1.15))
        self.assertFalse(_is_local_phi_step(1.1, 1.4, 1.15))

    def test_nonpositive_ratio_is_explicit_unbounded_ablation(self) -> None:
        self.assertTrue(_is_local_phi_step(0.7, 1.4, 0.0))
        self.assertFalse(_is_local_phi_step(0.0, 1.0, 0.0))


class AdaptivePredictorCorrectorTests(unittest.TestCase):
    def test_uses_log_phi_and_inserts_a_bridge(self) -> None:
        controller = AdaptiveContinuationController(
            ContinuationConfig(initial_ratio=1.15, min_ratio=1.02, max_ratio=1.20)
        )
        proposal = controller.propose(0.7, 0.9)
        self.assertTrue(proposal.is_bridge)
        self.assertAlmostEqual(proposal.phi_trial / 0.7, 1.15, places=12)
        self.assertAlmostEqual(proposal.log_step, np.log(1.15), places=12)

    def test_endpoint_is_preserved_when_inside_step(self) -> None:
        controller = AdaptiveContinuationController()
        proposal = controller.propose(0.9, 1.0)
        self.assertFalse(proposal.is_bridge)
        self.assertEqual(proposal.phi_trial, 1.0)

    def test_secant_defects_calibrate_and_change_next_step(self) -> None:
        controller = AdaptiveContinuationController()
        for defect in (10.0, 12.0, 14.0):
            controller.accept(defect, "secant")
        self.assertEqual(controller.defect_reference, 12.0)
        before = float(controller.log_step)
        after = controller.accept(48.0, "secant")
        self.assertLess(after, before)
        self.assertGreaterEqual(after, controller.config.min_log_step)

    def test_rejection_shrinks_and_exhausts_at_minimum_step(self) -> None:
        controller = AdaptiveContinuationController(
            ContinuationConfig(initial_ratio=1.04, min_ratio=1.02, max_ratio=1.20,
                               max_retries=4)
        )
        step, retry, can_retry = controller.reject()
        self.assertEqual(retry, 1)
        self.assertFalse(can_retry)
        self.assertEqual(step, controller.config.min_log_step)

    def test_four_rejections_trigger_cold_fallback_budget(self) -> None:
        controller = AdaptiveContinuationController(
            ContinuationConfig(
                initial_ratio=1.20,
                min_ratio=1.0001,
                max_ratio=1.20,
                max_retries=4,
            )
        )
        for expected_retry in range(1, 5):
            _, retry, can_retry = controller.reject()
            self.assertEqual(retry, expected_retry)
            self.assertEqual(can_retry, expected_retry < 4)


class PressurePredictorTests(unittest.TestCase):
    @staticmethod
    def _state(offset: float) -> dict[str, np.ndarray]:
        z = np.array([0.0, 0.4, 1.0])
        u = np.array([0.3, 0.4, 0.5]) + offset
        temperature = np.array([300.0, 1100.0, 1800.0]) + 100.0 * offset
        species = np.array([
            [0.8, 0.4, 0.1],
            [0.2, 0.6, 0.9],
        ])
        return {"z": z, "u": u, "T": temperature, "Y": species}

    def test_secant_uses_log_pressure_and_keeps_mass_closure(self) -> None:
        first = self._state(0.0)
        second = self._state(0.1)
        predicted, kind = _pressure_secant_seed(
            previous=second,
            previous_pressure=2.0,
            previous_previous=first,
            previous_previous_pressure=1.0,
            target_pressure=4.0,
        )
        # With a log-pressure ratio of two and damping 0.7, the state moves
        # 70% of the latest secant increment.
        np.testing.assert_allclose(predicted["u"], second["u"] + 0.7 * (second["u"] - first["u"]))
        np.testing.assert_allclose(np.sum(predicted["Y"], axis=0), 1.0)
        self.assertEqual(kind, "secant_log_pressure")

    def test_first_pressure_transition_is_a_copy_and_defect_is_mesh_invariant(self) -> None:
        state = self._state(0.1)
        predicted, kind = _pressure_secant_seed(
            previous=state,
            previous_pressure=1.0,
            previous_previous=None,
            previous_previous_pressure=None,
            target_pressure=1.5,
        )
        corrected = {
            "z": np.linspace(0.0, 1.0, 9),
            "u": np.interp(np.linspace(0.0, 1.0, 9), state["z"], state["u"]),
            "T": np.interp(np.linspace(0.0, 1.0, 9), state["z"], state["T"]),
            "Y": np.vstack([
                np.interp(np.linspace(0.0, 1.0, 9), state["z"], state["Y"][k])
                for k in range(state["Y"].shape[0])
            ]),
        }
        self.assertEqual(kind, "copy")
        self.assertLess(_pressure_prediction_defect(corrected, predicted), 1.0e-10)


class ProgressGridTests(unittest.TestCase):
    def test_indicator_uses_envelope_across_flamelets(self) -> None:
        c = np.linspace(0.0, 1.0, 5)
        first = SimpleNamespace(
            c=c,
            T=np.zeros_like(c),
            qdot=np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
            Y=np.zeros((1, c.size)),
        )
        second = SimpleNamespace(
            c=c,
            T=np.zeros_like(c),
            qdot=np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
            Y=np.zeros((1, c.size)),
        )
        indicator = build_global_indicator(
            [first, second],
            species_names=["X"],
            indicator_species=[],
            c_fine=c,
            w_grad=0.0,
            w_conc=0.0,
            w_temp=0.0,
            w_qdot=1.0,
        )
        self.assertAlmostEqual(float(indicator[1]), 1.0)
        self.assertAlmostEqual(float(indicator[3]), 1.0)

    def test_adaptive_grid_is_strict_and_has_exact_endpoints(self) -> None:
        c_fine = np.linspace(0.0, 1.0, 101)
        indicator = np.exp(-((c_fine - 0.5) / 0.04) ** 2)
        grid = build_adaptive_c_grid(c_fine, indicator, n_c=41, bias=5.0)
        self.assertEqual(grid.size, 41)
        self.assertEqual(float(grid[0]), 0.0)
        self.assertEqual(float(grid[-1]), 1.0)
        self.assertTrue(np.all(np.diff(grid) > 0.0))


class TableValidationTests(unittest.TestCase):
    @staticmethod
    def valid_table() -> dict[str, np.ndarray]:
        nz, ns, nc = 2, 3, 5
        shape = (nz, nc)
        mass_fractions = np.full((nz, ns, nc), 1.0 / ns)
        return {
            "phi_grid": np.array([0.9, 1.1]),
            "Z_grid": np.array([0.05, 0.06]),
            "c_grid": np.linspace(0.0, 1.0, nc),
            "Su": np.array([0.3, 0.4]),
            "T": np.full(shape, 1000.0),
            "u": np.full(shape, 0.4),
            "rho": np.full(shape, 0.5),
            "cp_mass": np.full(shape, 1200.0),
            "conductivity": np.full(shape, 0.1),
            "qdot": np.zeros(shape),
            "omega_c": np.zeros(shape),
            "beta": np.zeros(shape),
            "Y": mass_fractions,
            "solve_ok": np.ones(nz, dtype=bool),
            "final_accepted": np.ones(nz, dtype=bool),
        }

    def test_valid_contract_reports_mass_closure(self) -> None:
        result = validate_fgm_table(self.valid_table())
        self.assertTrue(result["valid"])
        self.assertEqual(result["n_species"], 3)
        self.assertLessEqual(result["max_mass_fraction_sum_error"], 1.0e-15)

    def test_rejects_wrong_species_axis_order(self) -> None:
        table = self.valid_table()
        table["Y"] = np.moveaxis(table["Y"], 1, 2)
        with self.assertRaisesRegex(ValueError, "N_especies"):
            validate_fgm_table(table)

    def test_rejects_unaccepted_flamelet(self) -> None:
        table = self.valid_table()
        table["final_accepted"][1] = False
        with self.assertRaisesRegex(ValueError, "final_accepted"):
            validate_fgm_table(table)


if __name__ == "__main__":
    unittest.main()
