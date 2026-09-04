from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
from generate_fgm_tables_native import (
    bound_continuation_seed_mesh,
    _front_aligned_secant_state,
    _is_local_phi_step,
    _jacobian_tangent_state,
    build_continuation_seed,
)
from state import pack_state, unpack_state
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


class ThermalFramePredictorTests(unittest.TestCase):
    @staticmethod
    def _profile(front: float, thickness: float) -> tuple[np.ndarray, np.ndarray]:
        z = np.linspace(0.0, 0.03, 1201)
        xi = (z - front) / thickness
        progress = 1.0 / (1.0 + np.exp(-xi))
        u = 0.25 + 0.30 * progress
        temperature = 300.0 + 1500.0 * progress
        species = np.vstack((0.80 - 0.70 * progress, 0.20 + 0.70 * progress))
        return z, pack_state(u, temperature, species)

    def test_front_aligned_secant_predicts_front_translation_and_thickness(self) -> None:
        z_old, x_old = self._profile(0.011, 0.0020)
        z_latest, x_latest = self._profile(0.013, 0.0018)
        _z_target, x_target = self._profile(0.015, 0.0016)

        result = _front_aligned_secant_state(
            x_old=x_old,
            z_old=z_old,
            x_latest=x_latest,
            z_latest=z_latest,
            n_species=2,
            factor=1.0,
        )
        self.assertIsNotNone(result)
        x_predicted, metadata = result
        _u_pred, T_pred, Y_pred = unpack_state(x_predicted, z_latest.size, 2)
        _u_target, T_target, _Y_target = unpack_state(x_target, z_latest.size, 2)

        self.assertGreater(metadata["front_prediction_m"], metadata["front_latest_m"])
        self.assertLess(metadata["thickness_prediction_m"], metadata["thickness_latest_m"])
        # The inlet boundary is overwritten by the physical fresh mixture in
        # the production seed. Check the transported thermal layer itself,
        # rather than flat extrapolation outside the source-frame overlap.
        active_layer = (T_target > 310.0) & (T_target < 1790.0)
        self.assertLess(float(np.max(np.abs(T_pred[active_layer] - T_target[active_layer]))), 1.0)
        np.testing.assert_allclose(np.sum(Y_pred, axis=0), 1.0, rtol=0.0, atol=1.0e-14)

    def test_inlet_projection_updates_the_unburned_mixture_not_only_one_node(self) -> None:
        z, x = self._profile(0.013, 0.0018)
        problem = SimpleNamespace(
            n_species=2,
            T_lower_bound=200.0,
            T_upper_bound=6000.0,
            T_in=300.0,
            Y_in=np.array([0.70, 0.30]),
        )
        seed = build_continuation_seed(
            problem=problem,
            phi=1.02,
            prev_solution={"phi": np.array([1.0]), "z": z, "x": x},
            prev_prev_solution=None,
            use_predictor=False,
            predictor_damping=0.7,
            predictor_frame="z",
            trust_ratio=1.15,
        )
        self.assertIsNotNone(seed)
        _u, temperature, species = unpack_state(seed["x"], z.size, 2)
        self.assertEqual(seed["predictor_kind"], "copy_fresh_projected")
        np.testing.assert_allclose(species[:, 0], problem.Y_in, rtol=0.0, atol=1.0e-14)
        # The next fresh/preheat point receives the same physical mixture
        # projection, instead of retaining the previous phi composition.
        self.assertLess(float(temperature[1]), 302.0)
        np.testing.assert_allclose(species[:, 1], problem.Y_in, rtol=0.0, atol=2.0e-3)

    def test_bounded_seed_mesh_preserves_endpoints_and_mass_closure(self) -> None:
        z, x = self._profile(0.013, 0.0018)
        transferred = bound_continuation_seed_mesh(
            {"z": z, "x": x, "predictor_kind": "copy"},
            n_species=2,
            max_points=61,
        )
        self.assertEqual(transferred["z"].size, 61)
        self.assertEqual(float(transferred["z"][0]), float(z[0]))
        self.assertEqual(float(transferred["z"][-1]), float(z[-1]))
        _u, _T, species = unpack_state(transferred["x"], 61, 2)
        np.testing.assert_allclose(np.sum(species, axis=0), 1.0, rtol=0.0, atol=1.0e-14)


class JacobianTangentChordTests(unittest.TestCase):
    def test_tangent_uses_the_parameter_chord_not_the_target_residual(self) -> None:
        """A nonzero accepted source residual must cancel from dF/dlambda."""
        source_state = np.array([0.30, 300.0, 0.35, 900.0, 0.40, 1800.0])
        source_residual = np.array([9.0, -4.0, 2.0, 7.0, -3.0, 5.0])
        target_residual = source_residual + np.array([0.5, -0.7, 0.1, 0.4, 0.2, -0.3])
        delta_lambda = 0.1
        problem = SimpleNamespace(
            n_species=0,
            z=np.array([0.0, 0.5, 1.0]),
            n_points=3,
            width=1.0,
            solve_energy=True,
            anchor_T=1000.0,
            backend_factory=lambda _problem: object(),
        )
        previous = {
            "phi": np.array([1.0]),
            "z": problem.z.copy(),
            "x": source_state.copy(),
            "continuation_linearization": {
                "lu": {"method": "block_tridiag"},
                "n_points": 3,
                "n_species": 0,
                "anchor_index": 1,
                "state": source_state.copy(),
                "residual": source_residual.copy(),
            },
        }
        observed: dict[str, np.ndarray] = {}

        def fake_solve(_lu, rhs):
            observed["rhs"] = np.asarray(rhs, dtype=float).copy()
            return np.zeros_like(rhs, dtype=float)

        with patch(
            "generate_fgm_tables_native.residual",
            return_value=target_residual,
        ), patch(
            "generate_fgm_tables_native.solve_linear",
            side_effect=fake_solve,
        ):
            result = _jacobian_tangent_state(
                problem=problem,
                target_phi=float(np.exp(delta_lambda)),
                previous=previous,
                damping=1.0,
            )

        self.assertIsNotNone(result)
        predicted, metadata = result
        np.testing.assert_allclose(predicted, source_state)
        np.testing.assert_allclose(
            observed["rhs"],
            -(target_residual - source_residual) / delta_lambda,
        )
        self.assertAlmostEqual(metadata["source_residual_inf"], 9.0)
        self.assertAlmostEqual(
            metadata["parameter_chord_inf"], 0.7,
        )

    def test_tangent_refuses_an_unpaired_source_residual(self) -> None:
        source_state = np.array([0.30, 300.0, 0.35, 900.0, 0.40, 1800.0])
        problem = SimpleNamespace(
            n_species=0,
            z=np.array([0.0, 0.5, 1.0]),
            n_points=3,
            width=1.0,
            solve_energy=True,
            anchor_T=1000.0,
            backend_factory=lambda _problem: object(),
        )
        previous = {
            "phi": np.array([1.0]),
            "z": problem.z.copy(),
            "x": source_state.copy(),
            "continuation_linearization": {
                "lu": {"method": "block_tridiag"},
                "n_points": 3,
                "n_species": 0,
                "anchor_index": 1,
                "state": source_state.copy(),
            },
        }
        self.assertIsNone(
            _jacobian_tangent_state(
                problem=problem,
                target_phi=1.02,
                previous=previous,
                damping=1.0,
            )
        )

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
