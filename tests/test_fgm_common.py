from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

from kava.fgm.common import (
    build_adaptive_c_grid,
    build_global_indicator,
    validate_fgm_table,
)
from kava.fgm.generate import (
    bound_continuation_seed_mesh,
    build_argparser,
    _is_local_phi_step,
    build_continuation_seed,
)
from kava.flame.state import pack_state, unpack_state


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


class LocalJacobianRefreshPolicyTests(unittest.TestCase):
    def test_fgm_continuation_enables_the_certified_rescue_by_default(self) -> None:
        parser = build_argparser()
        self.assertTrue(parser.parse_args([]).local_jacobian_refresh)
        self.assertFalse(
            parser.parse_args(["--no-local-jacobian-refresh"]).local_jacobian_refresh
        )


class ContinuationSeedTests(unittest.TestCase):
    @staticmethod
    def _profile(front: float, thickness: float) -> tuple[np.ndarray, np.ndarray]:
        z = np.linspace(0.0, 0.03, 1201)
        xi = (z - front) / thickness
        progress = 1.0 / (1.0 + np.exp(-xi))
        u = 0.25 + 0.30 * progress
        temperature = 300.0 + 1500.0 * progress
        species = np.vstack((0.80 - 0.70 * progress, 0.20 + 0.70 * progress))
        return z, pack_state(u, temperature, species)

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
            predictor_damping=0.7,
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
