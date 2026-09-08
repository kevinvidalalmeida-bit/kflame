"""Safety contracts for the isolated cold-start experiments."""
import copy
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'validation'), str(ROOT/'V2'), str(ROOT/'V2/materiales')]
from benchmark_cold_strategies import strict_final_gate, parse_arguments, selected_kernels
from cold_mesh_budget import indicator
from cold_two_grid import restriction_indices, fas_forcing
from cold_strategy_kernels import replace_once
import solver


class ColdStrategyGuardTests(unittest.TestCase):
    def test_default_does_not_build_archived_kernels(self):
        args = parse_arguments([])
        self.assertEqual(args.variants, ['baseline'])
        with patch('benchmark_cold_strategies.build_candidates', side_effect=AssertionError('Archived code built')):
            self.assertEqual(list(selected_kernels(args.variants)), ['baseline'])

    def test_archived_variant_requires_explicit_reproduction(self):
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            parse_arguments(['--variants', 'baseline', 'compiled-lu'])
        self.assertEqual(error.exception.code, 2)
        args = parse_arguments(['--variants', 'baseline', 'compiled-lu', '--reproduce-archived'])
        self.assertTrue(args.reproduce_archived)

    def test_source_replacement_fails_closed(self):
        with self.assertRaises(RuntimeError):
            replace_once('old old', 'old', 'new')
        with self.assertRaises(RuntimeError):
            replace_once('absent', 'old', 'new')
        self.assertEqual(replace_once('old', 'old', 'new'), 'new')

    def test_restriction_preserves_anchor_and_boundaries(self):
        for n in (7, 8, 19, 20):
            for anchor in range(1, n-1):
                keep = restriction_indices(n, anchor)
                self.assertTrue(np.all(np.diff(keep) > 0))
                self.assertIn(anchor, keep)
                self.assertEqual(keep[0], 0)
                self.assertEqual(keep[-1], n-1)

    def test_fas_coarse_equation_preserves_fine_fixed_point(self):
        coarse_f = np.random.default_rng(6).normal(size=55*9)
        tau = fas_forcing(coarse_f, np.zeros_like(coarse_f))
        np.testing.assert_array_equal(coarse_f-tau, np.zeros_like(coarse_f))

    def test_spatial_budget_cannot_relax_an_unmarked_mesh(self):
        problem = SimpleNamespace(n_points=7, n_species=1, solve_energy=True,
                                  j_fixed=3, z=np.linspace(0, 1, 7), steady_rtol=1e-4, steady_atol=1e-9)
        x = np.column_stack([np.ones(7), np.linspace(300, 2200, 7)**1.05, np.ones(7)]).ravel()
        opts = solver.SolveOptions()
        with patch.object(solver, 'build_freeflame_refiner_profiles', return_value={}), \
             patch.object(solver.AdaptiveRefiner, 'refine', return_value=(problem.z, False, 0, 0)):
            budget, changed = indicator(problem, x, opts)
        self.assertEqual(budget, 1.)
        self.assertFalse(changed)

    def test_final_gate_checks_raw_norm_not_just_accepted_flag(self):
        base = dict(accepted=True, stages=[dict(grid_converged=True, weighted_step_norm_final=.5)],
                    Finf=1., species_sum_error=1e-10, mass_flow_error=1.3e-6,
                    Y=np.ones((1, 7)), T=np.full(7, 300.))
        self.assertTrue(strict_final_gate(base))
        for field, value in [('weighted_step_norm_final', 1.01), ('grid_converged', False)]:
            changed = copy.deepcopy(base)
            changed['stages'][-1][field] = value
            self.assertFalse(strict_final_gate(changed))
        for field, value in [('Finf', 1e4+1), ('Finf', np.nan), ('accepted', False)]:
            self.assertFalse(strict_final_gate(dict(base, **{field: value})))


if __name__ == '__main__':
    unittest.main()
