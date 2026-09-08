"""Contracts for the two explicitly authorized candidates, without flame timing."""
from pathlib import Path
import inspect
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'validation'), str(ROOT/'V2'), str(ROOT/'V2/materiales')]
import solver
import species_backend_native as backend
from cold_exact_residual import exact_residual_context
from cold_generated_chemistry import generate
from benchmark_cold_strategies import parse_arguments, selected_kernels


class NewCandidatesTests(unittest.TestCase):
    def run_acceptance(self, lag, values):
        problem = SimpleNamespace(n_points=2,
            backend=SimpleNamespace(uses_multicomponent_flux=lag))
        options = solver.SolveOptions(verbose=False, lag_multicomponent_transport=lag,
                                      final_Finf_limit=1e4)
        seen, trace = [], []
        def remember(p, x, jac, source_residual):
            seen.append(source_residual.copy())
        original = solver._hybrid_newton
        with patch.object(solver, 'newton_solve', return_value=(np.ones(2), True, [], None)), \
             patch.object(solver, '_remember_newton_metrics', return_value=(0., 0.)), \
             patch.object(solver, 'residual', side_effect=values) as residual, \
             patch.object(solver, '_remember_continuation_linearization', side_effect=remember):
            with exact_residual_context(trace):
                x, accepted, history = solver._hybrid_newton(problem, np.zeros(2), options)
            self.assertIs(solver._hybrid_newton, original)
        self.assertTrue(accepted)
        return residual.call_count, seen, trace, history

    def test_same_exact_vector_used_for_guard_and_handoff(self):
        count, seen, trace, _ = self.run_acceptance(True, [np.array([2., 3.])])
        self.assertEqual(count, 1)
        np.testing.assert_array_equal(seen[0], [2., 3.])
        self.assertEqual(len(trace), 1)

    def test_rejected_exact_vector_never_reused(self):
        count, seen, trace, history = self.run_acceptance(True,
            [np.array([1e5, 0.]), np.array([2., 3.])])
        self.assertEqual(count, 2)
        self.assertEqual(len(seen), 1)
        np.testing.assert_array_equal(seen[0], [2., 3.])
        self.assertTrue(any(h['phase'] == 'exact_transport_reject' for h in history))

    def test_without_lag_retains_one_evaluation(self):
        count, seen, trace, _ = self.run_acceptance(False, [np.array([2., 3.])])
        self.assertEqual(count, 1)
        self.assertEqual(trace, [])

    def test_generated_sums_preserve_order_and_indices(self):
        reference = backend._eval_thermo_kinetics_sparse_numba_core
        names = list(inspect.signature(reference.py_func).parameters)
        data = dict(W=np.array([2., 16., 18.]), net_count=np.array([3, 2]),
            net_idx=np.array([[0, 1, 2], [2, 0, 0]]),
            net_nu=np.array([[-2., -1., 2.], [-1., 1., 0.]]))
        _, source = generate(reference, [data.get(n) for n in names])
        helpers = source.split('def _eval_thermo_kinetics_sparse_numba_core', 1)[0]
        ns = dict(np=np)
        exec(helpers, ns)
        g = np.array([1.5, -4., 3.])
        delta = ns['generated_delta'](g)
        expected = np.array([sum(data['net_nu'][r,i]*g[data['net_idx'][r,i]]
            for i in range(data['net_count'][r])) for r in range(2)])
        np.testing.assert_array_equal(delta, expected)
        out, expected = np.zeros((3,1)), np.zeros((3,1))
        q = np.array([2., -3.])
        ns['generated_production'](q, data['W'], out, 0)
        for r in range(2):
            for i in range(data['net_count'][r]):
                k = data['net_idx'][r,i]
                expected[k,0] += data['net_nu'][r,i]*q[r]*data['W'][k]
        np.testing.assert_array_equal(out, expected)

    def test_new_candidates_do_not_reopen_archived_kernels(self):
        args = parse_arguments(['--variants', 'baseline', 'exact-residual', 'nonlinear-blocks'])
        with patch('benchmark_cold_strategies.build_candidates', side_effect=AssertionError):
            self.assertEqual(list(selected_kernels(args.variants)), ['baseline'])
