"""Safety checks for a first-grid nonlinear overlapping-block sweep."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'validation'), str(ROOT/'V2'), str(ROOT/'V2/materiales')]
import solver
import equations
from cold_nonlinear_blocks import local_matrix, sweep


class NonlinearBlockTests(unittest.TestCase):
    def fixture(self):
        j = equations.BlockTridiagJacobian(np.full((2,1,1), -.2),
            np.full((3,1,1), 2.), np.full((2,1,1), -.2))
        return j, SimpleNamespace(n_points=3), solver.SolveOptions()

    def test_two_node_block_preserves_couplings(self):
        j, _, _ = self.fixture()
        np.testing.assert_array_equal(local_matrix(j, 1), [[2., -.2], [-.2, 2.]])

    def test_residual_reduction_and_inputs_unchanged(self):
        j, p, opts = self.fixture()
        x = np.zeros(3)
        rhs = np.array([1., 2., 3.])
        records = []
        with patch.object(solver, 'residual', side_effect=lambda v, p, **kw: j.matvec(v)-rhs), \
             patch.object(solver, 'build_jacobian_steady', return_value=(j,j.diagonal())), \
             patch.object(solver, 'bound_step_limit', return_value=(1., 'ok')):
            out = sweep(p, x, opts, records)
        np.testing.assert_array_equal(x, np.zeros(3))
        self.assertTrue(records[0]['accepted'])
        self.assertGreater(records[0]['local_updates'], 0)
        self.assertLess(np.linalg.norm(j.matvec(out)-rhs), np.linalg.norm(rhs))

    def test_root_is_fixed_without_building_jacobian(self):
        _, p, opts = self.fixture()
        with patch.object(solver, 'residual', return_value=np.zeros(3)), \
             patch.object(solver, 'build_jacobian_steady', side_effect=AssertionError):
            x = np.ones(3)
            np.testing.assert_array_equal(sweep(p,x,opts,[]), x)

    def test_global_variant_keeps_monotone_merit(self):
        j, p, opts = self.fixture()
        records = []
        rhs = np.array([1., 2., 3.])
        with patch.object(solver, 'residual', side_effect=lambda v, p, **kw: j.matvec(v)-rhs), \
             patch.object(solver, 'build_jacobian_steady', return_value=(j,j.diagonal())), \
             patch.object(solver, 'bound_step_limit', return_value=(1., 'ok')):
            sweep(p, np.zeros(3), opts, records, global_descent=True)
        merits = [records[0]['merit_initial']]+records[0]['accepted_merits']
        self.assertGreater(len(merits), 1)
        self.assertTrue(np.all(np.diff(merits) < 0))

    def test_singular_blocks_and_deadline_roll_back(self):
        j, p, opts = self.fixture()
        j.lower[:] = 0; j.upper[:] = 0; j.diag[:] = 0
        records = []
        with patch.object(solver, 'residual', return_value=np.ones(3)), \
             patch.object(solver, 'build_jacobian_steady', return_value=(j,j.diagonal())):
            x = np.zeros(3)
            np.testing.assert_array_equal(sweep(p,x,opts,records), x)
            self.assertEqual(records[0]['singular_blocks'], 2)
            np.testing.assert_array_equal(sweep(p,x,opts,[],deadline=0.), x)
