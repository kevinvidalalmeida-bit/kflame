"""Contracts for new isolated cold-start prototypes, without heavy timing."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'validation'), str(ROOT/'V2'), str(ROOT/'V2/materiales')]
import equations
import solver
from cold_dependencies import precompute
from cold_transport_candidates import blend_context


class ColdFollowupGuards(unittest.TestCase):
    def test_blend_residual_and_jacobian_have_identical_weights(self):
        mixture, target = object(), object()
        problem = SimpleNamespace(backend=target, n_points=1)
        def residual(x, p, **kwargs):
            return np.full(3, 2. if equations._residual_backend(p) is mixture else 4.)
        def jacobian(fun, x, p, eps=1e-5):
            value = 2. if equations._jacobian_backend(p) is mixture else 4.
            j = equations.BlockTridiagJacobian(np.empty((0, 3, 3)),
                (value*np.eye(3))[None], np.empty((0, 3, 3)))
            return j, j.diagonal()
        with patch.object(equations, 'residual', residual), \
             patch.object(equations, 'build_jacobian_steady', jacobian):
            for alpha in (0., .5, 1.):
                with blend_context(problem, mixture, alpha, []):
                    np.testing.assert_array_equal(solver.residual(np.zeros(3), problem),
                                                  np.full(3, 2+2*alpha))
                    j, diagonal = solver.build_jacobian_steady(None, np.zeros(3), problem)
                    np.testing.assert_array_equal(diagonal, np.full(3, 2+2*alpha))
        self.assertFalse(hasattr(problem, 'residual_backend'))
        self.assertFalse(hasattr(problem, 'jacobian_backend'))
        self.assertIs(problem.backend, target)

    def test_dependency_layout_reuses_only_velocity_column(self):
        class Backend:
            def eval_grid_thermo_kinetics_into(self, t, y, omega, enthalpy):
                omega[:] = 3*y
                enthalpy[:] = t[None, :]
                return t+y.sum(axis=0), 2*t
        state = np.array([[1., 300., .8, .2], [2., 1500., 1., -1e-8]])
        n, k, nv = 2, 2, 4
        t, y = state[:, 1], state[:, 2:].T
        cache = dict(n_pts=n, n_sp=k, nv=nv, rho=t+y.sum(axis=0), cp_n=2*t,
                     omega=3*y, hk_n=np.broadcast_to(t, (k, n)).copy())
        problem = SimpleNamespace(backend=Backend())
        trace = []
        result = precompute(state.ravel(), problem, cache, 1e-5, 1e-10, 1e-5,
                            SimpleNamespace(indexed=lambda *a: None), trace)
        for j in range(n):
            for v in range(nv):
                trial = state[j].copy()
                trial[v] += (abs(trial[v])*1e-5+1e-10)*(-1 if trial[v] < 0 else 1)
                self.assertAlmostEqual(result[0][j, v], trial[1]+sum(trial[2:]))
                self.assertAlmostEqual(result[1][j, v], 2*trial[1])
                np.testing.assert_array_equal(result[2][:, j, v], 3*trial[2:])
                np.testing.assert_array_equal(result[3][:, j, v], np.full(k, trial[1]))
        self.assertEqual(trace[0]['chemistry_states'], n*(nv-1))
        self.assertEqual(trace[0]['skipped_velocity_states'], n)


if __name__ == '__main__':
    unittest.main()
