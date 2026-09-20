"""Exact thermal reuse must not approximate any Jacobian perturbation."""
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'V2'), str(ROOT / 'V2/materiales')]

import numpy as np
from config import FlameCase
from problem import FreeFlameProblem
from species_backend_native import NativeSpeciesBackend
import species_backend_native as native
from equations import build_jacobian_steady, residual
from state import pack_state


def inputs(backend):
    ns, nv = backend.n_species, backend.n_species + 2
    temperatures = np.array([300., 999.9999, 1000., 1000.0001, 1800., 2500.])
    rng = np.random.default_rng(1920)
    y = rng.uniform(.01, 1., (ns, temperatures.size))
    y /= y.sum(axis=0)
    y[0, 1], y[1, 1] = -1e-8, 0.
    # Include exactly zero species and a reactive, non-random fresh mixture.
    y[:, 0] = backend.problem.Y_in
    t = np.repeat(temperatures, nv)
    y = np.repeat(y, nv, axis=1)
    for j in range(temperatures.size):
        t[j * nv + 1] += temperatures[j] * 1e-5 + 1e-10
        for k in range(ns):
            y[k, j * nv + k + 2] += abs(y[k, j * nv]) * 1e-5 + 1e-10
    return t, y


def evaluate(method, t, y):
    omega, h = np.empty_like(y), np.empty_like(y)
    rho, cp = method(t, y, omega, h)
    return rho, cp, omega, h


@unittest.skipIf(native._eval_jacobian_grouped_core is None, 'Numba not available')
class ThermalReuseTests(unittest.TestCase):
    def backend(self, mechanism='h2o2.yaml', pressure=101325.):
        p = FreeFlameProblem(FlameCase(mech=mechanism, fuel='H2', P=pressure))
        p.assume_finite_y = True
        p.backend = NativeSpeciesBackend(p)
        return p.backend

    def assert_equal_outputs(self, expected, actual):
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)

    def test_two_mechanisms_three_pressures_signed_and_nasa_switch_states(self):
        for mechanism in ('gri30.yaml', 'h2o2.yaml'):
            for pressure in (101325., 1013250., 10132500.):
                with self.subTest(mechanism=mechanism, pressure=pressure):
                    b = self.backend(mechanism, pressure)
                    t, y = inputs(b)
                    self.assert_equal_outputs(
                        evaluate(b.eval_grid_thermo_kinetics_into, t, y),
                        evaluate(b.eval_jacobian_thermo_kinetics_into, t, y))

    def test_unstructured_and_empty_batches_fall_back(self):
        b = self.backend()
        t, y = inputs(b)
        for size, change in ((5, False), (len(t), True), (0, False)):
            with self.subTest(size=size, change=change):
                test_t, test_y = t[:size].copy(), y[:, :size].copy()
                if change:
                    test_t[2] += .1
                expected = evaluate(b.eval_grid_thermo_kinetics_into, test_t, test_y)
                with patch.object(native, '_eval_jacobian_grouped_core', side_effect=AssertionError('invalid layout')):
                    self.assert_equal_outputs(expected, evaluate(b.eval_jacobian_thermo_kinetics_into, test_t, test_y))

    def test_no_fused_kernel_falls_back(self):
        b = self.backend()
        b.problem.use_fused_numba_thermo_kinetics = False
        t, y = inputs(b)
        expected = evaluate(b.eval_grid_thermo_kinetics_into, t, y)
        with patch.object(native, '_eval_jacobian_grouped_core', side_effect=AssertionError('fused disabled')):
            self.assert_equal_outputs(expected, evaluate(b.eval_jacobian_thermo_kinetics_into, t, y))

    def test_strided_inputs_and_outputs(self):
        b = self.backend()
        t, y = inputs(b)
        expected = evaluate(b.eval_grid_thermo_kinetics_into, t, y)
        storage_t = np.repeat(t, 2)
        storage_y = np.repeat(y, 2, axis=1)
        omega, h = np.empty_like(storage_y), np.empty_like(storage_y)
        rho, cp = b.eval_jacobian_thermo_kinetics_into(storage_t[::2], storage_y[:, ::2], omega[:, ::2], h[:, ::2])
        self.assert_equal_outputs(expected, (rho, cp, omega[:, ::2], h[:, ::2]))

    def test_real_jacobian_blocks_equal_with_and_without_reuse(self):
        b = self.backend()
        p = b.problem
        p.jacobian_mode = 'block_tridiag'
        p.precompute_jacobian_thermo = True
        f = np.linspace(0., 1., p.n_points)
        t = p.T_in + f * (p.T_ad - p.T_in)
        y = p.Y_in[:, None] * (1-f) + p.Y_eq[:, None] * f
        p.setup_fixed_temperature(T_profile=t)
        x = pack_state(np.ones(p.n_points), t, y)
        with patch.object(b, 'eval_jacobian_thermo_kinetics_into', wraps=b.eval_jacobian_thermo_kinetics_into) as grouped:
            new, new_diagonal = build_jacobian_steady(residual, x, p)
            self.assertTrue(grouped.called)
        with patch.object(b, 'eval_jacobian_thermo_kinetics_into', b.eval_grid_thermo_kinetics_into):
            old, old_diagonal = build_jacobian_steady(residual, x, p)
        np.testing.assert_array_equal(new_diagonal, old_diagonal)
        for name in ('lower', 'diag', 'upper'):
            np.testing.assert_array_equal(getattr(new, name), getattr(old, name))


if __name__ == '__main__':
    unittest.main()
