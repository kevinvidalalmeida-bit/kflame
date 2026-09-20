"""Allowed Newton trial compositions must not lose their chemical restoring term."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import cantera as ct
except ImportError:
    ct = None
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'V2' / 'materiales'))
from species_backend_native import NativeSpeciesBackend
from kinetics_native import _negative_mass_action_factor_python
from kinetics_native import _make_mass_action_plan, _mass_action_product_python


class SignedKineticsTests(unittest.TestCase):
    @unittest.skipIf(ct is None, 'Optional Cantera reference dependency not installed')
    def test_specialized_fused_rates_across_temperature_pressure_and_progress(self):
        for mech, fuel in (('gri30.yaml', 'CH4'), ('h2o2.yaml', 'H2')):
            path = str(Path(ct.get_data_directories()[-1]) / mech)
            for atm in (1., 10., 100.):
                case = SimpleNamespace(mech=path, transport_model='mixture-averaged', soret_enabled=False)
                backend = NativeSpeciesBackend(SimpleNamespace(case=case, P=atm*ct.one_atm))
                gas = ct.Solution(path)
                states, temperatures, references = [], [], []
                for phi in (.7, 1., 1.4):
                    gas.TP = 300., atm*ct.one_atm
                    gas.set_equivalence_ratio(phi, fuel, 'O2:1,N2:3.76')
                    fresh = gas.Y.copy()
                    gas.equilibrate('HP')
                    burned = gas.Y.copy()
                    for fraction, temperature in zip((0., .1, .5, 1.), (300., 700., 1200., 2400.)):
                        y = (1-fraction)*fresh + fraction*burned
                        gas.TPY = temperature, atm*ct.one_atm, y
                        states.append(gas.Y.copy())
                        temperatures.append(temperature)
                        references.append(gas.net_production_rates*gas.molecular_weights)
                y = np.array(states).T.copy()
                output = np.empty_like(y)
                backend.eval_grid_thermo_kinetics_into(np.array(temperatures), y, output, np.empty_like(y))
                with self.subTest(mechanism=mech, atm=atm):
                    np.testing.assert_allclose(output, np.array(references).T, rtol=2e-7, atol=1e-7)

    def test_product_preserves_representable_extreme_results(self):
        idx = np.array([[0, 1, 2]])
        orders = np.ones((1, 3))
        plan = _make_mass_action_plan(idx, orders, np.array([3]))[0]
        for c in (np.array([1e200, 1e200, 1e-200]),
                  np.array([1e-200, 1e-200, 1e200])):
            with np.errstate(over='ignore', under='ignore'):
                actual = _mass_action_product_python(c, np.log(c), idx[0], orders[0], 3, plan, False)
            self.assertTrue(np.isfinite(actual) and actual > 0)
            np.testing.assert_allclose(actual, np.exp(np.log(c).sum()), rtol=2e-13, atol=0)

    def test_specialized_products_and_general_order_fallback(self):
        idx = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
        orders = np.array([[1., 1.], [2., 1.], [.5, 1.], [1., 3.], [0., 0.]])
        count = np.array([2, 2, 2, 2, 0])
        plan = _make_mass_action_plan(idx, orders, count)
        np.testing.assert_array_equal(plan[:, 0], [2, 3, -1, -1, 0])
        for c in (np.array([2., 3.]), np.array([-2., 3.]), np.array([-2., -3.]),
                  np.array([0., 3.]), np.array([1e-100, 2e-100])):
            logs = np.log(np.maximum(abs(c), 1e-300))
            for r in range(len(count)):
                expected = np.exp(sum(orders[r, i] * logs[idx[r, i]] for i in range(count[r])))
                expected *= _negative_mass_action_factor_python(c, idx[r], orders[r], count[r])
                actual = _mass_action_product_python(c, logs, idx[r], orders[r], count[r],
                                                     plan[r], bool(np.any(c < 0)))
                np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-300)

    def test_mass_action_extension_does_not_create_spurious_negative_reactions(self):
        for c, orders, expected in (([-1., 2.], [1., 1.], -1.),
                                    ([-1., -2.], [1., 1.], 0.),
                                    ([-1., 2.], [2., 1.], 0.),
                                    ([-1., 2.], [.5, 1.], 0.),
                                    ([-1., 2.], [1., 3.], 0.),
                                    ([-1., 2.], [0., 1.], 1.)):
            self.assertEqual(_negative_mass_action_factor_python(
                np.array(c), np.arange(2), np.array(orders), 2), expected)

    @unittest.skipIf(ct is None, 'Optional Cantera reference dependency not installed')
    def test_negative_trace_sources_match_reference_in_all_native_paths(self):
        for mech in ('h2o2.yaml', 'gri30.yaml'):
            for atm in (1, 10):
                case = SimpleNamespace(mech=str(Path(ct.get_data_directories()[-1]) / mech),
                                       transport_model='mixture-averaged', soret_enabled=False)
                backend = NativeSpeciesBackend(SimpleNamespace(case=case, P=atm * ct.one_atm))
                gas = ct.Solution(case.mech)
                gas.TP = 1200., atm * ct.one_atm
                gas.set_equivalence_ratio(1., 'H2', 'O2:1,N2:3.76')
                fresh = gas.Y.copy()
                states = []
                for names in (('H',), ('OH',), ('HO2',), ('H2O2',), ('H', 'OH')):
                    y = fresh.copy()
                    for name in names:
                        y[gas.species_index(name)] = -1e-8
                    states.append(y)
                # Straddle zero for the Jacobian's restoring derivative.
                for value in (-1e-10, 0., 1e-10):
                    state = fresh.copy()
                    state[gas.species_index('H')] = value
                    states.append(state)
                y = np.array(states).T.copy()
                t = np.full(y.shape[1], 1200.)
                reference, concentrations = [], []
                for state in states:
                    gas.set_unnormalized_mass_fractions(state)
                    gas.TP = 1200., atm * ct.one_atm
                    reference.append(gas.net_production_rates * gas.molecular_weights)
                    concentrations.append(gas.concentrations.copy())
                reference = np.array(reference).T
                c = np.array(concentrations).T.copy()
                g = backend.thermo.g_RT(t)
                omega, hk = np.empty_like(y), np.empty_like(y)
                backend.eval_grid_thermo_kinetics_into(t, y, omega, hk)
                outputs = {'fused': omega.copy()}
                for label, sparse, dense in (('numpy', False, False), ('sparse', True, False),
                                             ('dense', False, True)):
                    backend.kinetics._use_sparse_numba = sparse
                    backend.kinetics._use_numba = dense
                    outputs[label] = backend.kinetics.net_production_rates(t, c, g) * gas.molecular_weights[:, None]
                for label, actual in outputs.items():
                    with self.subTest(mechanism=mech, atm=atm, path=label):
                        np.testing.assert_allclose(actual, reference, rtol=2e-7, atol=1e-9)
                        self.assertLess(np.max(np.abs(np.sum(actual, axis=0))),
                                        1e-11 * max(1., np.max(np.abs(actual))))
                        np.testing.assert_allclose((actual[:, -1] - actual[:, -3]) / 2e-10,
                                                   (reference[:, -1] - reference[:, -3]) / 2e-10,
                                                   rtol=2e-7, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
