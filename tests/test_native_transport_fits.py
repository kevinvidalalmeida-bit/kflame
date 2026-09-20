"""Transport must regenerate its own fits, including for modified mechanisms."""
from pathlib import Path
import copy
import os
import sys
import unittest

os.environ.setdefault('NUMBA_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
import numpy as np
from kava.chemistry.mechanism import load_mechanism, resolve_mechanism
from kava.chemistry.transport import NativeTransport

try:
    import cantera as ct
except ImportError:
    ct = None


class NativeFitContracts(unittest.TestCase):
    def test_immutable_molecular_and_property_cache_tracks_inputs_not_names(self):
        mech = load_mechanism('h2o2.yaml')
        a = NativeTransport(mech)
        mech.pressure *= 10
        b = NativeTransport(mech)
        for name in ('_eps_pair', '_sigma_pair', '_visc_poly', '_cond_poly', '_diff_poly'):
            self.assertIs(getattr(a, name), getattr(b, name))
            self.assertFalse(getattr(a, name).flags.writeable)
        changed = copy.deepcopy(mech)
        changed.diameter[0] *= 1.02
        c = NativeTransport(changed)
        self.assertIsNot(a._eps_pair, c._eps_pair)
        np.testing.assert_allclose(c.species_viscosities(900.)[0] / a.species_viscosities(900.)[0],
                                   1 / 1.02**2, rtol=1e-9)
        self.assertFalse(np.array_equal(a._cond_poly, c._cond_poly))

    def test_conductivity_cache_tracks_nasa_and_rotational_inputs(self):
        mech = load_mechanism('h2o2.yaml')
        a = NativeTransport(mech)
        for field in ('nasa_low', 'nasa_high', 'nasa_Tmid', 'rot_relax', 'geometry'):
            with self.subTest(field=field):
                modified = copy.deepcopy(mech)
                if field.startswith('nasa_') and field != 'nasa_Tmid':
                    getattr(modified, field)[0, 0] += .1
                else:
                    getattr(modified, field)[0] += 1
                b = NativeTransport(modified)
                self.assertIsNot(a._cond_poly, b._cond_poly)
                self.assertIs(a._visc_poly, b._visc_poly)

    def test_fast_and_vectorized_evaluation_agree(self):
        mech = load_mechanism('gri30.yaml')
        transport = NativeTransport(mech)
        t = np.array([300., 700., 1500., 2500.])
        y = np.random.default_rng(21).uniform(.01, 1., (mech.n_species, t.size))
        y /= y.sum(axis=0)
        x = y * mech.inv_molecular_weights[:, None]
        x /= x.sum(axis=0)
        rho, diff, conductivity, mw = transport.eval_faces_poly_fast(t, 101325., y, mech.inv_molecular_weights)
        np.testing.assert_allclose(diff, transport.mix_diff_coeffs(t, 101325., x), rtol=2e-13)
        np.testing.assert_allclose(conductivity, transport.thermal_conductivity(t, x), rtol=2e-13)
        self.assertTrue(np.all(rho > 0) and np.all(mw > 0))


@unittest.skipIf(ct is None, 'Optional independent Cantera reference')
class NativeFitReference(unittest.TestCase):
    def test_generated_fits_follow_each_mechanism(self):
        for name in ('gri30.yaml', 'h2o2.yaml'):
            mech = load_mechanism(name)
            native = NativeTransport(mech)
            gas = ct.Solution(resolve_mechanism(name))
            x = np.random.default_rng(12).uniform(.01, 1., mech.n_species)
            x /= x.sum()
            for t in (300., 700., 1000., 1800., 2800.):
                with self.subTest(mechanism=name, temperature=t):
                    gas.TPX = t, 1013250., x
                    np.testing.assert_allclose(native.species_viscosities(t), gas.species_viscosities, rtol=2e-7)
                    np.testing.assert_allclose(native.viscosity(t, x), gas.viscosity, rtol=2e-7)
                    np.testing.assert_allclose(native.thermal_conductivity(t, x), gas.thermal_conductivity, rtol=2e-7)
                    np.testing.assert_allclose(native.binary_diff_coeffs(t)/gas.P, gas.binary_diff_coeffs, rtol=2e-7)


if __name__ == '__main__':
    unittest.main()
