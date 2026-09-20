"""Native initialization, FGM diagnostics and molecular-fit cache contracts."""
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in ('V2', 'V2/materiales', 'FGM/scripts'):
    sys.path.insert(0, str(ROOT / folder))

from config import FlameCase
from problem import FreeFlameProblem
from mechanism_data import load_mechanism, resolve_mechanism
from initialization_native import fresh_mixture, NativeMixture
from thermo_native import NativeThermo
from transport_native import NativeTransport
from species_backend_native import NativeSpeciesBackend
from collision_integrals_native import native_transport_fits
from fgm_common import compute_bilger_Z, invert_bilger_Z_to_phi
from generate_fgm_tables_native import build_argparser, tabulated_properties, seed_cache_key

try:
    import cantera as ct
except ImportError:
    ct = None


class NativeContracts(unittest.TestCase):
    def test_seed_cache_tracks_mechanism_content_not_just_path(self):
        args = build_argparser().parse_args([])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'mechanism.yaml'
            path.write_bytes(Path(resolve_mechanism('h2o2.yaml')).read_bytes())
            initial = seed_cache_key(args, str(path))
            self.assertEqual(initial, seed_cache_key(args, str(path)))
            path.write_bytes(path.read_bytes() + b'\n# different input bytes\n')
            self.assertNotEqual(initial, seed_cache_key(args, str(path)))

    def test_default_construction_and_generator_import_without_cantera(self):
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join([
            str(ROOT / 'validation/no_cantera'), str(ROOT / 'V2'),
            str(ROOT / 'FGM/scripts')])
        code = '''
import sys
from pathlib import Path
from config import FlameCase
from problem import FreeFlameProblem
from solver import _make_backend
from generate_fgm_tables_native import build_argparser
p = FreeFlameProblem(FlameCase())
assert _make_backend(p).backend_kind == 'native'
assert build_argparser().parse_args([]).transport_backend == 'native'
assert 'cantera' not in sys.modules
for path in (str(Path('V2/materiales/cantera_transport_poly_coeffs.json')),
             bytes('V2/materiales/cantera_transport_poly_coeffs.json', 'utf-8')):
    try:
        open(path, 'rb')
    except RuntimeError:
        pass
    else:
        raise AssertionError('exported property fit guard was inactive')
try:
    import cantera
except ImportError:
    print('blocked and native setup passed')
else:
    raise AssertionError('import guard was inactive')
'''
        result = subprocess.run([sys.executable, '-c', code], env=env, cwd=ROOT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('blocked and native setup passed', result.stdout)

    def test_explicit_missing_mechanism_is_not_replaced_with_bundled_file(self):
        with self.assertRaises(FileNotFoundError):
            resolve_mechanism(ROOT / 'does_not_exist/gri30.yaml')

    def test_fit_cache_is_immutable_and_tracks_molecular_data(self):
        mech = load_mechanism('h2o2.yaml')
        base = NativeTransport(mech)
        fits = native_transport_fits(mech, base)
        mech.pressure *= 10.0
        self.assertIs(fits, native_transport_fits(mech, base))
        with self.assertRaises(ValueError):
            fits[1][0, 0] = 0.0
        mech.diameter[0] *= 1.01
        changed = native_transport_fits(mech, NativeTransport(mech))
        self.assertIsNot(fits, changed)
        self.assertFalse(np.array_equal(fits[1], changed[1]))


@unittest.skipIf(ct is None, 'Optional Cantera comparison dependency absent')
class IndependentReferenceChecks(unittest.TestCase):
    def test_initialization_over_fuels_mixtures_pressure_and_inlet_temperature(self):
        for mech_name, fuel in [('gri30.yaml', 'CH4'), ('gri30.yaml', 'H2'), ('h2o2.yaml', 'H2')]:
            mech = load_mechanism(mech_name)
            thermo = NativeThermo(mech)
            gas = ct.Solution(resolve_mechanism(mech_name))
            for phi in (.7, 1., 1.4):
                for pressure in (101325., 1013250.):
                    for temperature in (300., 700.):
                        with self.subTest(mech=mech_name, phi=phi, P=pressure, T=temperature):
                            case = FlameCase(mech=mech_name, fuel=fuel, phi=phi, P=pressure, T_in=temperature)
                            p = FreeFlameProblem(case, mech_data=mech)
                            gas.TP = temperature, pressure
                            gas.set_equivalence_ratio(phi, fuel, case.oxidizer)
                            np.testing.assert_allclose(p.Y_in, gas.Y, atol=1e-14, rtol=0)
                            gas.equilibrate('HP')
                            self.assertAlmostEqual(p.T_ad, gas.T, delta=2e-5)
                            np.testing.assert_allclose(p.Y_eq, gas.Y, atol=2e-9, rtol=0)
                            # Check conservation independently of the reference solver.
                            elements = mech.atom_matrix / mech.molecular_weights
                            np.testing.assert_allclose(elements @ p.Y_in, elements @ p.Y_eq, atol=1e-10, rtol=1e-8)
                            enthalpy_in = p.Y_in @ (thermo.partial_molar_enthalpies(temperature) / mech.molecular_weights)
                            enthalpy_out = p.Y_eq @ (thermo.partial_molar_enthalpies(p.T_ad) / mech.molecular_weights)
                            self.assertAlmostEqual(enthalpy_in, enthalpy_out, delta=.01)

    def test_very_lean_equilibrium_below_400K_and_diluted_streams(self):
        for fuel, oxidizer, phi in [('H2', 'O2:1,N2:3.76', .01),
                                    ('CH4', 'O2:1,N2:3.76', .01),
                                    ('CH4:1,H2:1,CO2:.2', 'O2:1,N2:3.76,CO2:.2', .8)]:
            p = FreeFlameProblem(FlameCase(fuel=fuel, oxidizer=oxidizer, phi=phi))
            gas = ct.Solution(resolve_mechanism('gri30.yaml'))
            gas.TP = 300., 101325.
            gas.set_equivalence_ratio(phi, fuel, oxidizer)
            np.testing.assert_allclose(p.Y_in, gas.Y, atol=1e-14)
            gas.equilibrate('HP')
            self.assertAlmostEqual(p.T_ad, gas.T, delta=2e-5)

    def test_bilger_preserves_historical_stream_basis_and_inverse(self):
        args = build_argparser().parse_args([])
        for fuel in ('CH4', 'H2', 'CH4:1,H2:1'):
            args.fuel = fuel
            native = NativeMixture(load_mechanism(args.mech))
            gas = ct.Solution(resolve_mechanism(args.mech))
            for phi in (.1, .7, 1., 1.4, 10.):
                z = compute_bilger_Z(phi, args, native)
                self.assertAlmostEqual(z, compute_bilger_Z(phi, args, gas), delta=1e-14)
                if z > 1e-10:
                    back = invert_bilger_Z_to_phi(z, args, phi_lo=1e-4, phi_hi=1e3, gas=native)
                    self.assertAlmostEqual(back, phi, delta=1e-7)

    def test_native_table_fields_follow_selected_transport(self):
        for model in ('mixture-averaged', 'multicomponent'):
            p = FreeFlameProblem(FlameCase(transport_model=model))
            p.backend = NativeSpeciesBackend(p)
            t = np.linspace(500., 2300., 11)
            fraction = np.linspace(.1, .99, 11)
            y = p.Y_in[:, None] * (1-fraction) + p.Y_eq[:, None] * fraction
            weights = {'CO2': 1., 'H2O': 1., 'CO': 1., 'H2': .5}
            actual = tabulated_properties(p, t, y, weights)
            gas = ct.Solution(resolve_mechanism(p.case.mech), transport_model=model)
            expected = np.empty((5, len(t)))
            w = np.array([weights.get(s, 0.) for s in gas.species_names])
            for j in range(len(t)):
                gas.TPY = t[j], p.P, y[:, j]
                expected[:, j] = (gas.density, gas.cp_mass, gas.thermal_conductivity,
                                  gas.heat_release_rate, w @ (gas.net_production_rates * gas.molecular_weights))
            for i in (0, 1, 3, 4):
                np.testing.assert_allclose(actual[i], expected[i], rtol=2e-9, atol=1e-6)
            np.testing.assert_allclose(actual[2], expected[2], rtol=2e-6, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
