"""Public input boundaries and composition/grid propagation to native physics."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

import kflame
from kflame.chemistry.initialization import fresh_mixture
from kflame.chemistry.mechanism import load_mechanism
from kflame.flame.config import FlameCase
from kflame.flame.problem import FreeFlameProblem


class PublicAPITests(unittest.TestCase):
    def test_invalid_inputs_fail_before_creating_output(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'invalid'
            for inputs in ({'transport': 'ionized'}, {'temperature': -1},
                           {'phi': 1., 'X': 'CH4:1'}, {'grid': [0., .02, .01, .03]},
                           {'dilution': .1}, {'species': ('unknown-species',)}):
                with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                    kflame.solve_flame(output=output, **inputs)
                self.assertFalse(output.exists())
            with self.assertRaises(TypeError):
                kflame.solve_flame(max_jac_age=80)
            with self.assertRaises(ValueError):
                kflame.generate_fgm(phis=(1., .8), output=output)
            self.assertFalse(output.exists())

    def test_direct_inlet_and_grid_keep_native_equilibrium(self):
        mech = load_mechanism('h2o2.yaml')
        y = fresh_mixture(mech, .8, 'H2', 'O2:1,N2:3.76')
        default = FreeFlameProblem(FlameCase(mech='h2o2.yaml', phi=.8, fuel='H2'))
        grid = (0., .002, .006, .009, .012, .016, .025, .03)
        direct = FreeFlameProblem(FlameCase(mech='h2o2.yaml', inlet_mass_fractions=tuple(y),
                                          initial_grid=grid, steady_rtol=2e-5, steady_atol=3e-10))
        np.testing.assert_array_equal(direct.Y_in, default.Y_in)
        np.testing.assert_array_equal(direct.Y_eq, default.Y_eq)
        np.testing.assert_array_equal(direct.z, grid)
        self.assertEqual(direct.T_ad, default.T_ad)
        self.assertEqual((direct.steady_rtol, direct.steady_atol), (2e-5, 3e-10))


if __name__ == '__main__':
    unittest.main()
