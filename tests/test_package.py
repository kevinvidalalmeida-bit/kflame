"""Installed package contracts, including CLI execution away from the checkout."""
from importlib.resources import files
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PackageTests(unittest.TestCase):
    def test_bundled_data_without_exported_property_archive(self):
        data = files('kava.chemistry').joinpath('data')
        for name in ('gri30.yaml', 'h2o2.yaml', 'collision_integrals_mm.json', 'CANTERA_TRANSPORT_LICENSE.txt'):
            self.assertTrue(data.joinpath(name).is_file(), name)
        self.assertFalse(data.joinpath('cantera_transport_poly_coeffs.json').is_file())

    def test_cli_native_help_outside_checkout_without_cantera(self):
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(__file__).resolve().parent / 'no_cantera')
        with tempfile.TemporaryDirectory() as folder:
            for command in ('fgm', 'export', 'soret'):
                result = subprocess.run([sys.executable, '-m', 'kava', command, '--help'],
                                        env=env, cwd=folder, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('--help', result.stdout)


if __name__ == '__main__':
    unittest.main()
