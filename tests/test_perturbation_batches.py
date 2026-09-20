"""Vectorized Jacobian preparation must preserve every perturbation bit for bit."""
from pathlib import Path
import sys
import unittest

import numpy as np

from kava.flame.equations import _center_perturbation_states


def legacy_perturbation_states(x_r, rel, absolute, eps):
    n_pts, nv = x_r.shape
    temperature = np.repeat(x_r[:, 1], nv).astype(float, copy=True)
    y = np.repeat(x_r[:, 2:].T, nv, axis=1).astype(float, copy=True)
    for j in range(n_pts):
        dx = np.abs(x_r[j]) * rel + absolute
        dx[dx <= 0.] = max(abs(rel), abs(eps), 1e-10)
        dx = np.where(x_r[j] < 0., -dx, dx)
        temperature[j * nv + 1] = x_r[j, 1] + dx[1]
        for k in range(nv - 2):
            y[k, j * nv + 2 + k] = x_r[j, 2 + k] + dx[2 + k]
    return temperature, y


class PerturbationBatchTests(unittest.TestCase):
    def test_exact_for_signed_zero_and_strided_states(self):
        rng = np.random.default_rng(721)
        for nodes, species in ((1, 1), (4, 10), (271, 53)):
            storage = rng.normal(size=(nodes * 2, species + 2))
            storage[:, 1] = rng.uniform(300., 3000., nodes * 2)
            storage[::3, 2:] = 0.
            x = storage[::2]
            original = x.copy()
            for rel, absolute, eps in ((1e-5, 1e-10, 1e-5), (0., 0., 0.), (-1e-5, -1e-10, 1e-5)):
                with self.subTest(nodes=nodes, species=species, rel=rel):
                    actual = _center_perturbation_states(x, rel, absolute, eps)
                    expected = legacy_perturbation_states(x, rel, absolute, eps)
                    for a, b in zip(actual, expected):
                        np.testing.assert_array_equal(a, b)
                        self.assertFalse(np.shares_memory(a, x))
                    np.testing.assert_array_equal(x, original)


if __name__ == '__main__':
    unittest.main()
