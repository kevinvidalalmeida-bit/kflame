"""Regression coverage for FGM figures generated from a saved table."""

import numpy as np
import pytest


matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from kflame.fgm.plot import plot_adaptive_fgm_map


def test_adaptive_map_recreates_thesis_figure(tmp_path):
    z_grid = np.array([0.02, 0.05, 0.08])
    c_grid = np.linspace(0.0, 1.0, 5)
    y = np.zeros((z_grid.size, 2, c_grid.size))
    y[:, 0, :] = z_grid[:, None] * c_grid[None, :]
    y[:, 1, :] = (1.0 - z_grid[:, None]) * c_grid[None, :] ** 2
    qdot = 4.0e8 * (z_grid[:, None] + c_grid[None, :])
    omega_c = 1.0e3 * (z_grid[:, None] + c_grid[None, :] ** 2)
    np.savez_compressed(
        tmp_path / "fgm_table.npz",
        Z_grid=z_grid,
        c_grid=c_grid,
        qdot=qdot,
        omega_c=omega_c,
        Y=y,
        species_names=np.array(["CO", "CO2"], dtype=object),
    )

    plot_adaptive_fgm_map(tmp_path, dpi=72)

    output = tmp_path / "fgm_mapa_adaptativo.pdf"
    assert output.exists()
    assert output.stat().st_size > 1_000
