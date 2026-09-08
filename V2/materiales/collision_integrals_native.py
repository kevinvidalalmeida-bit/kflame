"""Monchick--Mason collision integrals and transport fits without Cantera.

The tabulated data and fit conventions follow Cantera 3.2 (BSD-3-Clause);
see collision_integrals_mm.json and CANTERA_TRANSPORT_LICENSE.txt. These are
universal molecular collision data, not flame solutions or fitted Soret outputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval

from mechanism_data import AVOGADRO, BOLTZMANN


class CollisionIntegrals:
    def __init__(self, minimum: float, maximum: float):
        with Path(__file__).with_name("collision_integrals_mm.json").open(encoding="utf-8") as stream:
            data = json.load(stream)
        self.temperatures = np.asarray(data["tstar22"])
        self.log_temperature = np.log(self.temperatures)
        delta = np.asarray(data["delta"])
        self.tables = np.stack([
            np.asarray(data[key]).reshape(-1, 8)[1:-1]
            if key != "omega22_table" else np.asarray(data[key]).reshape(37, 8)
            for key in ("omega22_table", "astar_table", "bstar_table", "cstar_table")
        ])
        self.delta_polynomials = polyfit(delta, self.tables.reshape(-1, 8).T, 6).T.reshape(4, 37, 7)
        lower = np.flatnonzero(minimum > self.temperatures)
        upper = np.flatnonzero(maximum > self.temperatures)
        nmin = int(lower[-1]) if lower.size else -1
        nmax = int(upper[-1] + 1) if upper.size else -1
        if nmin < 0 or nmin >= 36 or nmax < 0 or nmax > 36:
            nmin, nmax = 0, 36
        self.fit_slice = slice(nmin, nmax + 1)

    def values(self, delta: float) -> np.ndarray:
        if delta == 0.0:
            return self.tables[:, :, 0]
        return polyval(float(delta), self.delta_polynomials.transpose(2, 0, 1))

    def fits(self, delta: float) -> np.ndarray:
        region = self.fit_slice
        return polyfit(self.log_temperature[region], self.values(delta)[:, region].T, 8).T

    def evaluate(self, temperature: np.ndarray, delta: float) -> np.ndarray:
        """Local quadratic interpolation in log reduced temperature."""
        temperature = np.asarray(temperature, dtype=float)
        left = np.clip(np.searchsorted(self.temperatures, temperature, side="right") - 1, 0, 33)
        indices = left[..., None] + np.arange(3)
        nodes = self.log_temperature[indices]
        target = np.log(temperature)
        weights = np.empty_like(nodes)
        for k in range(3):
            others = [i for i in range(3) if i != k]
            weights[..., k] = ((target - nodes[..., others[0]]) * (target - nodes[..., others[1]])
                               / ((nodes[..., k] - nodes[..., others[0]]) * (nodes[..., k] - nodes[..., others[1]])))
        return np.sum(self.values(delta)[:, indices] * weights, axis=-1)


def native_transport_fits(mech, base):
    """Fit viscosity, binary diffusion and A*, B*, C* from molecular data.

    Fits use the common NASA temperature interval. Relative least squares is
    evaluated natively; no pre-exported, mechanism-indexed polynomial is read.
    """
    eps = np.asarray(base._eps_pair)
    sigma = np.asarray(base._sigma_pair)
    delta = np.asarray(base._delta_pair)
    if np.any(eps <= 0) or np.any(sigma <= 0):
        raise ValueError("Multicomponent transport requires positive collision diameter and well depth.")
    tmin, tmax = mech.min_temperature, mech.max_temperature
    if not 0 < tmin < tmax:
        raise ValueError("No common positive temperature interval for transport fitting.")
    integrals = CollisionIntegrals(float(np.min(tmin / eps)), float(np.max(tmax / eps)))
    temperature = np.linspace(tmin, tmax, 50)
    logt = np.log(temperature)
    n = mech.n_species
    stars = np.empty((3, n, n, 9))
    viscosity = np.empty((n, 5))
    diffusion = np.empty((n, n, 5))
    fit_cache = {}
    for i in range(n):
        omega22 = integrals.evaluate(temperature / mech.well_depth[i], delta[i, i])[0]
        mu = (5.0 / 16.0 * np.sqrt(np.pi * mech.molecular_weights[i] * BOLTZMANN * temperature / (1000.0 * AVOGADRO))
              / (omega22 * np.pi * mech.diameter[i] ** 2))
        target = np.sqrt(mu / np.sqrt(temperature))
        viscosity[i] = polyfit(logt, target, 4, w=1.0 / target)
        for j in range(i, n):
            d = float(delta[i, j])
            if d not in fit_cache:
                fit_cache[d] = integrals.fits(d)[1:]
            stars[:, i, j] = stars[:, j, i] = fit_cache[d]
            values = integrals.evaluate(temperature / eps[i, j], d)
            omega11 = values[0] / values[1]
            # mech molecular weights are kg/kmol; reduced mass is kg/molecule.
            reduced_mass = (mech.molecular_weights[i] * mech.molecular_weights[j]
                            / (mech.molecular_weights[i] + mech.molecular_weights[j]) / (1000.0 * AVOGADRO))
            target = (3.0 / 16.0 * np.sqrt(2.0 * np.pi / reduced_mass)
                      * BOLTZMANN ** 1.5 / (np.pi * sigma[i, j] ** 2 * omega11))
            diffusion[i, j] = diffusion[j, i] = polyfit(logt, target, 4, w=1.0 / target)
    return stars, viscosity, diffusion
