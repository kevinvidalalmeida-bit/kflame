"""Native fresh-mixture and HP-equilibrium initialization; no Cantera import."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, least_squares, linprog
from scipy.special import logsumexp

from mechanism_data import MechanismData
from thermo_native import NativeThermo


def _composition(text: str, names: list[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(names)}
    amount = np.zeros(len(names))
    for token in text.split(","):
        fields = token.strip().split(":", 1)
        name = fields[0].strip()
        if name not in index:
            raise ValueError(f"Species '{name}' is not in the mechanism")
        value = float(fields[1]) if len(fields) == 2 else 1.0
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"Invalid amount specified for '{name}'")
        amount[index[name]] += value
    if amount.sum() <= 0.0:
        raise ValueError("The composition must contain at least one species")
    return amount


def fresh_mixture(mech: MechanismData, phi: float, fuel: str, oxidizer: str) -> np.ndarray:
    """Fresh mass fractions from fuel, oxidizer and equivalence ratio."""
    if not np.isfinite(phi) or phi < 0.0:
        raise ValueError("phi must be finite and nonnegative")
    fuel_n = _composition(fuel, mech.species_names)
    oxidizer_n = _composition(oxidizer, mech.species_names)
    demand = oxygen_demand(mech)
    oxygen_need = float(demand @ fuel_n)
    oxygen_available = -float(demand @ oxidizer_n)
    if oxygen_need <= 0.0 or oxygen_available <= 0.0:
        raise ValueError("Fuel/oxidizer streams do not define a combustible mixture")
    n = phi * oxygen_available / oxygen_need * fuel_n + oxidizer_n
    mass = n * mech.molecular_weights
    return mass / mass.sum()


def oxygen_demand(mech: MechanismData) -> np.ndarray:
    """O2 demand per mole of species (C -> CO2, H -> H2O, S -> SO2)."""
    weights = {'C': 1.0, 'H': 0.25, 'S': 1.0, 'O': -0.5}
    return np.array([weights.get(e, 0.0) for e in mech.element_names]) @ mech.atom_matrix


class NativeMixture:
    """Minimal mixture interface shared by native and reference FGM drivers.

    Stream basis is explicit. FGM historically mixes on a mole basis but
    normalizes Bilger against mass-basis streams; preserve that convention.
    """
    def __init__(self, mech: MechanismData):
        self.mech = mech
        self.species_names = mech.species_names
        self.n_species = mech.n_species
        self.Y = np.zeros(mech.n_species)

    def set_equivalence_ratio(self, phi, fuel, oxidizer):
        self.Y = fresh_mixture(self.mech, phi, fuel, oxidizer)

    def mixture_fraction(self, fuel, oxidizer, basis='mass'):
        beta = oxygen_demand(self.mech) / self.mech.molecular_weights
        streams = []
        for stream in (fuel, oxidizer):
            amounts = _composition(stream, self.species_names)
            if basis == 'mole':
                amounts *= self.mech.molecular_weights
            elif basis != 'mass':
                raise ValueError('Stream basis must be mole or mass')
            streams.append(float(beta @ (amounts / amounts.sum())))
        denominator = streams[0] - streams[1]
        if denominator <= 0.0:
            raise ValueError('Fuel must have greater oxygen demand than oxidizer')
        return float(np.clip((beta @ self.Y - streams[1]) / denominator, 0.0, 1.0))


def _equilibrium_moles(mech: MechanismData, thermo: NativeThermo, T: float,
                       P: float, atom_totals: np.ndarray) -> np.ndarray:
    g_rt = np.asarray(thermo.g_RT(T), dtype=float)
    active = atom_totals > 0.0
    A = mech.atom_matrix[active]
    b = atom_totals[active]
    # Species containing an element absent from the feed must have zero moles.
    participating = ((A.sum(axis=0) > 0.0)
                     & (mech.atom_matrix[~active].sum(axis=0) == 0.0))
    with np.errstate(divide='ignore'):
        log_a = np.log(A[:, participating])
    a_used = A[:, participating]
    g_used = g_rt[participating]
    def residual(unknown: np.ndarray) -> np.ndarray:
        lam, c = unknown[:-1], unknown[-1]
        log_n = -g_used - a_used.T @ lam - c
        log_atoms = logsumexp(log_a + log_n, axis=1)
        return np.append(log_atoms - np.log(b),
                         logsumexp(-g_used - a_used.T @ lam) - np.log(P / mech.ref_pressure))

    def jacobian(unknown: np.ndarray) -> np.ndarray:
        log_n = -g_used - a_used.T @ unknown[:-1] - unknown[-1]
        atom_weights = np.exp(log_a + log_n - logsumexp(log_a + log_n, axis=1)[:, None])
        mole_weights = np.exp(log_n - logsumexp(log_n))
        jac = np.empty((A.shape[0] + 1, A.shape[0] + 1))
        jac[:-1, :-1] = -atom_weights @ a_used.T
        jac[:-1, -1] = -1.0
        jac[-1, :-1] = -mole_weights @ a_used.T
        jac[-1, -1] = 0.0
        return jac

    result = least_squares(residual, np.zeros(A.shape[0] + 1), jac=jacobian, xtol=1e-11,
                           ftol=1e-11, gtol=1e-11, max_nfev=1000)
    if np.max(np.abs(residual(result.x))) > 1e-7:
        # Cold, nearly complete combustion can flatten the element-potential
        # Jacobian at the zero guess. A feasible Gibbs LP supplies dual element
        # potentials; the nonlinear solve still enforces ideal-gas mixing/P.
        estimate = linprog(g_used, A_eq=a_used, b_eq=b, bounds=(0, None), method='highs')
        if estimate.success:
            guess = np.append(-estimate.eqlin.marginals,
                              np.log(P / mech.ref_pressure / estimate.x.sum()))
            result = least_squares(residual, guess, jac=jacobian, xtol=1e-11,
                                   ftol=1e-11, gtol=1e-11, max_nfev=1000)
    if not result.success or np.max(np.abs(residual(result.x))) > 1e-7:
        raise RuntimeError("Native chemical-equilibrium solve did not converge")
    log_n = -g_rt - A.T @ result.x[:-1] - result.x[-1]
    log_n[~participating] = -np.inf
    return np.exp(np.clip(log_n, -700.0, 700.0)) * participating


def hp_equilibrium(mech: MechanismData, T_in: float, P: float, Y_in: np.ndarray):
    """Ideal-gas adiabatic equilibrium at constant enthalpy and pressure."""
    thermo = NativeThermo(mech)
    Y_in = np.asarray(Y_in, dtype=float)
    if not np.isfinite(T_in) or not np.isfinite(P) or T_in <= 0.0 or P <= 0.0:
        raise ValueError('Temperature and pressure must be finite and positive')
    if (Y_in.shape != (mech.n_species,) or not np.all(np.isfinite(Y_in))
            or np.any(Y_in < 0.0) or not np.isclose(Y_in.sum(), 1.0, atol=1e-10, rtol=0)):
        raise ValueError('Equilibrium requires normalized nonnegative mass fractions')
    W_in = float(thermo.mean_molecular_weight(Y_in))
    n_in = Y_in * W_in / mech.molecular_weights
    atom_totals = mech.atom_matrix @ n_in
    h_target = float(np.dot(Y_in, thermo.partial_molar_enthalpies(T_in) / mech.molecular_weights))
    def energy_error(T: float) -> float:
        n = _equilibrium_moles(mech, thermo, T, P, atom_totals)
        h = thermo.partial_molar_enthalpies(T)
        return float(np.dot(n, h) / np.dot(n, mech.molecular_weights) - h_target)
    T_eq = brentq(energy_error, float(mech.min_temperature),
                  float(mech.max_temperature), xtol=1e-7, rtol=1e-10)
    n_eq = _equilibrium_moles(mech, thermo, T_eq, P, atom_totals)
    Y_eq = n_eq * mech.molecular_weights / np.dot(n_eq, mech.molecular_weights)
    return T_eq, Y_eq, float(thermo.density(T_eq, P, Y_eq))
