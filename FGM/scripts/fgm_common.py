"""Shared numerical and parsing utilities for the FGM generators."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np

_MATERIALS = Path(__file__).resolve().parents[2] / 'V2' / 'materiales'
if str(_MATERIALS) not in sys.path:
    sys.path.insert(0, str(_MATERIALS))
from mechanism_data import load_mechanism
from initialization_native import NativeMixture


def parse_progress_weights(text: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    if not text.strip():
        return weights
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                f"Formato inválido en progress-species: '{token}'. "
                "Usa especie:peso."
            )
        species, weight = token.split(":", 1)
        weights[species.strip()] = float(weight.strip())
    return weights


def parse_species_list(text: str) -> list[str]:
    return [species.strip() for species in text.split(",") if species.strip()]


def parse_phi_values(args: argparse.Namespace) -> np.ndarray:
    if args.phi_values.strip():
        values = [
            float(value.strip())
            for value in args.phi_values.split(",")
            if value.strip()
        ]
        return np.array(sorted(values), dtype=float)
    if args.n_phi < 2:
        return np.array([float(args.phi_min)], dtype=float)
    return np.linspace(float(args.phi_min), float(args.phi_max), int(args.n_phi))


def parse_z_values(args: argparse.Namespace) -> np.ndarray:
    if args.z_values.strip():
        values = [
            float(value.strip())
            for value in args.z_values.split(",")
            if value.strip()
        ]
        return np.array(sorted(values), dtype=float)
    if args.n_z < 2:
        return np.array([float(args.z_min)], dtype=float)
    return np.linspace(float(args.z_min), float(args.z_max), int(args.n_z))


def _bilger_z(
    gas: Any,
    phi: float,
    args: argparse.Namespace,
) -> float:
    gas.TP = float(args.T_in), float(args.P)
    gas.set_equivalence_ratio(float(phi), args.fuel, args.oxidizer)
    return float(gas.mixture_fraction(args.fuel, args.oxidizer, basis="mass"))


def compute_bilger_Z(
    phi: float,
    args: argparse.Namespace,
    gas: Any | None = None,
) -> float:
    """Return the Bilger mixture fraction for a premixed composition."""
    return _bilger_z(gas if gas is not None else NativeMixture(load_mechanism(args.mech)), phi, args)


def invert_bilger_Z_to_phi(
    Z_target: float,
    args: argparse.Namespace,
    *,
    phi_lo: float,
    phi_hi: float,
    tol: float = 1e-10,
    max_iter: int = 80,
    gas: Any | None = None,
) -> float:
    """Invert the monotone Bilger ``Z(phi)`` relation on a logarithmic bracket."""
    if not 0.0 <= Z_target <= 1.0:
        raise ValueError(f"Z objetivo fuera de [0,1]: {Z_target}")

    target = float(np.clip(Z_target, 1e-12, 1.0 - 1e-12))
    lower = max(float(phi_lo), 1e-12)
    upper = max(float(phi_hi), lower * 1.001)

    gas = gas if gas is not None else NativeMixture(load_mechanism(args.mech))
    z_lower = _bilger_z(gas, lower, args)
    z_upper = _bilger_z(gas, upper, args)
    if not z_lower <= target <= z_upper:
        raise ValueError(
            "Z objetivo no alcanzable con el bracket de phi actual. "
            f"Z_target={target:.6f}, Z(phi_lo={lower:.3e})={z_lower:.6f}, "
            f"Z(phi_hi={upper:.3e})={z_upper:.6f}"
        )

    for _ in range(int(max_iter)):
        midpoint = float(np.sqrt(lower * upper))
        z_midpoint = _bilger_z(gas, midpoint, args)
        if abs(z_midpoint - target) <= float(tol):
            return midpoint
        if z_midpoint < target:
            lower = midpoint
        else:
            upper = midpoint

    return 0.5 * (lower + upper)


def compute_progress_variable(
    species_names: list[str],
    Y: np.ndarray,
    T: np.ndarray,
    progress_weights: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    species_indices = {species: index for index, species in enumerate(species_names)}
    used: list[str] = []
    beta = np.zeros(Y.shape[1], dtype=float)
    for species, weight in progress_weights.items():
        if species not in species_indices:
            continue
        beta += float(weight) * Y[species_indices[species]]
        used.append(species)

    if not used:
        delta_temperature = float(T[-1] - T[0])
        c = (
            np.linspace(0.0, 1.0, T.size)
            if abs(delta_temperature) < 1e-14
            else (T - T[0]) / delta_temperature
        )
        return np.clip(c, 0.0, 1.0), c.copy(), used

    denominator = float(beta[-1] - beta[0])
    if abs(denominator) < 1e-14:
        delta_temperature = float(T[-1] - T[0])
        c = (
            np.linspace(0.0, 1.0, T.size)
            if abs(delta_temperature) < 1e-14
            else (T - T[0]) / delta_temperature
        )
    else:
        c = (beta - beta[0]) / denominator
    return np.clip(c, 0.0, 1.0), beta, used


def monotonicize_on_c(
    c: np.ndarray,
    fields: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    mask = np.isfinite(c)
    for values in fields.values():
        mask &= np.isfinite(values)
    if int(np.count_nonzero(mask)) < 2:
        raise RuntimeError("No hay puntos finitos suficientes para construir c monótona.")

    order = np.argsort(np.asarray(c[mask], dtype=float))
    c_sorted = np.asarray(c[mask], dtype=float)[order]
    unique_c, inverse = np.unique(c_sorted, return_inverse=True)

    output: dict[str, np.ndarray] = {}
    for key, values in fields.items():
        values_sorted = np.asarray(values[mask], dtype=float)[order]
        accumulated = np.zeros(unique_c.size, dtype=float)
        counts = np.zeros(unique_c.size, dtype=float)
        np.add.at(accumulated, inverse, values_sorted)
        np.add.at(counts, inverse, 1.0)
        output[key] = accumulated / np.maximum(counts, 1.0)

    c_unique = unique_c.copy()
    c_unique[0] = max(0.0, c_unique[0])
    c_unique[-1] = min(1.0, c_unique[-1])

    if c_unique[0] > 0.0:
        c_unique = np.concatenate(([0.0], c_unique))
        for key in output:
            output[key] = np.concatenate(([output[key][0]], output[key]))
    if c_unique[-1] < 1.0:
        c_unique = np.concatenate((c_unique, [1.0]))
        for key in output:
            output[key] = np.concatenate((output[key], [output[key][-1]]))

    return c_unique, output


def build_global_indicator(
    records: list[Any],
    species_names: list[str],
    indicator_species: list[str],
    c_fine: np.ndarray,
    w_grad: float,
    w_conc: float,
    w_temp: float,
    w_qdot: float,
) -> np.ndarray:
    species_indices = {species: index for index, species in enumerate(species_names)}
    selected = [
        species_indices[species]
        for species in indicator_species
        if species in species_indices
    ]
    epsilon = 1e-30
    # A common c-grid must retain a narrow feature even when it appears in a
    # single mixture only.  An arithmetic mean diluted precisely those local
    # structures as the number of flamelets increased; the envelope is the
    # conservative monitor for a shared structured table.
    envelope = np.zeros_like(c_fine)

    for record in records:
        c_unique, output = monotonicize_on_c(
            record.c,
            {"T": record.T, "qdot": record.qdot},
        )
        score = np.zeros_like(c_unique)
        temperature_gradient = np.abs(np.gradient(output["T"], c_unique, edge_order=1))
        score += float(w_temp) * (
            temperature_gradient / (np.max(temperature_gradient) + epsilon)
        )
        heat_release = np.abs(output["qdot"])
        score += float(w_qdot) * (
            heat_release / (np.max(heat_release) + epsilon)
        )
        for index in selected:
            c_species, species_output = monotonicize_on_c(
                record.c,
                {"Y": record.Y[index]},
            )
            mass_fraction = species_output["Y"]
            species_gradient = np.abs(
                np.gradient(mass_fraction, c_species, edge_order=1)
            )
            local_score = float(w_grad) * (
                species_gradient / (np.max(species_gradient) + epsilon)
            ) + float(w_conc) * (
                mass_fraction / (np.max(mass_fraction) + epsilon)
            )
            score += np.interp(
                c_unique,
                c_species,
                local_score,
                left=local_score[0],
                right=local_score[-1],
            )
        if np.max(score) > 0.0:
            score /= np.max(score)
        local_indicator = np.interp(
            c_fine, c_unique, score, left=score[0], right=score[-1]
        )
        np.maximum(envelope, local_indicator, out=envelope)

    return envelope if records else np.ones_like(c_fine)


def build_adaptive_c_grid(
    c_fine: np.ndarray,
    indicator: np.ndarray,
    n_c: int,
    bias: float,
) -> np.ndarray:
    n_c = int(max(8, n_c))
    weights = 1.0 + float(bias) * np.maximum(indicator, 0.0)
    dc = np.diff(c_fine)
    midpoint_weights = 0.5 * (weights[:-1] + weights[1:])
    cdf = np.concatenate(([0.0], np.cumsum(midpoint_weights * dc)))
    total = float(cdf[-1])
    if total <= 0.0:
        return np.linspace(0.0, 1.0, n_c)
    cdf /= total
    adaptive = np.interp(np.linspace(0.0, 1.0, n_c), cdf, c_fine)
    uniform = np.linspace(0.0, 1.0, max(10, n_c // 6))
    combined = np.unique(np.concatenate(([0.0], adaptive, uniform, [1.0])))
    if combined.size != n_c:
        combined = np.interp(
            np.linspace(0.0, 1.0, n_c),
            np.linspace(0.0, 1.0, combined.size),
            combined,
        )
    combined[0] = 0.0
    combined[-1] = 1.0
    return combined


def validate_fgm_table(
    table: dict[str, np.ndarray],
    *,
    mass_tolerance: float = 1.0e-6,
    species_tolerance: float = 1.0e-8,
) -> dict[str, float | int | bool]:
    """Validate the numerical contract of a structured ``(Z, c)`` table.

    This check runs before persistence so a completed generator cannot silently
    publish an array with swapped axes, non-finite values, invalid endpoints, or
    a composition that no longer sums to one after interpolation.
    """

    required = {
        "phi_grid",
        "Z_grid",
        "c_grid",
        "Su",
        "T",
        "u",
        "rho",
        "cp_mass",
        "conductivity",
        "qdot",
        "omega_c",
        "beta",
        "Y",
    }
    missing = sorted(required.difference(table))
    if missing:
        raise ValueError(f"La tabla FGM no contiene los campos requeridos: {missing}")

    phi = np.asarray(table["phi_grid"], dtype=float)
    z_grid = np.asarray(table["Z_grid"], dtype=float)
    c_grid = np.asarray(table["c_grid"], dtype=float)
    if phi.ndim != 1 or z_grid.ndim != 1 or c_grid.ndim != 1:
        raise ValueError("Los ejes phi, Z y c deben ser unidimensionales.")
    if phi.size != z_grid.size or phi.size < 1 or c_grid.size < 2:
        raise ValueError("Las dimensiones de los ejes FGM no son compatibles.")
    if not (np.all(np.isfinite(phi)) and np.all(np.isfinite(z_grid)) and np.all(np.isfinite(c_grid))):
        raise ValueError("Los ejes FGM contienen NaN o infinitos.")
    if phi.size > 1 and (np.any(np.diff(phi) <= 0.0) or np.any(np.diff(z_grid) <= 0.0)):
        raise ValueError("Los ejes phi y Z deben ser estrictamente crecientes.")
    if np.any(np.diff(c_grid) <= 0.0) or c_grid[0] != 0.0 or c_grid[-1] != 1.0:
        raise ValueError("El eje c debe ser estricto y tener extremos exactos 0 y 1.")

    nz, nc = phi.size, c_grid.size
    su = np.asarray(table["Su"], dtype=float)
    if su.shape != (nz,) or not np.all(np.isfinite(su)):
        raise ValueError("Su debe tener forma (N_Z,) y valores finitos.")

    scalar_fields = (
        "T",
        "u",
        "rho",
        "cp_mass",
        "conductivity",
        "qdot",
        "omega_c",
        "beta",
    )
    arrays: dict[str, np.ndarray] = {}
    for field in scalar_fields:
        values = np.asarray(table[field], dtype=float)
        if values.shape != (nz, nc):
            raise ValueError(
                f"{field} tiene forma {values.shape}; se esperaba {(nz, nc)}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{field} contiene NaN o infinitos.")
        arrays[field] = values

    for field in ("T", "rho", "cp_mass", "conductivity"):
        if np.any(arrays[field] <= 0.0):
            raise ValueError(f"{field} debe ser estrictamente positivo.")

    mass_fractions = np.asarray(table["Y"], dtype=float)
    if mass_fractions.ndim != 3 or mass_fractions.shape[0] != nz or mass_fractions.shape[2] != nc:
        raise ValueError(
            "Y debe usar el orden (N_Z, N_especies, N_c); "
            f"se recibió {mass_fractions.shape}."
        )
    if not np.all(np.isfinite(mass_fractions)):
        raise ValueError("Y contiene NaN o infinitos.")
    min_species = float(np.min(mass_fractions))
    if min_species < -float(species_tolerance):
        raise ValueError(
            f"La fracción másica mínima ({min_species:.3e}) excede la tolerancia."
        )
    mass_sum_error = float(
        np.max(np.abs(np.sum(mass_fractions, axis=1) - 1.0))
    )
    if mass_sum_error > float(mass_tolerance):
        raise ValueError(
            f"El error máximo de suma de fracciones ({mass_sum_error:.3e}) "
            f"excede {mass_tolerance:.3e}."
        )

    for flag in ("solve_ok", "final_accepted"):
        if flag in table:
            values = np.asarray(table[flag], dtype=bool)
            if values.shape != (nz,) or not np.all(values):
                raise ValueError(f"Todos los flamelets deben satisfacer {flag}.")

    return {
        "valid": True,
        "n_Z": int(nz),
        "n_c": int(nc),
        "n_species": int(mass_fractions.shape[1]),
        "max_mass_fraction_sum_error": mass_sum_error,
        "min_mass_fraction": min_species,
        "min_temperature_K": float(np.min(arrays["T"])),
        "min_density_kg_m3": float(np.min(arrays["rho"])),
        "min_conductivity_W_mK": float(np.min(arrays["conductivity"])),
    }
