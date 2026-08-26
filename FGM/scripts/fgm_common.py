"""Shared numerical and parsing utilities for the FGM generators."""

from __future__ import annotations

import argparse
from typing import Any

import cantera as ct
import numpy as np


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
    gas: ct.Solution,
    phi: float,
    args: argparse.Namespace,
) -> float:
    gas.TP = float(args.T_in), float(args.P)
    gas.set_equivalence_ratio(float(phi), args.fuel, args.oxidizer)
    return float(gas.mixture_fraction(args.fuel, args.oxidizer, basis="mass"))


def compute_bilger_Z(
    phi: float,
    args: argparse.Namespace,
    gas: ct.Solution | None = None,
) -> float:
    """Return the Bilger mixture fraction for a premixed composition."""
    return _bilger_z(gas or ct.Solution(args.mech), phi, args)


def invert_bilger_Z_to_phi(
    Z_target: float,
    args: argparse.Namespace,
    *,
    phi_lo: float,
    phi_hi: float,
    tol: float = 1e-10,
    max_iter: int = 80,
    gas: ct.Solution | None = None,
) -> float:
    """Invert the monotone Bilger ``Z(phi)`` relation on a logarithmic bracket."""
    if not 0.0 <= Z_target <= 1.0:
        raise ValueError(f"Z objetivo fuera de [0,1]: {Z_target}")

    target = float(np.clip(Z_target, 1e-12, 1.0 - 1e-12))
    lower = max(float(phi_lo), 1e-12)
    upper = max(float(phi_hi), lower * 1.001)

    gas = gas or ct.Solution(args.mech)
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
    accumulated = np.zeros_like(c_fine)

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
        accumulated += np.interp(
            c_fine,
            c_unique,
            score,
            left=score[0],
            right=score[-1],
        )

    return accumulated / float(len(records)) if records else np.ones_like(c_fine)


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
