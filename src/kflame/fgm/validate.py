"""Validate an FGM table against independently solved retained flamelets.

The retained flamelets must not be rows of the construction table.  Both tables
use the same normalized progress coordinate, but this script deliberately
interpolates only the construction table before comparing every requested field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from kflame.fgm.common import _relative_l2, _temperature_dynamic_l2


SCALAR_FIELDS = ("T", "rho", "cp_mass", "qdot", "omega_c")


def _interpolate_scalar(
    source: dict[str, np.ndarray],
    field: str,
    z_value: float,
    c_target: np.ndarray,
) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (source["Z_grid"], source["c_grid"]),
        source[field],
        method="linear",
        bounds_error=True,
    )
    points = np.column_stack((np.full(c_target.size, z_value), c_target))
    return np.asarray(interpolator(points), dtype=float)


def _interpolate_species(
    source: dict[str, np.ndarray],
    z_value: float,
    c_target: np.ndarray,
) -> np.ndarray:
    # Stored layout is (Z, species, c); interpolator expects leading grid axes.
    values = np.moveaxis(source["Y"], 1, -1)
    interpolator = RegularGridInterpolator(
        (source["Z_grid"], source["c_grid"]),
        values,
        method="linear",
        bounds_error=True,
    )
    points = np.column_stack((np.full(c_target.size, z_value), c_target))
    return np.asarray(interpolator(points), dtype=float).T


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara una tabla FGM con flamelets de retención independientes."
    )
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--holdouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--active-species-peak",
        type=float,
        default=1.0e-5,
        help="Pico mínimo Y para informar una especie como activa.",
    )
    args = parser.parse_args()

    source = _load(args.table)
    retained = _load(args.holdouts)
    source_phi = np.asarray(source["phi_grid"], dtype=float)
    retained_phi = np.asarray(retained["phi_grid"], dtype=float)
    source_z = np.asarray(source["Z_grid"], dtype=float)
    retained_z = np.asarray(retained["Z_grid"], dtype=float)
    if np.any(np.isclose(retained_phi[:, None], source_phi[None, :], rtol=0.0, atol=1e-12)):
        raise SystemExit("Un flamelet de retención coincide con una fila de construcción.")
    if not np.all(np.asarray(retained["final_accepted"], dtype=bool)):
        raise SystemExit("No todos los flamelets de retención están certificados.")

    names = [str(value) for value in source["species_names"]]
    if names != [str(value) for value in retained["species_names"]]:
        raise SystemExit("Las especies de tabla y retención no coinciden.")

    rows: list[dict[str, object]] = []
    for index, (phi_value, z_value) in enumerate(zip(retained_phi, retained_z, strict=True)):
        if not source_z[0] < z_value < source_z[-1]:
            raise SystemExit(f"Z={z_value:.8g} queda fuera de la tabla de construcción.")
        c_target = np.asarray(retained["c_grid"], dtype=float)
        result: dict[str, object] = {
            "phi": float(phi_value),
            "Z": float(z_value),
            "Su_relative_error": float(
                abs(np.interp(z_value, source_z, source["Su"]) - retained["Su"][index])
                / abs(retained["Su"][index])
            ),
        }
        for field in SCALAR_FIELDS:
            prediction = _interpolate_scalar(source, field, z_value, c_target)
            reference = np.asarray(retained[field][index], dtype=float)
            result[f"{field}_relative_L2"] = _relative_l2(prediction, reference)
            if field == "T":
                result["T_dynamic_relative_L2"] = _temperature_dynamic_l2(
                    prediction, reference
                )

        y_prediction = _interpolate_species(source, z_value, c_target)
        y_reference = np.asarray(retained["Y"][index], dtype=float)
        active_errors = {
            names[k]: _relative_l2(y_prediction[k], y_reference[k])
            for k in range(len(names))
            if float(np.max(y_reference[k])) >= float(args.active_species_peak)
        }
        worst_name = max(active_errors, key=active_errors.get)
        result["active_species_worst"] = worst_name
        result["active_species_worst_relative_L2"] = active_errors[worst_name]
        result["mass_sum_error"] = float(np.max(np.abs(np.sum(y_prediction, axis=0) - 1.0)))
        rows.append(result)

    max_keys = (
        "Su_relative_error",
        "T_dynamic_relative_L2",
        "rho_relative_L2",
        "cp_mass_relative_L2",
        "qdot_relative_L2",
        "omega_c_relative_L2",
        "active_species_worst_relative_L2",
        "mass_sum_error",
    )
    maximum = {key: float(max(row[key] for row in rows)) for key in max_keys}
    payload = {
        "method": "independent retained flamelets; linear table interpolation in (Z,c)",
        "construction_table": str(args.table.resolve()),
        "holdout_table": str(args.holdouts.resolve()),
        "n_holdouts": len(rows),
        "active_species_peak": float(args.active_species_peak),
        "all_retained_certified": True,
        "per_holdout": rows,
        "maximum": maximum,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Validated {len(rows)} independent retained flamelets.")
    for key, value in maximum.items():
        print(f"{key}: {value:.6e}")


if __name__ == "__main__":
    main()
