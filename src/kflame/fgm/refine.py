"""Propose certified bridge flamelets from a tabulated FGM interpolation defect.

The procedure never manufactures table rows.  It reads an already certified
FGM table, estimates a leave-one-out defect at its interior flamelets, and
places one bridge at the logarithmic midpoint of every interval whose local
defect exceeds the selected tolerance.  The resulting JSON is consumed by the
native and Cantera generators; the bridge rows are solved and certified before
they enter a new table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kflame.fgm.common import _relative_l2, _temperature_dynamic_l2


def _defects(table: dict[str, np.ndarray]) -> tuple[np.ndarray, list[dict[str, float]]]:
    phi = np.asarray(table["phi_grid"], dtype=float)
    z = np.asarray(table["Z_grid"], dtype=float)
    temperature = np.asarray(table["T"], dtype=float)
    qdot = np.asarray(table["qdot"], dtype=float)
    omega = np.asarray(table["omega_c"], dtype=float)
    species = np.asarray(table["Y"], dtype=float)
    interval_score = np.zeros(phi.size - 1, dtype=float)
    diagnostics: list[dict[str, float]] = []

    for i in range(1, phi.size - 1):
        weight = float((z[i] - z[i - 1]) / (z[i + 1] - z[i - 1]))
        predict = lambda values: (1.0 - weight) * values[i - 1] + weight * values[i + 1]
        metrics = {
            "temperature_dynamic_l2": _temperature_dynamic_l2(predict(temperature), temperature[i]),
            "qdot_l2": _relative_l2(predict(qdot), qdot[i]),
            "omega_c_l2": _relative_l2(predict(omega), omega[i]),
            "species_l2": _relative_l2(predict(species), species[i]),
        }
        score = float(max(metrics.values()))
        interval_score[i - 1] = max(interval_score[i - 1], score)
        interval_score[i] = max(interval_score[i], score)
        diagnostics.append({"phi": float(phi[i]), "score": score, **metrics})
    return interval_score, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Selecciona flamelets puente con defecto leave-one-out en (Z,c)."
    )
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-defect", type=float, default=0.10)
    parser.add_argument("--max-bridges", type=int, default=4)
    args = parser.parse_args()

    if args.target_defect <= 0.0:
        raise SystemExit("--target-defect debe ser positivo.")
    if args.max_bridges < 1:
        raise SystemExit("--max-bridges debe ser al menos uno.")

    with np.load(args.table, allow_pickle=True) as source:
        table = {key: source[key] for key in source.files}
    phi = np.asarray(table["phi_grid"], dtype=float)
    if phi.size < 3 or np.any(phi <= 0.0) or np.any(np.diff(phi) <= 0.0):
        raise SystemExit("La tabla necesita al menos tres phi positivos y crecientes.")

    interval_score, diagnostics = _defects(table)
    selected = np.flatnonzero(interval_score > args.target_defect)
    if selected.size > args.max_bridges:
        selected = selected[np.argsort(interval_score[selected])[-args.max_bridges:]]
    selected = np.sort(selected)
    bridges = np.sqrt(phi[selected] * phi[selected + 1])

    requested_existing = np.asarray(
        table.get("requested", np.ones(phi.size, dtype=bool)), dtype=bool
    )
    bridge_existing = np.asarray(
        table.get("bridge", np.zeros(phi.size, dtype=bool)), dtype=bool
    )
    if requested_existing.shape != phi.shape or bridge_existing.shape != phi.shape:
        raise SystemExit("Las marcas requested/bridge de la tabla no coinciden con phi_grid.")
    schedule: list[tuple[float, bool, bool]] = [
        (float(value), bool(requested), bool(bridge))
        for value, requested, bridge in zip(phi, requested_existing, bridge_existing, strict=True)
    ] + [(float(value), False, True) for value in bridges]
    schedule.sort(key=lambda item: item[0])
    payload = {
        "coordinate": "log(phi)",
        "selection": {
            "method": "leave-one-out maximum relative L2 defect",
            "target_defect": float(args.target_defect),
            "max_bridges": int(args.max_bridges),
            "parent_table": str(args.table.resolve()),
        },
        "phi_requested": [
            float(value) for value, requested in zip(
                phi, requested_existing, strict=True
            ) if requested
        ],
        "phi_resolved": [value for value, _, _ in schedule],
        "requested": [requested for _, requested, _ in schedule],
        "bridge": [bridge for _, _, bridge in schedule],
        "interval_defect": [float(value) for value in interval_score],
        "leave_one_out": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Bridge flamelets: {bridges.tolist()}")
    print(f"Schedule written: {args.output.resolve()}")


if __name__ == "__main__":
    main()
