"""Evaluate the pre-registered predictor--corrector promotion criteria.

Pass paired runs in their execution order.  The script compares only requested
flamelets, so adaptive bridge rows improve FGM resolution without contaminating
the physical-equivalence check against the fixed-region baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_run(path: Path) -> dict:
    path = path.resolve()
    with np.load(path / "fgm_table.npz", allow_pickle=True) as raw:
        table = {key: np.asarray(raw[key]) for key in raw.files}
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    trace_path = path / str(meta.get("continuation_trace", "continuation_trace.json"))
    trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else []
    return {"path": str(path), "table": table, "metadata": meta, "trace": trace}


def _requested_indices(table: dict[str, np.ndarray]) -> np.ndarray:
    if "requested" not in table:
        return np.arange(table["phi_grid"].size, dtype=int)
    return np.flatnonzero(np.asarray(table["requested"], dtype=bool))


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(
        np.linalg.norm(candidate - reference)
        / max(float(np.linalg.norm(reference)), 1.0e-300)
    )


def _active_species_indices(
    table: dict[str, np.ndarray], active_species: tuple[str, ...] | None
) -> list[int]:
    names = tuple(str(name) for name in np.asarray(table["species_names"]).tolist())
    if active_species is None:
        return list(range(len(names)))
    selected = [names.index(name) for name in active_species if name in names]
    if not selected:
        raise ValueError("None of the requested active species are present in the table.")
    return selected


def _compare_pair(
    fixed: dict, adaptive: dict, active_species: tuple[str, ...] | None
) -> dict:
    left, right = fixed["table"], adaptive["table"]
    fixed_idx, adaptive_idx = _requested_indices(left), _requested_indices(right)
    phi_fixed, phi_adaptive = left["phi_grid"][fixed_idx], right["phi_grid"][adaptive_idx]
    if not np.array_equal(phi_fixed, phi_adaptive):
        raise ValueError("Paired runs do not contain the same requested phi values.")
    if not np.array_equal(left["species_names"], right["species_names"]):
        raise ValueError("Paired runs do not use the same ordered species list.")
    active_indices = _active_species_indices(left, active_species)

    rows = []
    for i, j in zip(fixed_idx, adaptive_idx, strict=True):
        c_ref, c_candidate = left["c_grid"], right["c_grid"]
        T_adapt = np.interp(c_ref, c_candidate, right["T"][j])
        q_adapt = np.interp(c_ref, c_candidate, right["qdot"][j])
        Y_adapt = np.vstack([
            np.interp(c_ref, c_candidate, right["Y"][j, species])
            for species in range(right["Y"].shape[1])
        ])
        rows.append({
            "phi": float(left["phi_grid"][i]),
            "Su_relative_error": float(
                abs(right["Su"][j] - left["Su"][i])
                / max(abs(left["Su"][i]), 1.0e-300)
            ),
            "T_l2_relative": _relative_l2(left["T"][i], T_adapt),
            "qdot_l2_relative": _relative_l2(left["qdot"][i], q_adapt),
            "Y_l2_relative_max": max(
                _relative_l2(left["Y"][i, species], Y_adapt[species])
                for species in active_indices
            ),
        })
    fixed_time = float(np.sum(left["solve_time"]))
    adaptive_time = float(np.sum(right["solve_time"]))
    return {
        "fixed_path": fixed["path"],
        "adaptive_path": adaptive["path"],
        "fixed_solve_time_s": fixed_time,
        "adaptive_solve_time_s": adaptive_time,
        "speedup": fixed_time / max(adaptive_time, 1.0e-300),
        "fixed_all_accepted": bool(np.all(left["final_accepted"])),
        "adaptive_all_accepted": bool(np.all(right["final_accepted"])),
        "adaptive_bridge_rows": int(np.sum(np.asarray(right.get("bridge", []), dtype=bool))),
        "adaptive_trace_rejections": int(sum(not item.get("accepted", False) for item in adaptive["trace"])),
        "active_species": [
            str(np.asarray(left["species_names"])[species]) for species in active_indices
        ],
        "requested_rows": rows,
    }


def _bootstrap_median_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for index in range(samples):
        draws[index] = float(np.median(generator.choice(values, size=values.size, replace=True)))
    return tuple(float(value) for value in np.quantile(draws, (0.025, 0.975)))


def _iqr(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, (0.25, 0.75))]


def analyse(
    fixed_paths: list[Path],
    adaptive_paths: list[Path],
    samples: int,
    seed: int,
    active_species: tuple[str, ...] | None,
) -> dict:
    if len(fixed_paths) != len(adaptive_paths) or not fixed_paths:
        raise ValueError("Provide the same non-zero number of fixed and adaptive runs.")
    pairs = [_compare_pair(_load_run(fixed), _load_run(adaptive), active_species)
             for fixed, adaptive in zip(fixed_paths, adaptive_paths, strict=True)]
    speedups = np.array([pair["speedup"] for pair in pairs], dtype=float)
    fixed_times = np.array([pair["fixed_solve_time_s"] for pair in pairs], dtype=float)
    adaptive_times = np.array([pair["adaptive_solve_time_s"] for pair in pairs], dtype=float)
    all_rows = [row for pair in pairs for row in pair["requested_rows"]]
    max_metric = {
        "Su_relative_error": max(row["Su_relative_error"] for row in all_rows),
        "T_l2_relative": max(row["T_l2_relative"] for row in all_rows),
        "Y_l2_relative_max": max(row["Y_l2_relative_max"] for row in all_rows),
        "qdot_l2_relative": max(row["qdot_l2_relative"] for row in all_rows),
    }
    ci_low, ci_high = _bootstrap_median_ci(speedups, samples=samples, seed=seed)
    promotion = {
        "time_ci_favors_adaptive": bool(ci_low > 1.0),
        "certification_not_worse": bool(
            np.mean([pair["adaptive_all_accepted"] for pair in pairs])
            >= np.mean([pair["fixed_all_accepted"] for pair in pairs])
        ),
        "Su_within_0p1_percent": bool(max_metric["Su_relative_error"] <= 1.0e-3),
        "T_within_0p5_percent": bool(max_metric["T_l2_relative"] <= 5.0e-3),
        "Y_within_2_percent": bool(max_metric["Y_l2_relative_max"] <= 2.0e-2),
        "qdot_within_5_percent": bool(max_metric["qdot_l2_relative"] <= 5.0e-2),
    }
    promotion["promote_adaptive_pc"] = bool(all(promotion.values()))
    return {
        "n_pairs": len(pairs),
        "paired_speedup_median": float(np.median(speedups)),
        "paired_speedup_iqr": _iqr(speedups),
        "paired_speedup_bootstrap_95_ci": [ci_low, ci_high],
        "fixed_solve_time_median_s": float(np.median(fixed_times)),
        "fixed_solve_time_iqr_s": _iqr(fixed_times),
        "adaptive_solve_time_median_s": float(np.median(adaptive_times)),
        "adaptive_solve_time_iqr_s": _iqr(adaptive_times),
        "max_requested_physical_difference": max_metric,
        "promotion": promotion,
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", type=Path, nargs="+", required=True)
    parser.add_argument("--adaptive", type=Path, nargs="+", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--active-species",
        type=str,
        default="CH4,O2,CO2,H2O,CO,H2,OH",
        help="Especies separadas por comas para el umbral E2(Y_k); vacio usa todas.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = tuple(
        item.strip() for item in args.active_species.split(",") if item.strip()
    ) or None
    result = analyse(
        args.fixed, args.adaptive, args.bootstrap_samples, args.seed, selected
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["promotion"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
