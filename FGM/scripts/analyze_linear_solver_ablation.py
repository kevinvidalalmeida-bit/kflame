"""Summarize the certified block-substitution performance ablation.

The script deliberately compares complete five-flame sweeps.  It does not use
the isolated random-matrix benchmark as evidence for end-to-end acceleration.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PAIR_NAMES = (
    ("lapack_substitution_1", "compiled_substitution_1"),
    ("lapack_substitution_2", "compiled_substitution_2"),
)


def _load_run(path: Path) -> dict:
    table_path = path / "fgm_table.npz"
    summary_path = path / "summary_phi.csv"
    metadata_path = path / "metadata.json"
    if not table_path.exists() or not summary_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Incomplete ablation run: {path}")

    # The table also contains object arrays with species names.  These files are
    # generated locally by the two benchmark entry points and are therefore a
    # trusted input to this reproducibility helper.
    with np.load(table_path, allow_pickle=True) as data:
        table = {key: np.asarray(data[key]) for key in data.files}
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        summary = list(csv.DictReader(handle))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "path": str(path.resolve()),
        "table": table,
        "summary": summary,
        "metadata": metadata,
        "solve_time_s": float(np.sum(table["solve_time"])),
    }


def analyze(root: Path) -> dict:
    root = root.resolve()
    pairs = []
    for fallback_name, compiled_name in PAIR_NAMES:
        fallback = _load_run(root / fallback_name)
        compiled = _load_run(root / compiled_name)
        fallback_table = fallback["table"]
        compiled_table = compiled["table"]
        if not np.array_equal(fallback_table["phi_grid"], compiled_table["phi_grid"]):
            raise ValueError(f"Incompatible phi grids in pair {fallback_name}/{compiled_name}")

        fallback_time = fallback["solve_time_s"]
        compiled_time = compiled["solve_time_s"]
        pairs.append(
            {
                "fallback_run": fallback_name,
                "compiled_run": compiled_name,
                "fallback_solve_time_s": fallback_time,
                "compiled_solve_time_s": compiled_time,
                "speedup": fallback_time / compiled_time,
                "reduction_percent": 100.0 * (1.0 - compiled_time / fallback_time),
                "same_node_counts": bool(
                    np.array_equal(
                        fallback_table["n_points"], compiled_table["n_points"]
                    )
                ),
                "max_abs_Su_m_per_s": float(
                    np.max(np.abs(fallback_table["Su"] - compiled_table["Su"]))
                ),
                "max_abs_T_K": float(
                    np.max(np.abs(fallback_table["T"] - compiled_table["T"]))
                ),
                "max_abs_Y": float(
                    np.max(np.abs(fallback_table["Y"] - compiled_table["Y"]))
                ),
                "all_compiled_flames_accepted": bool(
                    np.all(compiled_table["final_accepted"])
                ),
            }
        )

    fallback_times = np.array(
        [item["fallback_solve_time_s"] for item in pairs], dtype=float
    )
    compiled_times = np.array(
        [item["compiled_solve_time_s"] for item in pairs], dtype=float
    )
    fallback_median = float(np.median(fallback_times))
    compiled_median = float(np.median(compiled_times))

    cantera = _load_run(root / "cantera_current_1")
    v2_before = _load_run(root / "compiled_substitution_2")
    v2_after = _load_run(root / "compiled_substitution_3")
    bracketed_v2_time = float(
        np.median([v2_before["solve_time_s"], v2_after["solve_time_s"]])
    )
    cantera_table = cantera["table"]
    v2_table = v2_after["table"]
    flame_speed_relative_error = np.abs(v2_table["Su"] - cantera_table["Su"]) / np.maximum(
        np.abs(cantera_table["Su"]), 1.0e-300
    )

    return {
        "scope": "complete certified five-flame sweeps",
        "pairs": pairs,
        "paired_summary": {
            "n_pairs": len(pairs),
            "fallback_median_solve_time_s": fallback_median,
            "compiled_median_solve_time_s": compiled_median,
            "speedup": fallback_median / compiled_median,
            "reduction_percent": 100.0 * (1.0 - compiled_median / fallback_median),
            "max_abs_Su_m_per_s": max(item["max_abs_Su_m_per_s"] for item in pairs),
            "max_abs_T_K": max(item["max_abs_T_K"] for item in pairs),
            "max_abs_Y": max(item["max_abs_Y"] for item in pairs),
            "all_compiled_flames_accepted": all(
                item["all_compiled_flames_accepted"] for item in pairs
            ),
        },
        "current_cantera_bracket": {
            "design": "one Cantera sweep bracketed by two compiled V2 sweeps",
            "v2_before_solve_time_s": v2_before["solve_time_s"],
            "cantera_solve_time_s": cantera["solve_time_s"],
            "v2_after_solve_time_s": v2_after["solve_time_s"],
            "v2_bracket_median_solve_time_s": bracketed_v2_time,
            "speedup_cantera_over_v2": cantera["solve_time_s"] / bracketed_v2_time,
            "max_relative_flame_speed_error_percent": float(
                100.0 * np.max(flame_speed_relative_error)
            ),
            "all_v2_flames_accepted": bool(np.all(v2_table["final_accepted"])),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("FGM/resultados/revision_20260903_blocksolve_ablation"),
    )
    args = parser.parse_args()
    result = analyze(args.root)
    output_json = args.root / "linear_solver_ablation.json"
    output_csv = args.root / "linear_solver_ablation.csv"
    output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["pairs"][0].keys()))
        writer.writeheader()
        writer.writerows(result["pairs"])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
