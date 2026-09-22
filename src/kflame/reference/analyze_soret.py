"""Paired timing and spatially weighted, aligned Soret profile diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from kflame.benchmarks.soret import elemental_flux_diagnostics


def crossing(z, t, target):
    hits = np.flatnonzero((t[:-1] <= target) & (t[1:] >= target))
    if not hits.size:
        raise ValueError("No rising temperature crossing for alignment")
    j = hits[0]
    return z[j] + (target - t[j]) / (t[j + 1] - t[j]) * (z[j + 1] - z[j])


def main():
    import cantera as ct  # This entry point compares native and Cantera runs.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    folder = args.directory
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    records = summary["runs"]
    grouped = {name: {r["repetition"]: r for r in records if r["solver"] == name} for name in ("native", "cantera")}
    pairs = sorted(set(grouped["native"]) & set(grouped["cantera"]))
    # A failed pair is a failure, never silently removed from a speed claim.
    complete_pairs = bool(pairs) and set(grouped["native"]) == set(grouped["cantera"])
    all_accepted = complete_pairs and all(grouped[name][i]["accepted"] for name in grouped for i in pairs)
    statistics = dict(n_pairs=len(pairs), all_pairs_accepted=all_accepted)
    if all_accepted and len(pairs) >= 2:
        native = np.array([grouped["native"][i]["time_s"] for i in pairs])
        cantera = np.array([grouped["cantera"][i]["time_s"] for i in pairs])
        rng = np.random.default_rng(20260904)
        samples = rng.integers(0, len(pairs), size=(20000, len(pairs)))
        boot_ratio = np.median(native[samples], axis=1) / np.median(cantera[samples], axis=1)
        statistics.update(
            native_median_s=float(np.median(native)), cantera_median_s=float(np.median(cantera)),
            native_iqr_s=np.quantile(native, [.25, .75]).tolist(),
            cantera_iqr_s=np.quantile(cantera, [.25, .75]).tolist(),
            paired_ratio_ci95=np.quantile(boot_ratio, [.025, .975]).tolist(),
            reduction_percent=float(100 * (1 - np.median(native) / np.median(cantera))),
            first_pair_included=True,
        )
    if not all_accepted:
        raise ValueError("Incomplete or rejected runs: no certified profile/timing comparison can be claimed")
    case = summary["case"]
    gas = ct.Solution(case["mech"])
    fields = {}
    for name in ("native", "cantera"):
        with np.load(folder / f"{name}_0.npz") as data:
            fields[name] = {k: data[k].copy() for k in data.files}
        f = fields[name]
        heat = np.empty(len(f["z"]))
        for j in range(len(heat)):
            gas.TPY = f["T"][j], case["P"], f["Y"][:, j]
            heat[j] = gas.heat_release_rate
        f["qdot"] = heat
    target = case["T_in"] + .25 * (min(f["T"].max() for f in fields.values()) - case["T_in"])
    alignment = {}
    for name, f in fields.items():
        offset = crossing(f["z"], f["T"], target)
        f["aligned_z"] = f["z"] - offset
        gradient = np.diff(f["T"]) / np.diff(f["z"])
        alignment[name] = dict(
            offset_m=float(offset), thermal_thickness_m=float((f["T"].max() - f["T"][0]) / np.max(np.abs(gradient))),
            heat_release_peak=float(f["qdot"].max()),
            heat_release_peak_aligned_m=float(f["aligned_z"][np.argmax(f["qdot"])]),
            burned_temperature=float(f["T"][-1]),
            relative_edge_gradients=(np.abs(gradient[[0, -1]]) / np.max(np.abs(gradient))).tolist(),
        )
    left = max(f["aligned_z"][0] for f in fields.values())
    right = min(f["aligned_z"][-1] for f in fields.values())
    # Union of both meshes; integral norms do not overweight densely spaced nodes.
    z = np.unique(np.concatenate([f["aligned_z"] for f in fields.values()] + [np.array([left, right])]))
    z = z[(z >= left) & (z <= right)]
    errors = {}
    active = np.flatnonzero(np.max(fields["cantera"]["Y"], axis=1) >= 1e-5)
    for label in ["T", "qdot"] + [gas.species_names[k] for k in active]:
        values = []
        for f in fields.values():
            q = f[label] if label in ("T", "qdot") else f["Y"][gas.species_index(label)]
            values.append(np.interp(z, f["aligned_z"], q))
        actual, reference = values
        denominator = max(float(np.trapezoid(reference ** 2, z)), 1e-300)
        errors[label] = dict(
            E2=float(np.sqrt(np.trapezoid((actual - reference) ** 2, z) / denominator)),
            max_absolute=float(np.max(np.abs(actual - reference))),
        )
    result = dict(timing=statistics, alignment=alignment, aligned_profile_errors=errors,
                  heat_release_diagnostic="Cantera postprocessing of both saved compositions; excluded from solve times",
                  speed_relative_difference=abs(fields["native"]["u"][0] / fields["cantera"]["u"][0] - 1),
                  active_species_threshold=1e-5,
                  limitations="Adaptive acceptance and local edge gradients are not an independent mesh/domain study.")
    if case["transport_model"] == "multicomponent":
        result["elemental_flux"] = {name: elemental_flux_diagnostics(case, f, name) for name, f in fields.items()}
    (folder / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
