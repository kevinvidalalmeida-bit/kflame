"""Paired timing and spatially weighted, aligned Soret profile diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cantera as ct
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "materiales"))
from species_backend_native import NativeSpeciesBackend
from equations import _multicomponent_flux


def elemental_flux_diagnostics(case, fields, solver):
    """Reconstruct physical elemental fluxes at faces, not local element constancy.

    Midpoint convection is a diagnostic of continuum conservation; it is not
    identical to the nonuniform-grid upwind residual used by either solver.
    Its error must be assessed under mesh refinement.
    """
    gas = ct.Solution(case["mech"], transport_model="multicomponent")
    gas.TP = case["T_in"], case["P"]
    gas.set_equivalence_ratio(case["phi"], case["fuel"], case["oxidizer"])
    element_mass = np.array([[gas.n_atoms(k, e) * gas.atomic_weight(e) / gas.molecular_weights[k]
                              for k in range(gas.n_species)] for e in range(gas.n_elements)])
    fresh_element = element_mass @ gas.Y
    z, t, y, u = (fields[k] for k in ("z", "T", "Y", "u"))
    w = gas.molecular_weights
    rho = case["P"] / (ct.gas_constant * t * np.sum(y / w[:, None], axis=0))
    tf, yf = .5 * (t[:-1] + t[1:]), .5 * (y[:, :-1] + y[:, 1:])
    if solver == "native":
        native = NativeSpeciesBackend(SimpleNamespace(case=SimpleNamespace(**case), P=case["P"]))
        rf, _lf, wf, dm, dt = native.eval_multicomponent_face_transport(tf, yf)
    else:
        rf, wf = np.empty(tf.size), np.empty(tf.size)
        dm, dt = np.empty((w.size, w.size, tf.size)), np.empty((w.size, tf.size))
        for j in range(tf.size):
            gas.TPY = tf[j], case["P"], yf[:, j]
            rf[j], wf[j] = gas.density, gas.mean_molecular_weight
            dm[:, :, j], dt[:, j] = gas.multi_diff_coeffs, gas.thermal_diff_coeffs
        if not case.get("soret_enabled", False):
            dt.fill(0.0)
    flux = _multicomponent_flux(y[:, :-1], y[:, 1:], t[:-1], t[1:], rf, wf, dm, dt, np.diff(z), w)
    mdot = rho * u
    expected = mdot[0] * fresh_element
    total = element_mass @ (yf * (.5 * (mdot[:-1] + mdot[1:]))[None, :] + flux)
    discrepancy = np.max(np.abs(total - expected[:, None]), axis=1)
    return dict(
        convention="Midpoint reconstructed convective + diffusive elemental flux; mesh-sensitive diagnostic",
        max_diffusive_mass_sum=float(np.max(np.abs(flux.sum(axis=0)))),
        min_mass_fraction=float(y.min()),
        elements={gas.element_names[e]: dict(inlet_flux=float(expected[e]),
                   max_absolute_deviation=float(discrepancy[e]),
                   relative_deviation=float(discrepancy[e] / abs(expected[e])) if abs(expected[e]) > 1e-12 else None)
                  for e in range(gas.n_elements)},
    )


def crossing(z, t, target):
    hits = np.flatnonzero((t[:-1] <= target) & (t[1:] >= target))
    if not hits.size:
        raise ValueError("No rising temperature crossing for alignment")
    j = hits[0]
    return z[j] + (target - t[j]) / (t[j + 1] - t[j]) * (z[j + 1] - z[j])


def main():
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
