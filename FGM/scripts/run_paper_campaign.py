"""Run the P0/P1 evidence campaign without weakening the flame certificate.

The driver deliberately separates three things that are often mixed in a
performance claim:

* P0 measures a cold V2/Cantera pair seven times in alternating order and
  writes physical diagnostics from an un-timed profile;
* P1 records robustness cases with the same acceptance policy; and
* the four continuation strategies are executed as independent, cache-free
  FGM sweeps, so their traces can be inspected or replayed later.

All heavy output belongs below ``FGM/resultados`` (which is ignored by Git).
The compact JSON/CSV summaries are suitable inputs for a versioned evidence
folder after the campaign has been reviewed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cantera as ct
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
V2_DIRECTORY = REPOSITORY / "V2"
if str(V2_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(V2_DIRECTORY))

from config import FlameCase  # noqa: E402
from run_saved_comparison import (  # noqa: E402
    _cantera_solve,
    _resample_v2_seed,
    _resolve_mechanism,
    _v2_solve,
)


ACTIVE_SPECIES = ("CH4", "O2", "CO2", "H2O", "CO", "H2", "OH")
ATM = 101325.0


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _git_value(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=REPOSITORY, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return completed.stdout.strip()


def environment_metadata() -> dict[str, Any]:
    """Return the experimental context without claiming host-specific facts."""
    memory: int | None = None
    try:
        import psutil  # type: ignore

        memory = int(psutil.virtual_memory().total)
    except Exception:
        pass
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_value("rev-parse", "HEAD"),
        "git_status_short": _git_value("status", "--short"),
        "platform": platform.platform(),
        "python": sys.version,
        "cantera": ct.__version__,
        "numpy": np.__version__,
        "cpu_count_logical": os.cpu_count(),
        "memory_bytes": memory,
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS"),
    }


def make_case(
    *,
    fuel: str = "CH4",
    phi: float = 1.0,
    temperature: float = 300.0,
    pressure_atm: float = 1.0,
    width: float = 0.03,
) -> FlameCase:
    return FlameCase(
        mech=_resolve_mechanism("gri30.yaml"),
        fuel=fuel,
        oxidizer="O2:1.0, N2:3.76",
        phi=float(phi),
        T_in=float(temperature),
        P=float(pressure_atm) * ATM,
        width=float(width),
        transport_model="mixture-averaged",
        flux_gradient_basis="molar",
        soret_enabled=False,
        ratio=2.5,
        slope=0.04,
        curve=0.08,
        prune=0.003,
    )


def _thermal_origin(profile: dict[str, Any]) -> float:
    z, temperature = np.asarray(profile["z"]), np.asarray(profile["T"])
    target = float(temperature[0] + 0.5 * (temperature[-1] - temperature[0]))
    index = int(np.argmin(np.abs(temperature - target)))
    if index == 0:
        return float(z[0])
    return float(np.interp(target, temperature[index - 1 : index + 1], z[index - 1 : index + 1]))


def _edge_slopes(profile: dict[str, Any]) -> tuple[float, float]:
    z, temperature = np.asarray(profile["z"]), np.asarray(profile["T"])
    scale = (temperature[-1] - temperature[0]) / max(z[-1] - z[0], np.finfo(float).tiny)
    if abs(scale) < np.finfo(float).tiny:
        return 0.0, 0.0
    left = abs((temperature[1] - temperature[0]) / (z[1] - z[0]) / scale)
    right = abs((temperature[-1] - temperature[-2]) / (z[-1] - z[-2]) / scale)
    return float(left), float(right)


def _thermal_thickness(profile: dict[str, Any]) -> float:
    z, temperature = np.asarray(profile["z"]), np.asarray(profile["T"])
    maximum = float(np.max(np.abs(np.gradient(temperature, z))))
    return abs(float(temperature[-1] - temperature[0])) / max(maximum, np.finfo(float).tiny)


def _interpolate_species(z_source: np.ndarray, values: np.ndarray, z_target: np.ndarray) -> np.ndarray:
    """Interpolate one species profile per row on a strictly ordered mesh."""
    return np.vstack([
        np.interp(z_target, z_source, values[k])
        for k in range(values.shape[0])
    ])


def _pressure_secant_seed(
    *,
    previous: dict[str, Any],
    previous_pressure: float,
    previous_previous: dict[str, Any] | None,
    previous_previous_pressure: float | None,
    target_pressure: float,
    damping: float = 0.7,
) -> tuple[dict[str, Any], str]:
    """Predict a pressure-neighbour state on the latest accepted mesh.

    The two profiles are first expressed on the latest adaptive mesh and the
    secant is formed in ``log(p)``.  This deliberately predicts the state, not
    the source mesh: the bounded mesh transfer that follows is a separate,
    auditable decision.  The first continuation transition has only one
    accepted state and therefore uses a physical copy predictor.
    """
    if target_pressure <= 0.0 or previous_pressure <= 0.0:
        raise ValueError("Pressure continuation requires strictly positive pressures.")
    z_latest = np.asarray(previous["z"], dtype=float)
    u_latest = np.asarray(previous["u"], dtype=float)
    T_latest = np.asarray(previous["T"], dtype=float)
    Y_latest = np.asarray(previous["Y"], dtype=float)
    u_seed, T_seed, Y_seed = u_latest.copy(), T_latest.copy(), Y_latest.copy()
    kind = "copy"

    if previous_previous is not None and previous_previous_pressure is not None:
        delta_log = float(np.log(previous_pressure / previous_previous_pressure))
        if abs(delta_log) > 1.0e-14:
            z_old = np.asarray(previous_previous["z"], dtype=float)
            u_old = np.interp(z_latest, z_old, np.asarray(previous_previous["u"], dtype=float))
            T_old = np.interp(z_latest, z_old, np.asarray(previous_previous["T"], dtype=float))
            Y_old = _interpolate_species(
                z_old, np.asarray(previous_previous["Y"], dtype=float), z_latest,
            )
            extrapolation = float(
                np.clip(
                    float(damping) * np.log(target_pressure / previous_pressure) / delta_log,
                    -0.5,
                    1.0,
                )
            )
            u_seed = u_latest + extrapolation * (u_latest - u_old)
            T_seed = T_latest + extrapolation * (T_latest - T_old)
            Y_seed = Y_latest + extrapolation * (Y_latest - Y_old)
            kind = "secant_log_pressure"

    # Projection is intentionally limited to state feasibility. The complete
    # residual, mesh, domain, and physics certificate remains the corrector's
    # responsibility.
    T_seed = np.clip(T_seed, 200.0, 6000.0)
    Y_seed = np.clip(Y_seed, 0.0, None)
    Y_seed /= np.maximum(np.sum(Y_seed, axis=0, keepdims=True), np.finfo(float).tiny)
    u_seed = np.maximum(u_seed, 1.0e-8)
    return {
        **previous,
        "z": z_latest.copy(),
        "u": u_seed,
        "T": T_seed,
        "Y": Y_seed,
        "x": np.column_stack((u_seed, T_seed, Y_seed.T)).ravel(),
        "n_points": int(z_latest.size),
        "width": float(z_latest[-1] - z_latest[0]),
        "pressure_predictor": kind,
    }, kind


def _pressure_prediction_defect(corrected: dict[str, Any], transmitted_seed: dict[str, Any] | None) -> float:
    """Measure the actual transmitted predictor against its corrected state.

    This is the same component-weighted convention used by V2's Newton
    convergence test: ``rtol * mean(abs(corrected)) + atol``.  The predictor
    is interpolated *after* the corrector chose its final mesh, so a mesh
    change cannot be mistaken for a thermochemical prediction error.
    """
    if transmitted_seed is None:
        return float("nan")
    z_final = np.asarray(corrected["z"], dtype=float)
    z_seed = np.asarray(transmitted_seed["z"], dtype=float)
    u_pred = np.interp(z_final, z_seed, np.asarray(transmitted_seed["u"], dtype=float))
    T_pred = np.interp(z_final, z_seed, np.asarray(transmitted_seed["T"], dtype=float))
    Y_pred = _interpolate_species(z_seed, np.asarray(transmitted_seed["Y"], dtype=float), z_final)
    predicted = np.column_stack((u_pred, T_pred, Y_pred.T))
    corrected_state = np.column_stack((
        np.asarray(corrected["u"], dtype=float),
        np.asarray(corrected["T"], dtype=float),
        np.asarray(corrected["Y"], dtype=float).T,
    ))
    scales = 1.0e-4 * np.mean(np.abs(corrected_state), axis=0) + 1.0e-9
    normalised = (corrected_state - predicted) / np.maximum(scales, np.finfo(float).tiny)
    return float(np.sqrt(np.mean(normalised * normalised)))


def _postprocess(profile: dict[str, Any], case: FlameCase) -> dict[str, np.ndarray]:
    """Evaluate comparison-only fields with one common thermochemical convention."""
    gas = ct.Solution(case.mech)
    temperature, fractions = np.asarray(profile["T"]), np.asarray(profile["Y"])
    rho = np.empty(temperature.size)
    qdot = np.empty(temperature.size)
    element_mass_fraction = np.empty((gas.n_elements, temperature.size))
    molecular_weights = np.asarray(gas.molecular_weights)
    atomic_weights = np.asarray(gas.atomic_weights)
    atom_matrix = np.array(
        [[gas.n_atoms(k, e) for k in range(gas.n_species)] for e in range(gas.n_elements)],
        dtype=float,
    )
    element_coefficients = atom_matrix * atomic_weights[:, None] / molecular_weights[None, :]
    for j, temp in enumerate(temperature):
        gas.TPY = float(temp), case.P, np.asarray(fractions[:, j], dtype=float)
        rho[j] = gas.density
        qdot[j] = gas.heat_release_rate
        element_mass_fraction[:, j] = element_coefficients @ fractions[:, j]
    return {"rho": rho, "qdot": qdot, "element_mass_fraction": element_mass_fraction}


def _aligned_metrics(v2: dict[str, Any], cantera: dict[str, Any], case: FlameCase) -> dict[str, Any]:
    v2_fields, cantera_fields = _postprocess(v2, case), _postprocess(cantera, case)
    z_v2 = np.asarray(v2["z"]) - _thermal_origin(v2)
    z_ct = np.asarray(cantera["z"]) - _thermal_origin(cantera)
    lower, upper = max(z_v2[0], z_ct[0]), min(z_v2[-1], z_ct[-1])
    common = np.linspace(lower, upper, 4001)

    def interpolate(profile: dict[str, Any], source: np.ndarray, field: str) -> np.ndarray:
        values = np.asarray(profile[field])
        if values.ndim == 1:
            return np.interp(common, source, values)
        return np.vstack([np.interp(common, source, row) for row in values])

    T_v2, T_ct = interpolate(v2, z_v2, "T"), interpolate(cantera, z_ct, "T")
    q_v2 = np.interp(common, z_v2, v2_fields["qdot"])
    q_ct = np.interp(common, z_ct, cantera_fields["qdot"])
    Y_v2, Y_ct = interpolate(v2, z_v2, "Y"), interpolate(cantera, z_ct, "Y")
    names = tuple(ct.Solution(case.mech).species_names)
    selected = [names.index(name) for name in ACTIVE_SPECIES if name in names]
    tiny = np.finfo(float).tiny
    mass_v2, mass_ct = v2_fields["rho"] * v2["u"], cantera_fields["rho"] * cantera["u"]

    def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
        return float(np.linalg.norm(candidate - reference) / max(np.linalg.norm(reference), tiny))

    v2_peak = float(common[int(np.argmax(q_v2))])
    ct_peak = float(common[int(np.argmax(q_ct))])
    v2_elements = v2_fields["element_mass_fraction"]
    ct_elements = cantera_fields["element_mass_fraction"]
    return {
        "Su_relative_error": abs(float(v2["Su"] - cantera["Su"])) / max(abs(float(cantera["Su"])), tiny),
        "T_l2_dynamic": float(np.linalg.norm(T_v2 - T_ct) / max(np.linalg.norm(T_ct - T_ct[0]), tiny)),
        "T_linf_K": float(np.max(np.abs(T_v2 - T_ct))),
        "qdot_l2_relative": relative_l2(q_v2, q_ct),
        "qdot_linf_relative": float(np.max(np.abs(q_v2 - q_ct)) / max(np.max(np.abs(q_ct)), tiny)),
        "Y_l2_relative_max_active": max(relative_l2(Y_v2[k], Y_ct[k]) for k in selected),
        "thermal_thickness_v2_m": _thermal_thickness(v2),
        "thermal_thickness_cantera_m": _thermal_thickness(cantera),
        "qdot_peak_offset_v2_minus_cantera_m": v2_peak - ct_peak,
        "mass_flow_relative_span_v2": float(np.ptp(mass_v2) / max(abs(np.mean(mass_v2)), tiny)),
        "mass_flow_relative_span_cantera": float(np.ptp(mass_ct) / max(abs(np.mean(mass_ct)), tiny)),
        "mass_fraction_sum_error_v2": float(np.max(np.abs(np.sum(v2["Y"], axis=0) - 1.0))),
        "mass_fraction_sum_error_cantera": float(np.max(np.abs(np.sum(cantera["Y"], axis=0) - 1.0))),
        # This is a composition diagnostic, not an elemental-flux residual:
        # unequal diffusivities can change elemental mass fractions locally.
        "element_mass_fraction_span_v2": {
            name: float(np.ptp(v2_elements[i])) for i, name in enumerate(ct.Solution(case.mech).element_names)
        },
        "element_mass_fraction_span_cantera": {
            name: float(np.ptp(ct_elements[i])) for i, name in enumerate(ct.Solution(case.mech).element_names)
        },
        "edge_slope_v2": _edge_slopes(v2),
        "edge_slope_cantera": _edge_slopes(cantera),
        "nodes_v2": int(v2["n_points"]),
        "nodes_cantera": int(cantera["n_points"]),
        "width_v2_m": float(v2["width"]),
        "width_cantera_m": float(cantera["width"]),
    }


def _bootstrap_median_ci(values: np.ndarray, samples: int = 10000, seed: int = 20260903) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for index in range(samples):
        draws[index] = np.median(rng.choice(values, size=values.size, replace=True))
    return [float(value) for value in np.quantile(draws, (0.025, 0.975))]


def _summary(values: list[float]) -> dict[str, Any]:
    data = np.asarray(values, dtype=float)
    return {
        "n": int(data.size), "median": float(np.median(data)),
        "iqr": [float(value) for value in np.quantile(data, (0.25, 0.75))],
        "bootstrap_median_95_ci": _bootstrap_median_ci(data),
    }


def _solve_pair(
    case: FlameCase,
    v2_first: bool,
    *,
    profile_v2: bool = False,
    acceptance_criterion: str = "combined",
    final_residual_inf: float = 10.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Solve an independent pair under the complete V2 acceptance predicate."""
    v2_kwargs = {
        "profile": profile_v2,
        "compiled_block_substitution": True,
        "acceptance_criterion": acceptance_criterion,
        "residual_guard_inf": final_residual_inf,
        "final_residual_inf": final_residual_inf,
    }
    if v2_first:
        v2 = _v2_solve(case, **v2_kwargs)
        cantera = _cantera_solve(case)
    else:
        cantera = _cantera_solve(case)
        v2 = _v2_solve(case, **v2_kwargs)
    return v2, cantera


def _row_for_pair(index: int, order: str, v2: dict[str, Any], cantera: dict[str, Any]) -> dict[str, Any]:
    return {
        "replicate": index, "order": order,
        "v2_ok": bool(v2["ok"]), "v2_final_accepted": bool(v2["report"].get("final_accepted", v2["ok"])),
        "v2_time_s": float(v2["time_s"]), "cantera_time_s": float(cantera["time_s"]),
        "speedup_cantera_over_v2": float(cantera["time_s"] / max(v2["time_s"], np.finfo(float).tiny)),
        "Su_v2_m_per_s": float(v2["Su"]), "Su_cantera_m_per_s": float(cantera["Su"]),
        "nodes_v2": int(v2["n_points"]), "nodes_cantera": int(cantera["n_points"]),
        "Finf_v2": float(v2["report"].get("Finf_final", np.nan)),
        "weighted_step_norm_v2": float(v2["report"].get("weighted_step_norm_final", np.nan)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({name for row in rows for name in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_p0(root: Path, repetitions: int, spatial: bool) -> None:
    output = root / "P0_certification"
    output.mkdir(parents=True, exist_ok=True)
    case = make_case()
    _write_json(output / "metadata.json", {"environment": environment_metadata(), "case": asdict(case), "repetitions": repetitions, "warmup_excluded": True})

    # JIT/runtime warm-up is deliberately recorded but excluded from every statistic.
    warm_v2, warm_ct = _solve_pair(case, v2_first=True, profile_v2=False)
    _write_json(output / "warmup_excluded.json", _row_for_pair(-1, "v2_then_cantera", warm_v2, warm_ct))

    rows: list[dict[str, Any]] = []
    saved_v2: dict[str, Any] | None = None
    saved_ct: dict[str, Any] | None = None
    for repetition in range(repetitions):
        v2_first = repetition % 2 == 0
        v2, cantera = _solve_pair(case, v2_first=v2_first, profile_v2=False)
        rows.append(_row_for_pair(repetition + 1, "v2_then_cantera" if v2_first else "cantera_then_v2", v2, cantera))
        saved_v2, saved_ct = v2, cantera
        print(f"P0 replica {repetition + 1}/{repetitions}: V2={v2['time_s']:.3f}s, Cantera={cantera['time_s']:.3f}s")
    _write_csv(output / "paired_timing.csv", rows)

    assert saved_v2 is not None and saved_ct is not None
    diagnostic_v2, diagnostic_ct = _solve_pair(case, v2_first=True, profile_v2=True)
    metrics = _aligned_metrics(diagnostic_v2, diagnostic_ct, case)
    np.savez_compressed(
        output / "diagnostic_profiles.npz",
        z_v2=diagnostic_v2["z"], T_v2=diagnostic_v2["T"], u_v2=diagnostic_v2["u"], Y_v2=diagnostic_v2["Y"],
        z_cantera=diagnostic_ct["z"], T_cantera=diagnostic_ct["T"], u_cantera=diagnostic_ct["u"], Y_cantera=diagnostic_ct["Y"],
        species_names=np.asarray(diagnostic_v2["species_names"], dtype="U"),
    )
    speedups = [row["speedup_cantera_over_v2"] for row in rows]
    result: dict[str, Any] = {
        "timing": {
            "v2_time_s": _summary([row["v2_time_s"] for row in rows]),
            "cantera_time_s": _summary([row["cantera_time_s"] for row in rows]),
            "paired_speedup_cantera_over_v2": _summary(speedups),
            "all_v2_certified": bool(all(row["v2_ok"] and row["v2_final_accepted"] for row in rows)),
        },
        "diagnostic_excluded_from_timing": {
            "metrics": metrics,
            "v2_profile": diagnostic_v2["report"].get("profile", {}),
            "v2_report": diagnostic_v2["report"],
        },
    }

    if spatial:
        mesh_levels = {
            "coarse": (3.5, 0.10, 0.16, 0.020),
            "base": (3.0, 0.08, 0.12, 0.010),
            "strict": (2.5, 0.04, 0.08, 0.003),
            "ultra": (2.0, 0.02, 0.04, 0.001),
        }
        spatial_rows: list[dict[str, Any]] = []
        for name, criteria in mesh_levels.items():
            mesh_case = make_case()
            mesh_case.ratio, mesh_case.slope, mesh_case.curve, mesh_case.prune = criteria
            v2, cantera = _solve_pair(mesh_case, v2_first=True, profile_v2=False)
            row = {"kind": "mesh", "level": name, **_row_for_pair(0, "v2_then_cantera", v2, cantera), **_aligned_metrics(v2, cantera, mesh_case)}
            spatial_rows.append(row)
            print(f"P0 mesh {name}: accepted={row['v2_final_accepted']}")
        for width in (0.015, 0.03, 0.06):
            domain_case = make_case(width=width)
            v2, cantera = _solve_pair(domain_case, v2_first=True, profile_v2=False)
            row = {"kind": "domain", "level": f"width_{width:g}", **_row_for_pair(0, "v2_then_cantera", v2, cantera), **_aligned_metrics(v2, cantera, domain_case)}
            spatial_rows.append(row)
            print(f"P0 domain {width:g} m: accepted={row['v2_final_accepted']}")
        _write_json(output / "spatial_study.json", spatial_rows)
    _write_json(output / "p0_summary.json", result)


def run_p1_matrix(root: Path) -> None:
    """Run the physically supported P1 matrix once, with V2 profiling enabled.

    These are robustness measurements, not a timing comparison: profiling is
    intentionally enabled to retain counts of Jacobian/linear work.
    """
    output = root / "P1_robustness"
    output.mkdir(parents=True, exist_ok=True)
    cases: list[tuple[str, FlameCase]] = []
    for temperature in (300.0, 500.0, 700.0):
        for pressure in (1.0, 5.0, 10.0):
            cases.append((f"CH4_phi1_T{temperature:g}_P{pressure:g}", make_case(temperature=temperature, pressure_atm=pressure)))
    # V2 does not implement Soret.  This is explicitly the mixture-averaged
    # H2 robustness screen, not a claim about preferential-diffusion accuracy.
    for phi in (0.6, 1.0, 1.5):
        cases.append((f"H2_phi{phi:g}_mixture_averaged_no_soret", make_case(fuel="H2", phi=phi)))

    rows: list[dict[str, Any]] = []
    for label, case in cases:
        try:
            v2, cantera = _solve_pair(case, v2_first=True, profile_v2=True)
            metrics = _aligned_metrics(v2, cantera, case)
            row = {"case": label, "status": "completed", **asdict(case), **_row_for_pair(0, "v2_then_cantera", v2, cantera), **metrics}
            row["v2_profile"] = v2["report"].get("profile", {})
        except Exception as error:
            row = {"case": label, "status": "failed", **asdict(case), "error": f"{type(error).__name__}: {error}"}
        rows.append(row)
        print(f"P1 {label}: {row['status']}")
        _write_json(output / "p1_matrix_partial.json", rows)
    _write_json(output / "p1_matrix.json", {"environment": environment_metadata(), "scope": "mixture-averaged; Soret intentionally excluded", "rows": rows})
    _write_csv(output / "p1_matrix.csv", [{key: value for key, value in row.items() if key not in {"v2_profile", "element_mass_fraction_span_v2", "element_mass_fraction_span_cantera", "edge_slope_v2", "edge_slope_cantera"}} for row in rows])


def run_p1_pressure_continuation(root: Path, ratio: float, seed_mesh_points: int) -> None:
    """Test mesh-aware pressure continuation without relaxing certification.

    The requested validation pressures are 1, 5, and 10 atm.  Intermediate
    pressures are bridge states in ``log(p)``; a failed corrector is never used
    as a seed, and its requested target is retried cold.  This is deliberately
    an experiment until it passes the same statistical and physical checks as
    the phi continuation controller.
    """
    if ratio <= 1.0:
        raise ValueError("The pressure continuation ratio must exceed one.")
    output = root / "P1_pressure_continuation"
    output.mkdir(parents=True, exist_ok=True)
    requested_pressures = (1.0, 5.0, 10.0)
    pressures: list[float] = [requested_pressures[0]]
    current = requested_pressures[0]
    for target in requested_pressures[1:]:
        while current * ratio < target * (1.0 - 1.0e-12):
            current *= ratio
            pressures.append(current)
        if target > current * (1.0 + 1.0e-12):
            pressures.append(target)
        current = target

    rows: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    previous_pressure: float | None = None
    previous_previous: dict[str, Any] | None = None
    previous_previous_pressure: float | None = None
    for pressure in pressures:
        requested = any(np.isclose(pressure, value, rtol=0.0, atol=1.0e-12) for value in requested_pressures)
        case = make_case(pressure_atm=pressure)
        transmitted_seed: dict[str, Any] | None = None
        if previous is None:
            predictor = "cold"
        else:
            seed, predictor = _pressure_secant_seed(
                previous=previous,
                previous_pressure=float(previous_pressure),
                previous_previous=previous_previous,
                previous_previous_pressure=previous_previous_pressure,
                target_pressure=pressure,
            )
            # Keep the state predictor and the mesh-transfer policy separate.
            # The defect below is measured against this exact transmitted seed.
            transmitted_seed = _resample_v2_seed(seed, seed_mesh_points)
        attempted_time_s = 0.0
        v2 = _v2_solve(
            case,
            profile=True,
            prev_v2_data=transmitted_seed,
            seed_mesh_points=None,
            compiled_block_substitution=True,
            acceptance_criterion="combined",
            residual_guard_inf=10.0,
            final_residual_inf=10.0,
        )
        attempted_time_s += float(v2["time_s"])
        continuation_attempt = {
            "ok": bool(v2["ok"]),
            "final_accepted": bool(v2["report"].get("final_accepted", v2["ok"])),
            "time_s": float(v2["time_s"]),
            "Finf": float(v2["report"].get("Finf_final", np.nan)),
            "weighted_step_norm": float(v2["report"].get("weighted_step_norm_final", np.nan)),
            "n_points": int(v2["n_points"]),
            "width_m": float(v2["width"]),
        }
        fallback = False
        prediction_defect = _pressure_prediction_defect(v2, transmitted_seed)
        continuation_attempt["prediction_defect"] = prediction_defect
        if not (v2["ok"] and v2["report"].get("final_accepted", False)) and previous is not None:
            fallback = True
            attempted_predictor = predictor
            predictor = "cold_fallback_after_" + attempted_predictor
            v2 = _v2_solve(
                case,
                profile=True,
                compiled_block_substitution=True,
                acceptance_criterion="combined",
                residual_guard_inf=10.0,
                final_residual_inf=10.0,
            )
            attempted_time_s += float(v2["time_s"])
            prediction_defect = float("nan")
        accepted = bool(v2["ok"] and v2["report"].get("final_accepted", False))
        row = {
            "pressure_atm": pressure,
            "requested": requested,
            "bridge": not requested,
            "previous_pressure_atm": previous_pressure,
            "step_log_pressure": None if previous_pressure is None else float(np.log(pressure / previous_pressure)),
            "predictor": predictor,
            "cold_fallback": fallback,
            "continuation_attempt": continuation_attempt,
            "accepted": accepted,
            "corrector_time_s": float(v2["time_s"]),
            "total_attempt_time_s": float(attempted_time_s),
            "prediction_defect": prediction_defect,
            "Su_m_per_s": float(v2["Su"]),
            "n_points": int(v2["n_points"]),
            "width_m": float(v2["width"]),
            "Finf": float(v2["report"].get("Finf_final", np.nan)),
            "weighted_step_norm": float(v2["report"].get("weighted_step_norm_final", np.nan)),
            "domain_expansions": int(len(v2["report"].get("expansion_events", []))),
            "profile": v2["report"].get("profile", {}),
        }
        rows.append(row)
        _write_json(output / "pressure_continuation_partial.json", rows)
        print(
            f"P1 pressure {pressure:g} atm: accepted={accepted}, "
            f"total={attempted_time_s:.3f}s, predictor={predictor}, "
            f"eta_p={prediction_defect:.3e}"
        )
        if accepted:
            previous_previous, previous_previous_pressure = previous, previous_pressure
            previous, previous_pressure = v2, pressure

    _write_json(output / "pressure_continuation.json", {
        "environment": environment_metadata(),
        "requested_pressures_atm": requested_pressures,
        "bridge_ratio": ratio,
        "seed_mesh_points": seed_mesh_points,
        "rows": rows,
        "interpretation": (
            "Experimental state-secant continuation in log(p), with a separate "
            "bounded adaptive-mesh transfer. The recorded time includes every "
            "rejected corrector and any cold fallback; no production promotion "
            "without paired repetitions and reference-profile checks."
        ),
    })
    _write_csv(
        output / "pressure_continuation.csv",
        [
            {key: value for key, value in row.items() if key not in {"profile", "continuation_attempt"}}
            for row in rows
        ],
    )


def _v2_to_v2_metrics(reference: dict[str, Any], candidate: dict[str, Any], case: FlameCase) -> dict[str, float]:
    """Compare two V2 profiles after thermal-front alignment."""
    z_ref = np.asarray(reference["z"]) - _thermal_origin(reference)
    z_candidate = np.asarray(candidate["z"]) - _thermal_origin(candidate)
    common = np.linspace(max(z_ref[0], z_candidate[0]), min(z_ref[-1], z_candidate[-1]), 4001)
    fields_ref, fields_candidate = _postprocess(reference, case), _postprocess(candidate, case)
    tiny = np.finfo(float).tiny
    T_ref = np.interp(common, z_ref, reference["T"])
    T_candidate = np.interp(common, z_candidate, candidate["T"])
    q_ref = np.interp(common, z_ref, fields_ref["qdot"])
    q_candidate = np.interp(common, z_candidate, fields_candidate["qdot"])
    names = tuple(ct.Solution(case.mech).species_names)
    y_errors = []
    for name in ACTIVE_SPECIES:
        if name not in names:
            continue
        species = names.index(name)
        y_ref = np.interp(common, z_ref, reference["Y"][species])
        y_candidate = np.interp(common, z_candidate, candidate["Y"][species])
        y_errors.append(float(np.linalg.norm(y_candidate - y_ref) / max(np.linalg.norm(y_ref), tiny)))
    return {
        "Su_relative_error": abs(float(candidate["Su"] - reference["Su"])) / max(abs(float(reference["Su"])), tiny),
        "T_l2_relative": float(np.linalg.norm(T_candidate - T_ref) / max(np.linalg.norm(T_ref), tiny)),
        "qdot_l2_relative": float(np.linalg.norm(q_candidate - q_ref) / max(np.linalg.norm(q_ref), tiny)),
        "Y_l2_relative_max_active": max(y_errors),
    }


def run_p1_domain_policy_ablation(root: Path, repetitions: int, pressures: tuple[float, ...]) -> None:
    """Pair the default and deferred domain checks at elevated pressure.

    The candidate changes *when* the width check is performed, never the
    threshold, final refinement criteria, or combined acceptance predicate.
    It can only be promoted if every run stays certified and the profiles agree
    with the existing V2 baseline.
    """
    output = root / "P1_domain_policy_ablation"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for pressure in pressures:
        case = make_case(pressure_atm=pressure)
        for repetition in range(1, repetitions + 1):
            policies: list[tuple[str, dict[str, Any]]] = [
                ("baseline", {"domain_check_after_refine_only": False}),
                ("deferred", {"domain_check_after_refine_only": True}),
            ]
            if repetition % 2 == 0:
                policies.reverse()
            solved: dict[str, dict[str, Any]] = {}
            for name, overrides in policies:
                solved[name] = _v2_solve(
                    case,
                    profile=False,
                    compiled_block_substitution=True,
                    acceptance_criterion="combined",
                    residual_guard_inf=10.0,
                    final_residual_inf=10.0,
                    solve_option_overrides=overrides,
                )
            baseline, deferred = solved["baseline"], solved["deferred"]
            row = {
                "pressure_atm": pressure,
                "replicate": repetition,
                "order": ">".join(name for name, _ in policies),
                "baseline_time_s": float(baseline["time_s"]),
                "deferred_time_s": float(deferred["time_s"]),
                "speedup_baseline_over_deferred": float(baseline["time_s"] / max(deferred["time_s"], np.finfo(float).tiny)),
                "baseline_accepted": bool(baseline["ok"] and baseline["report"].get("final_accepted", False)),
                "deferred_accepted": bool(deferred["ok"] and deferred["report"].get("final_accepted", False)),
                "baseline_nodes": int(baseline["n_points"]),
                "deferred_nodes": int(deferred["n_points"]),
                "baseline_width_m": float(baseline["width"]),
                "deferred_width_m": float(deferred["width"]),
                **_v2_to_v2_metrics(baseline, deferred, case),
            }
            rows.append(row)
            _write_json(output / "domain_policy_partial.json", rows)
            print(f"P1 domain policy {pressure:g} atm rep {repetition}: speedup={row['speedup_baseline_over_deferred']:.3f}")
    speedups = np.asarray([row["speedup_baseline_over_deferred"] for row in rows], dtype=float)

    def promotion_for(selected: list[dict[str, Any]]) -> dict[str, Any]:
        selected_speedups = np.asarray([row["speedup_baseline_over_deferred"] for row in selected], dtype=float)
        result = {
            "n_pairs": int(selected_speedups.size),
            "speedup": _summary(selected_speedups.tolist()),
            "all_deferred_certified": bool(all(row["deferred_accepted"] for row in selected)),
            "time_ci_favors_deferred": bool(_bootstrap_median_ci(selected_speedups)[0] > 1.0),
            "max_Su_relative_error": max(row["Su_relative_error"] for row in selected),
            "max_T_l2_relative": max(row["T_l2_relative"] for row in selected),
            "max_qdot_l2_relative": max(row["qdot_l2_relative"] for row in selected),
            "max_Y_l2_relative_active": max(row["Y_l2_relative_max_active"] for row in selected),
        }
        result["promote_deferred_domain_check"] = bool(
            result["all_deferred_certified"]
            and result["time_ci_favors_deferred"]
            and result["max_Su_relative_error"] <= 1.0e-3
            and result["max_T_l2_relative"] <= 5.0e-3
            and result["max_qdot_l2_relative"] <= 5.0e-2
            and result["max_Y_l2_relative_active"] <= 2.0e-2
        )
        return result

    promotion = {
        "all_deferred_certified": bool(all(row["deferred_accepted"] for row in rows)),
        "time_ci_favors_deferred": bool(_bootstrap_median_ci(speedups)[0] > 1.0),
        "max_Su_relative_error": max(row["Su_relative_error"] for row in rows),
        "max_T_l2_relative": max(row["T_l2_relative"] for row in rows),
        "max_qdot_l2_relative": max(row["qdot_l2_relative"] for row in rows),
        "max_Y_l2_relative_active": max(row["Y_l2_relative_max_active"] for row in rows),
    }
    promotion["promote_deferred_domain_check"] = bool(
        promotion["all_deferred_certified"]
        and promotion["time_ci_favors_deferred"]
        and promotion["max_Su_relative_error"] <= 1.0e-3
        and promotion["max_T_l2_relative"] <= 5.0e-3
        and promotion["max_qdot_l2_relative"] <= 5.0e-2
        and promotion["max_Y_l2_relative_active"] <= 2.0e-2
    )
    by_pressure = {
        f"{pressure:g}_atm": promotion_for([row for row in rows if np.isclose(row["pressure_atm"], pressure)])
        for pressure in pressures
    }
    _write_json(output / "domain_policy_ablation.json", {"environment": environment_metadata(), "rows": rows, "speedup": _summary(speedups.tolist()), "promotion": promotion, "promotion_by_pressure": by_pressure})
    _write_csv(output / "domain_policy_ablation.csv", rows)


def _strategy_summary(path: Path) -> dict[str, Any]:
    with np.load(path / "fgm_table.npz", allow_pickle=True) as raw:
        table = {key: np.asarray(raw[key]) for key in raw.files}
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    trace = json.loads((path / "continuation_trace.json").read_text(encoding="utf-8"))
    return {
        "path": str(path), "continuation_mode": metadata["continuation_mode"],
        "solve_time_s": float(np.sum(table["solve_time"])),
        "all_accepted": bool(np.all(table["final_accepted"])),
        "n_rows": int(table["phi_grid"].size), "n_bridge": int(np.sum(table["bridge"])),
        "n_requested": int(np.sum(table["requested"])),
        "trace_rejections": int(sum(not item.get("accepted", False) for item in trace)),
        "peak_nodes": int(np.max(table["n_points"])),
    }


def run_p1_strategies(root: Path, repetitions: int, phi_values: str) -> None:
    output = root / "P1_strategies"
    output.mkdir(parents=True, exist_ok=True)
    generator = REPOSITORY / "FGM" / "scripts" / "generate_fgm_tables_native.py"
    modes = ("cold", "fixed", "fixed-bridges", "adaptive-pc")
    all_runs: list[dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        for mode in modes:
            name = f"{mode}_rep{repetition:02d}"
            run_dir = output / name
            if not (run_dir / "fgm_table.npz").exists():
                command = [
                    sys.executable, str(generator), "--output-root", str(output), "--run-name", name,
                    "--phi-values", phi_values, "--continuation-mode", mode,
                    "--disable-seed-cache", "--n-c", "121", "--c-fine", "1001",
                    "--max-flame-time-s", "300", "--loglevel", "0",
                ]
                if mode == "fixed-bridges":
                    command.extend(("--continuation-trust-ratio", "1.15"))
                completed = subprocess.run(command, cwd=REPOSITORY, text=True, check=False)
                if completed.returncode:
                    raise RuntimeError(f"Strategy {name} failed with exit code {completed.returncode}.")
            item = _strategy_summary(run_dir)
            item["replicate"] = repetition
            all_runs.append(item)
            print(f"P1 strategy {name}: {item['solve_time_s']:.3f}s")
        _write_json(output / "strategy_summary_partial.json", all_runs)

    fixed = [output / f"fixed-bridges_rep{rep:02d}" for rep in range(1, repetitions + 1)]
    adaptive = [output / f"adaptive-pc_rep{rep:02d}" for rep in range(1, repetitions + 1)]
    from analyze_predictor_corrector_ablation import analyse  # noqa: E402

    promotion = analyse(fixed, adaptive, samples=10000, seed=20260903, active_species=ACTIVE_SPECIES)
    _write_json(output / "adaptive_pc_vs_fixed_bridges.json", promotion)
    _write_json(output / "strategy_summary.json", {"environment": environment_metadata(), "phi_values": phi_values, "runs": all_runs, "promotion": promotion["promotion"]})
    _write_csv(output / "strategy_summary.csv", all_runs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPOSITORY / "FGM" / "resultados" / "paper_campaign_20260903")
    parser.add_argument("--p0", action="store_true", help="Execute the P0 timing, diagnostics, mesh, and domain studies.")
    parser.add_argument("--p1-matrix", action="store_true", help="Execute the P1 CH4 T/p and H2 mixture-averaged robustness screen.")
    parser.add_argument("--p1-pressure-continuation", action="store_true", help="Test mesh-aware bridge continuation in log(p) for 1, 5, and 10 atm.")
    parser.add_argument("--p1-domain-policy-ablation", action="store_true", help="Pair default/deferred domain checks at 5 and 10 atm.")
    parser.add_argument("--p1-strategies", action="store_true", help="Execute cache-free cold/fixed/fixed-bridge/adaptive-PC sweeps.")
    parser.add_argument("--p0-repetitions", type=int, default=7)
    parser.add_argument("--p1-repetitions", type=int, default=7)
    parser.add_argument("--pressure-bridge-ratio", type=float, default=1.5)
    parser.add_argument("--pressure-seed-mesh-points", type=int, default=48)
    parser.add_argument("--domain-policy-pressures", type=str, default="5,10", help="Comma-separated pressure values [atm] for the deferred-domain ablation.")
    parser.add_argument("--phi-values", type=str, default="0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5")
    parser.add_argument("--skip-spatial", action="store_true", help="Skip P0 mesh/domain checks (only for a diagnostic dry run).")
    args = parser.parse_args()
    if not any((args.p0, args.p1_matrix, args.p1_pressure_continuation, args.p1_domain_policy_ablation, args.p1_strategies)):
        parser.error("Select at least one campaign task.")
    if args.p0_repetitions < 1 or args.p1_repetitions < 1:
        parser.error("Repetition counts must be positive.")
    if args.p0:
        run_p0(args.output_root, repetitions=args.p0_repetitions, spatial=not args.skip_spatial)
    if args.p1_matrix:
        run_p1_matrix(args.output_root)
    if args.p1_pressure_continuation:
        run_p1_pressure_continuation(
            args.output_root,
            ratio=float(args.pressure_bridge_ratio),
            seed_mesh_points=int(args.pressure_seed_mesh_points),
        )
    if args.p1_domain_policy_ablation:
        pressures = tuple(float(value.strip()) for value in args.domain_policy_pressures.split(",") if value.strip())
        if not pressures or any(value <= 0.0 for value in pressures):
            parser.error("--domain-policy-pressures must contain positive values.")
        run_p1_domain_policy_ablation(args.output_root, repetitions=args.p1_repetitions, pressures=pressures)
    if args.p1_strategies:
        run_p1_strategies(args.output_root, repetitions=args.p1_repetitions, phi_values=args.phi_values)


if __name__ == "__main__":
    main()
