"""Controlled pseudo-arclength seed experiment for one certified transition.

This is an ablation driver, not the FGM production path. It compares the
established fixed-parameter Jacobian tangent with a bordered
pseudo-arclength seed, then sends both through the same V2 solver and
certificate. The JSON records every cost, including the augmented seed work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
V2_DIRECTORY = ROOT / "V2"
MATERIALS_DIRECTORY = V2_DIRECTORY / "materiales"
for directory in (V2_DIRECTORY, MATERIALS_DIRECTORY, ROOT / "FGM" / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from equations import residual  # noqa: E402
from mechanism_data import load_mechanism  # noqa: E402
from problem import FreeFlameProblem  # noqa: E402
from run_saved_comparison import _resolve_mechanism, _v2_solve  # noqa: E402
from species_backend_native import NativeSpeciesBackend  # noqa: E402
from config import FlameCase  # noqa: E402
from generate_fgm_tables_native import _jacobian_tangent_state  # noqa: E402
from pseudo_arclength_flame import pseudo_arclength_flame_seed  # noqa: E402


ATM = 101325.0


def _case(phi: float, pressure_atm: float, width: float) -> FlameCase:
    return FlameCase(
        mech=_resolve_mechanism("gri30.yaml"),
        fuel="CH4",
        oxidizer="O2:1.0, N2:3.76",
        phi=float(phi),
        T_in=300.0,
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


def _accepted(result: dict[str, Any]) -> bool:
    report = result.get("report", {})
    return bool(result.get("ok", False)) and bool(report.get("final_accepted", False))


def _target_problem(
    case: FlameCase,
    source: dict[str, Any],
    mech_data: Any,
) -> FreeFlameProblem:
    problem = FreeFlameProblem(case, n_points=max(2, int(np.asarray(source["z"]).size)))
    problem.assume_finite_y = True
    problem.z = np.asarray(source["z"], dtype=float).copy()
    problem.n_points = int(problem.z.size)
    problem.width = float(problem.z[-1] - problem.z[0])
    problem.backend_factory = lambda prob: NativeSpeciesBackend(prob, mech_data=mech_data)
    return problem


def _target_summary(result: dict[str, Any], seed_time_s: float, seed_meta: dict[str, Any]) -> dict[str, Any]:
    report = dict(result.get("report", {}))
    return {
        "accepted": _accepted(result),
        "solve_time_s": float(result["time_s"]),
        "seed_time_s": float(seed_time_s),
        "end_to_end_time_s": float(seed_time_s + float(result["time_s"])),
        "Su_m_per_s": float(result["Su"]),
        "n_points": int(result["n_points"]),
        "width_m": float(result["width"]),
        "residual_inf": float(report.get("Finf_final", np.nan)),
        "weighted_step_norm": float(report.get("weighted_step_norm_final", np.nan)),
        "seed": seed_meta,
    }


def _tangent_corrected_target(
    *,
    source: dict[str, Any],
    target_case: FlameCase,
    target_phi: float,
    mech_data: Any,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """Use the established tangent, then the normal V2 certificate."""
    tangent_problem = _target_problem(target_case, source, mech_data)
    t0 = time.perf_counter()
    tangent = _jacobian_tangent_state(
        tangent_problem, target_phi, source, damping=1.0
    )
    tangent_seed_time = float(time.perf_counter() - t0)
    if tangent is None:
        raise RuntimeError("The established Jacobian tangent predictor was unavailable.")
    tangent_state, tangent_meta = tangent
    result = _v2_solve(
        target_case,
        profile=True,
        acceptance_criterion="cantera",
        prev_v2_data={"z": source["z"], "x": tangent_state},
    )
    return result, tangent_seed_time, tangent_meta


def run_experiment(
    *,
    source_phi: float,
    target_phi: float,
    pressure_atm: float,
    width: float,
    output: Path,
) -> dict[str, Any]:
    source_case = _case(source_phi, pressure_atm, width)
    target_case = _case(target_phi, pressure_atm, width)
    source = _v2_solve(source_case, profile=True, acceptance_criterion="cantera")
    if not _accepted(source):
        raise RuntimeError("The source flame failed certification; no continuation experiment is valid.")
    if not isinstance(source.get("continuation_linearization"), dict):
        raise RuntimeError("Source solve did not retain an exact stationary continuation LU.")

    mech_data = load_mechanism(source_case.mech)
    tangent_result, tangent_seed_time, tangent_meta = _tangent_corrected_target(
        source=source,
        target_case=target_case,
        target_phi=target_phi,
        mech_data=mech_data,
    )

    t0 = time.perf_counter()
    pseudo_seed, pseudo_meta = pseudo_arclength_flame_seed(
        case=source_case,
        target_phi=target_phi,
        source=source,
        mech_data=mech_data,
    )
    pseudo_seed_time = float(time.perf_counter() - t0)
    if pseudo_seed is None:
        pseudo_result = None
        pseudo_summary = {
            "accepted": False,
            "seed_time_s": pseudo_seed_time,
            "seed": pseudo_meta,
        }
    else:
        pseudo_phi = float(np.asarray(pseudo_seed["phi"], dtype=float).ravel()[0])
        if bool(pseudo_meta.get("bridge", False)):
            bridge_case = _case(pseudo_phi, pressure_atm, width)
            bridge_result = _v2_solve(
                bridge_case,
                profile=True,
                acceptance_criterion="cantera",
                prev_v2_data=pseudo_seed,
            )
            bridge_summary = _target_summary(bridge_result, pseudo_seed_time, pseudo_meta)
            if not _accepted(bridge_result):
                pseudo_result = None
                pseudo_summary = {
                    "accepted": False,
                    "bridge": bridge_summary,
                    "reason": "bridge_certificate_failed",
                }
            else:
                pseudo_result, target_seed_time, target_seed_meta = _tangent_corrected_target(
                    source=bridge_result,
                    target_case=target_case,
                    target_phi=target_phi,
                    mech_data=mech_data,
                )
                target_summary = _target_summary(
                    pseudo_result, target_seed_time, target_seed_meta
                )
                pseudo_summary = {
                    "accepted": bool(_accepted(pseudo_result)),
                    "bridge": bridge_summary,
                    "target": target_summary,
                    "seed_time_s": float(pseudo_seed_time),
                    "end_to_end_time_s": float(
                        bridge_summary["end_to_end_time_s"]
                        + target_summary["end_to_end_time_s"]
                    ),
                    "Su_m_per_s": float(pseudo_result["Su"]),
                    "n_points": int(pseudo_result["n_points"]),
                    "seed": pseudo_meta,
                }
        else:
            pseudo_result = _v2_solve(
                target_case,
                profile=True,
                acceptance_criterion="cantera",
                prev_v2_data=pseudo_seed,
            )
            pseudo_summary = _target_summary(pseudo_result, pseudo_seed_time, pseudo_meta)

    source_summary = _target_summary(source, 0.0, {"predictor_kind": "cold"})
    tangent_summary = _target_summary(tangent_result, tangent_seed_time, tangent_meta)
    payload: dict[str, Any] = {
        "scope": "fixed_mesh_bordered_pseudo_arclength_seed_then_full_v2_certificate",
        "case": {
            "fuel": "CH4",
            "oxidizer": "O2:1.0, N2:3.76",
            "T_in_K": 300.0,
            "pressure_atm": pressure_atm,
            "width_m": width,
            "phi_source": source_phi,
            "phi_target": target_phi,
            "acceptance": "cantera_weighted_step_and_Finf_guard_1e4",
        },
        "source": source_summary,
        "tangent": tangent_summary,
        "pseudo_arclength": pseudo_summary,
    }
    if pseudo_result is not None:
        payload["comparison"] = {
            "relative_Su_difference": abs(float(pseudo_result["Su"]) - float(tangent_result["Su"]))
            / max(abs(float(tangent_result["Su"])), 1.0e-300),
            "end_to_end_speedup_tangent_over_pseudo": float(
                tangent_summary["end_to_end_time_s"] / pseudo_summary["end_to_end_time_s"]
            ),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-phi", type=float, default=1.0)
    parser.add_argument("--target-phi", type=float, default=1.02)
    parser.add_argument("--pressure-atm", type=float, default=10.0)
    parser.add_argument("--width", type=float, default=0.06)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "FGM" / "resultados" / "pseudo_arclength_experiment.json",
    )
    args = parser.parse_args()
    payload = run_experiment(
        source_phi=args.source_phi,
        target_phi=args.target_phi,
        pressure_atm=args.pressure_atm,
        width=args.width,
        output=args.output,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
