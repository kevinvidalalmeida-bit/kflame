"""Benchmark harness for the global nonlinear startup optimization campaign.

Runs four variants × four cases with paired alternating execution order.
Each variant and case use identical physics, tolerances, and acceptance criteria.
Only the experimental nonlinear startup flags differ.

Usage:
    python benchmarks/benchmark_nonlinear_startup.py --output runs/nonlinear/
    python benchmarks/benchmark_nonlinear_startup.py --cases CH4_1 H2_10 --pairs 1 --output tmp/quick.json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")

import numpy as np

from kflame.benchmarks.soret import benchmark_options, compact_result, json_safe, native_solve
from kflame.flame.config import FlameCase
from kflame.flame.solver import SolveOptions


# ---------------------------------------------------------------------------
#  Variant definitions
# ---------------------------------------------------------------------------
VARIANTS = {
    "baseline": dict(),
    "A": dict(adaptive_newton_corrections=True),
    "B": dict(early_refinement=True),
    "C": dict(combined_startup_strategy=True),
}

CASES = {
    "CH4_1": dict(fuel="CH4", P=101325.0 * 1.0,
                  transport_model="mixture-averaged", soret_enabled=False),
    "CH4_10": dict(fuel="CH4", P=101325.0 * 10.0,
                   transport_model="mixture-averaged", soret_enabled=False),
    "H2_1": dict(fuel="H2", P=101325.0 * 1.0,
                 transport_model="multicomponent", soret_enabled=True),
    "H2_10": dict(fuel="H2", P=101325.0 * 10.0,
                  transport_model="multicomponent", soret_enabled=True),
}


def _make_options(variant_name: str, verbose: bool = False, profile: bool = True) -> SolveOptions:
    """Build certified solve options with the experimental flags for this variant."""
    base = benchmark_options(profile=profile)
    overrides = VARIANTS.get(variant_name, {})
    return replace(base, verbose=verbose, **overrides)


def _make_case(case_name: str) -> FlameCase:
    """Build a FlameCase for the given test condition."""
    params = CASES[case_name]
    return FlameCase(
        mech="gri30.yaml",
        fuel=params["fuel"],
        oxidizer="O2:1, N2:3.76",
        phi=1.0,
        T_in=300.0,
        P=params["P"],
        width=0.03,
        transport_model=params["transport_model"],
        soret_enabled=params["soret_enabled"],
        ratio=2.5,
        slope=0.04,
        curve=0.08,
        prune=0.003,
    )


def run_one(case_name: str, variant_name: str, verbose: bool = False) -> dict:
    """Solve a single flame with the given variant. Returns metrics dict."""
    case = _make_case(case_name)
    opts = _make_options(variant_name, verbose=verbose)
    multi = case.transport_model == "multicomponent"
    bootstrap = multi  # H2 cases use transport bootstrap

    start = time.perf_counter()
    result = native_solve(case, opts, bootstrap=bootstrap, bootstrap_mesh_factor=2.0)
    wall = time.perf_counter() - start

    # Extract detailed metrics from the report stages
    stages = result.get("stages", [])
    total_residual_evals = 0
    total_jac_evals = 0
    total_lu_factorizations = 0
    total_ptc_steps = 0
    total_anc_corrections = 0
    total_early_refinements = 0

    for stage in stages:
        prof = stage.get("profile", {})
        if isinstance(prof, dict):
            for key, val in prof.items():
                if isinstance(val, dict):
                    if "residual" in key.lower():
                        total_residual_evals += val.get("count", 0)
                    if "linear_model" == key:
                        total_jac_evals += val.get("count", 0)
                    if "factori" in key.lower():
                        total_lu_factorizations += val.get("count", 0)

        # Count PTC steps and ANC corrections from history
        for p in stage.get("passes", []):
            if isinstance(p, dict):
                pass  # Already counted in profile

    record = dict(
        case=case_name,
        variant=variant_name,
        wall_s=wall,
        time_s=result.get("time_s", wall),
        setup_time_s=result.get("setup_time_s", 0.0),
        end_to_end_s=result.get("end_to_end_s", wall),
        accepted=result.get("accepted", False),
        Su=result.get("Su", float("nan")),
        n_points=result.get("n_points", 0),
        width=result.get("width", 0.0),
        Finf=result.get("Finf", float("nan")),
        species_sum_error=result.get("species_sum_error", float("nan")),
        mass_flow_error=result.get("mass_flow_error", float("nan")),
    )
    # Keep profiles for comparison
    record["_profiles"] = {
        "z": result.get("z"),
        "T": result.get("T"),
        "Y": result.get("Y"),
        "u": result.get("u"),
    }
    record["stages"] = stages
    return record


def compare_profiles(rec_a: dict, rec_b: dict) -> dict:
    """Compare two run profiles field by field."""
    diffs = {}
    for key in ("z", "T", "u"):
        a = rec_a.get("_profiles", {}).get(key)
        b = rec_b.get("_profiles", {}).get(key)
        if a is not None and b is not None:
            a, b = np.asarray(a), np.asarray(b)
            if a.shape == b.shape:
                diffs[key] = float(np.max(np.abs(a - b)))
            else:
                diffs[key] = f"shape_mismatch: {a.shape} vs {b.shape}"
    a_y = rec_a.get("_profiles", {}).get("Y")
    b_y = rec_b.get("_profiles", {}).get("Y")
    if a_y is not None and b_y is not None:
        a_y, b_y = np.asarray(a_y), np.asarray(b_y)
        if a_y.shape == b_y.shape:
            diffs["Y"] = float(np.max(np.abs(a_y - b_y)))
        else:
            diffs["Y"] = f"shape_mismatch: {a_y.shape} vs {b_y.shape}"
    # Su difference
    su_a = rec_a.get("Su", float("nan"))
    su_b = rec_b.get("Su", float("nan"))
    if np.isfinite(su_a) and np.isfinite(su_b) and abs(su_a) > 1e-12:
        diffs["relative_Su_difference"] = abs(su_a - su_b) / abs(su_a)
    return diffs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="+", default=list(CASES.keys()),
                        choices=list(CASES.keys()))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()),
                        choices=list(VARIANTS.keys()))
    parser.add_argument("--pairs", type=int, default=3,
                        help="Number of paired runs per case×variant")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output_dir = args.output
    if output_dir.suffix == ".json":
        output_file = output_dir
        output_dir = output_dir.parent
    else:
        output_dir = output_dir / datetime.now().strftime("campaign_%Y%m%d_%H%M%S")
        output_file = output_dir / "results.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    source_root = Path(__file__).resolve().parents[1] / "src" / "kflame"
    summary = dict(
        protocol="Paired alternating benchmark; 1 warmup excluded per case; "
                 "all variants use identical physics, tolerances, acceptance criteria",
        cases=args.cases,
        variants=args.variants,
        pairs=args.pairs,
        platform=platform.platform(),
        cpu=platform.processor(),
        python=sys.version,
        numpy_version=np.__version__,
        threads={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")},
        source_sha256={
            str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source_root.rglob("*.py"))
        },
        results={},
    )

    for case_name in args.cases:
        print(f"\n{'='*60}")
        print(f"  CASE: {case_name}")
        print(f"{'='*60}")

        # Warmup: one run per variant (excluded from measurements)
        print("  Warming up...", flush=True)
        for variant_name in args.variants:
            try:
                warm = run_one(case_name, variant_name, verbose=False)
                print(f"    {variant_name}: {warm['end_to_end_s']:.2f}s "
                      f"accepted={warm['accepted']} Su={warm['Su']:.6f}", flush=True)
            except Exception as e:
                print(f"    {variant_name}: WARMUP FAILED: {e}", flush=True)

        case_results: dict[str, list] = {v: [] for v in args.variants}

        for pair_idx in range(args.pairs):
            # Alternate order to reduce systematic bias
            if pair_idx % 2 == 0:
                order = list(args.variants)
            else:
                order = list(reversed(args.variants))

            print(f"\n  Pair {pair_idx + 1}/{args.pairs} order: {order}", flush=True)

            for variant_name in order:
                try:
                    rec = run_one(case_name, variant_name, verbose=args.verbose)
                    # Save profiles separately
                    profiles = rec.pop("_profiles", {})
                    npz_name = f"{case_name}_{variant_name}_{pair_idx}.npz"
                    if profiles.get("z") is not None:
                        np.savez_compressed(
                            output_dir / npz_name,
                            **{k: v for k, v in profiles.items() if v is not None},
                        )
                    rec["pair"] = pair_idx
                    rec["npz"] = npz_name
                    case_results[variant_name].append(rec)
                    print(f"    {variant_name}: {rec['end_to_end_s']:.2f}s "
                          f"accepted={rec['accepted']} Su={rec['Su']:.6f} "
                          f"n_pts={rec['n_points']}", flush=True)
                except Exception as e:
                    import traceback
                    case_results[variant_name].append(dict(
                        case=case_name, variant=variant_name, pair=pair_idx,
                        error=str(e), traceback=traceback.format_exc(),
                        accepted=False,
                    ))
                    print(f"    {variant_name}: FAILED: {e}", flush=True)

        # Compute medians and comparisons
        case_summary: dict[str, dict] = {}
        for variant_name, runs in case_results.items():
            times = [r["end_to_end_s"] for r in runs if "end_to_end_s" in r]
            accepted = [r.get("accepted", False) for r in runs]
            case_summary[variant_name] = dict(
                runs=[{k: v for k, v in r.items() if k != "stages"} for r in runs],
                median_s=float(np.median(times)) if times else float("nan"),
                all_accepted=all(accepted),
                n_runs=len(runs),
            )

        # Compare each variant against baseline
        baseline_runs = case_results.get("baseline", [])
        for variant_name in args.variants:
            if variant_name == "baseline":
                continue
            variant_runs = case_results.get(variant_name, [])
            if baseline_runs and variant_runs:
                base_median = case_summary["baseline"]["median_s"]
                var_median = case_summary[variant_name]["median_s"]
                if np.isfinite(base_median) and base_median > 0:
                    case_summary[variant_name]["speedup"] = base_median / var_median
                    case_summary[variant_name]["reduction_pct"] = (
                        100.0 * (1.0 - var_median / base_median)
                    )

        summary["results"][case_name] = case_summary
        output_file.write_text(
            json.dumps(json_safe(summary), indent=2), encoding="utf-8"
        )

    # Final summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Case':<12} {'Variant':<12} {'Median (s)':>12} {'Speedup':>10} {'Accepted':>10}")
    print("-" * 60)
    for case_name in args.cases:
        case_data = summary["results"].get(case_name, {})
        for variant_name in args.variants:
            vdata = case_data.get(variant_name, {})
            median = vdata.get("median_s", float("nan"))
            speedup = vdata.get("speedup", 1.0) if variant_name != "baseline" else 1.0
            accepted = vdata.get("all_accepted", False)
            print(f"{case_name:<12} {variant_name:<12} {median:>12.3f} {speedup:>10.3f} {str(accepted):>10}")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
