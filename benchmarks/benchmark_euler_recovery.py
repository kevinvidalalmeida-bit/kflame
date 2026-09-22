"""Benchmark harness for Persistent Backward Euler Recovery Campaign.

Runs 3 variants x 4 cases with paired alternating execution order:
  - Variant 1: ptc_ser (Baseline PTC-SER)
  - Variant 2: full_backward_euler (Full Multi-iteration Backward Euler)
  - Variant 3: persistent_backward_euler (Persistent Backward Euler switching)

Metrics recorded:
  - End-to-end wall time (median, std, min)
  - Residual evaluations count
  - Jacobian builds count
  - LU factorizations count
  - Pseudo-time steps count
  - Physical solution metrics: Su (m/s), T_max (K), Mesh points (N)
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "4"

import argparse
import json
from dataclasses import replace
import time
import numpy as np

from kflame.benchmarks.soret import benchmark_options, native_solve, json_safe
from kflame.flame.config import FlameCase

ONE_ATM = 101325.0

CASES = {
    "CH4_1": FlameCase(
        fuel="CH4", P=ONE_ATM,
        transport_model="multicomponent", soret_enabled=True,
        ratio=2.5, slope=0.04, curve=0.08, prune=0.003,
    ),
    "CH4_10": FlameCase(
        fuel="CH4", P=10.0 * ONE_ATM,
        transport_model="multicomponent", soret_enabled=True,
        ratio=2.5, slope=0.04, curve=0.08, prune=0.003,
    ),
    "H2_1": FlameCase(
        fuel="H2", P=ONE_ATM,
        transport_model="multicomponent", soret_enabled=True,
        ratio=2.5, slope=0.04, curve=0.08, prune=0.003,
    ),
    "H2_10": FlameCase(
        fuel="H2", P=10.0 * ONE_ATM,
        transport_model="multicomponent", soret_enabled=True,
        ratio=2.5, slope=0.04, curve=0.08, prune=0.003,
    ),
}


def get_options_for_variant(variant_name: str):
    base = benchmark_options(profile=True)
    if variant_name == "ptc_ser":
        return replace(base, transient_solver_mode="ptc_ser")
    elif variant_name == "full_backward_euler":
        return replace(base, transient_solver_mode="full_backward_euler")
    elif variant_name == "persistent_backward_euler":
        return replace(
            base,
            transient_solver_mode="persistent_backward_euler",
            persistent_be_steps=5,
            persistent_be_stall_threshold=0.8,
        )
    else:
        raise ValueError(f"Unknown variant: {variant_name}")


def extract_counts(solve_dict: dict):
    stages = solve_dict.get("stages", [])
    total_res = 0
    total_jac = 0
    total_lu = 0
    total_steps = 0
    res_time = 0.0
    jac_time = 0.0
    lu_time = 0.0
    solve_time = 0.0
    for st in stages:
        prof = st.get("profile", {})
        rf = prof.get("residual_full", {})
        jb = prof.get("jacobian_build", {})
        lf = prof.get("linear_factorize", {})
        ls = prof.get("linear_solve", {})

        total_res += rf.get("count", 0)
        res_time += float(rf.get("time_s", 0.0))

        total_jac += jb.get("count", 0)
        jac_time += float(jb.get("time_s", 0.0))

        total_lu += lf.get("count", 0)
        lu_time += float(lf.get("time_s", 0.0))

        solve_time += float(ls.get("time_s", 0.0))

        passes = st.get("passes", [])
        for p in passes:
            if isinstance(p, dict):
                for st_name in ("stage_A", "stage_B", "stage_C"):
                    if st_name in p:
                        st_info = p[st_name]
                        total_steps += st_info.get("steps", 0)
    return {
        "res_cnt": total_res,
        "jac_cnt": total_jac,
        "lu_cnt": total_lu,
        "steps": total_steps,
        "res_time_s": res_time,
        "jac_time_s": jac_time,
        "lu_time_s": lu_time,
        "solve_time_s": solve_time,
    }


def run_campaign(cases_to_run=None, pairs=3, output_path="runs/nonlinear/euler_recovery_campaign.json"):
    if cases_to_run is None:
        cases_to_run = list(CASES.keys())
    
    variant_names = ["ptc_ser", "full_backward_euler", "persistent_backward_euler"]
    
    # Warmup JIT
    print("=== Warmup JIT ===")
    dummy_case = CASES["CH4_1"]
    for vname in variant_names:
        native_solve(dummy_case, get_options_for_variant(vname), bootstrap=False)
    print("=== Warmup complete ===\n")

    campaign_results = {}

    for case_name in cases_to_run:
        print(f"\n============================================================")
        print(f"  BENCHMARK CASE: {case_name}")
        print(f"============================================================")
        case = CASES[case_name]
        
        variant_times = {v: [] for v in variant_names}
        variant_runs = {v: [] for v in variant_names}

        # Paired execution
        for pair_idx in range(pairs):
            order = list(variant_names)
            if pair_idx % 2 == 1:
                order.reverse()
            print(f" Pair {pair_idx + 1}/{pairs} order: {order}")

            for vname in order:
                opts = get_options_for_variant(vname)
                t0 = time.perf_counter()
                res = native_solve(case, opts, bootstrap=True, bootstrap_mesh_factor=2.0)
                elapsed = time.perf_counter() - t0

                variant_times[vname].append(elapsed)
                diag = extract_counts(res)
                
                run_data = {
                    "pair": pair_idx,
                    "time_s": elapsed,
                    "accepted": res.get("accepted", False),
                    "Su": res.get("Su"),
                    "Tmax": float(np.max(res.get("T", [0]))),
                    "n_points": res.get("n_points"),
                    "res_evals": diag["res_cnt"],
                    "jac_evals": diag["jac_cnt"],
                    "lu_facts": diag["lu_cnt"],
                    "steps": diag["steps"],
                    "res_time_s": diag["res_time_s"],
                    "jac_time_s": diag["jac_time_s"],
                    "lu_time_s": diag["lu_time_s"],
                    "solve_time_s": diag["solve_time_s"],
                }
                variant_runs[vname].append(run_data)
                print(f"   [{vname:<25}] {elapsed:.3f} s | Res: {diag['res_cnt']} ({diag['res_time_s']:.2f}s) | Jac: {diag['jac_cnt']} ({diag['jac_time_s']:.2f}s) | LU: {diag['lu_cnt']} ({diag['lu_time_s']:.2f}s) | Steps: {diag['steps']}")

        # Summary for case
        case_summary = {}
        for vname in variant_names:
            times = variant_times[vname]
            runs = variant_runs[vname]
            last_run = runs[-1]
            case_summary[vname] = {
                "median_time_s": float(np.median(times)),
                "min_time_s": float(np.min(times)),
                "mean_time_s": float(np.mean(times)),
                "std_time_s": float(np.std(times)),
                "times": times,
                "accepted": last_run["accepted"],
                "Su": last_run["Su"],
                "Tmax": last_run["Tmax"],
                "n_points": last_run["n_points"],
                "res_evals": last_run["res_evals"],
                "jac_evals": last_run["jac_evals"],
                "lu_facts": last_run["lu_facts"],
                "steps": last_run["steps"],
                "res_time_s": float(np.median([r["res_time_s"] for r in runs])),
                "jac_time_s": float(np.median([r["jac_time_s"] for r in runs])),
                "lu_time_s": float(np.median([r["lu_time_s"] for r in runs])),
                "solve_time_s": float(np.median([r["solve_time_s"] for r in runs])),
                "all_runs": runs,
            }
        campaign_results[case_name] = case_summary

    # Save campaign output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_safe({"results": campaign_results}), f, indent=2)

    print(f"\nSaved campaign results to: {output_path}")

    # Print summary table
    print("\n" + "="*115)
    print(f"{'Case':<8} | {'Variant':<25} | {'Median (s)':<10} | {'Res Time (s)':<12} | {'Jac Time (s)':<12} | {'LU Time (s)':<12} | {'Steps':<6} | {'Su (m/s)':<10}")
    print("="*115)
    for cname, cdata in campaign_results.items():
        for vname, vdata in cdata.items():
            print(f"{cname:<8} | {vname:<25} | {vdata['median_time_s']:<10.3f} | {vdata['res_time_s']:<12.3f} | {vdata['jac_time_s']:<12.3f} | {vdata['lu_time_s']:<12.3f} | {vdata['steps']:<6} | {vdata['Su']:<10.5f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["CH4_1", "CH4_10", "H2_1", "H2_10"])
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--output", default="runs/nonlinear/euler_recovery_campaign.json")
    args = parser.parse_args()

    run_campaign(args.cases, pairs=args.pairs, output_path=args.output)
