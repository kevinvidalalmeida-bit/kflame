"""Compare KFlame against Cantera on the 4 certified validation cases.

Cases:
  - CH4_1:  CH4/air, 1 atm, phi=1.0, multicomponent, Soret
  - CH4_10: CH4/air, 10 atm, phi=1.0, multicomponent, Soret
  - H2_1:   H2/air, 1 atm, phi=1.0, multicomponent, Soret
  - H2_10:  H2/air, 10 atm, phi=1.0, multicomponent, Soret

Solvers compared:
  1. Cantera 3.2.0 (direct ct.FreeFlame with auto=True)
  2. KFlame PTC-SER (Baseline)
  3. KFlame Persistent Backward Euler (euler_recovery)
  4. KFlame Full Backward Euler
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "4"

import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kflame.flame.config import FlameCase
from kflame.reference.compare import _cantera_solve
from benchmarks.benchmark_euler_recovery import CASES, get_options_for_variant
from kflame.benchmarks.soret import native_solve, json_safe


def run_cantera_comparison(cases_to_run=None, output_path="runs/nonlinear/cantera_vs_kflame.json"):
    if cases_to_run is None:
        cases_to_run = ["CH4_1", "CH4_10", "H2_1", "H2_10"]

    results = {}
    print("=" * 90)
    print("CAMPAÑA COMPARATIVA CANTERA 3.2.0 VS KFLAME")
    print("=" * 90)

    for case_name in cases_to_run:
        case = CASES[case_name]
        print(f"\n--- Ejecutando Caso: {case_name} ---")
        case_data = {}

        # 1. Cantera solve
        print(f" -> Ejecutando Cantera 3.2.0...")
        t0 = time.perf_counter()
        ct_res = _cantera_solve(case, loglevel=0)
        ct_time = time.perf_counter() - t0
        ct_su = float(ct_res["Su"])
        ct_nodes = int(ct_res["n_points"])
        ct_tmax = float(np.max(ct_res["T"]))
        case_data["cantera"] = {
            "time_s": ct_time,
            "Su": ct_su,
            "Tmax": ct_tmax,
            "n_points": ct_nodes,
        }
        print(f"    [Cantera] {ct_time:.3f} s | Su: {ct_su:.5f} m/s | Tmax: {ct_tmax:.1f} K | Nodos: {ct_nodes}")

        # 2. KFlame variants (ptc_ser, persistent_backward_euler, full_backward_euler)
        for vname in ["ptc_ser", "persistent_backward_euler", "full_backward_euler"]:
            print(f" -> Ejecutando KFlame ({vname})...")
            opts = get_options_for_variant(vname)
            t0 = time.perf_counter()
            k_res = native_solve(case, opts, bootstrap=True, bootstrap_mesh_factor=1.0)
            k_time = time.perf_counter() - t0
            k_su = float(k_res["Su"])
            k_nodes = int(k_res["n_points"])
            k_tmax = float(np.max(k_res["T"]))
            speedup = ct_time / k_time if k_time > 0 else float("inf")
            case_data[vname] = {
                "time_s": k_time,
                "Su": k_su,
                "Tmax": k_tmax,
                "n_points": k_nodes,
                "speedup_vs_cantera": speedup,
            }
            print(f"    [KFlame-{vname}] {k_time:.3f} s | Speedup vs Cantera: {speedup:.2f}x | Su: {k_su:.5f} m/s | Nodos: {k_nodes}")

        results[case_name] = case_data

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(json_safe(results), indent=2), encoding="utf-8")
    print(f"\nResultados guardados en: {out_file}")

    # Tabla resumen
    print("\n" + "=" * 105)
    print(f"{'Caso':<8} | {'Solver / Variante':<30} | {'Tiempo (s)':<12} | {'Speedup vs CT':<15} | {'Su (m/s)':<12} | {'Nodos':<6}")
    print("=" * 105)
    for cname, cdata in results.items():
        ct_t = cdata["cantera"]["time_s"]
        print(f"{cname:<8} | {'Cantera 3.2.0':<30} | {ct_t:<12.3f} | {'1.00x (ref)':<15} | {cdata['cantera']['Su']:<12.5f} | {cdata['cantera']['n_points']:<6}")
        for vname in ["ptc_ser", "persistent_backward_euler", "full_backward_euler"]:
            vinfo = cdata[vname]
            sp = f"{vinfo['speedup_vs_cantera']:.2f}x"
            print(f"{'':<8} | {f'KFlame ({vname})':<30} | {vinfo['time_s']:<12.3f} | {sp:<15} | {vinfo['Su']:<12.5f} | {vinfo['n_points']:<6}")
        print("-" * 105)

    return results


if __name__ == "__main__":
    run_cantera_comparison()
