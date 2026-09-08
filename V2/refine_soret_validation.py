"""Independent finer-mesh check from each solver's own accepted profile.

This is a seeded spatial diagnostic, NEVER a cold-start performance benchmark.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

from benchmark_soret_native import json_safe
import cantera as ct
import numpy as np
from config import FlameCase
from run_saved_comparison import _v2_solve
from state import pack_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--factor", type=float, default=.5)
    args = parser.parse_args()
    if not 0 < args.factor < 1:
        parser.error("factor must lie strictly between zero and one")
    summary_path = args.directory / "summary.json"
    old = json.loads(summary_path.read_text(encoding="utf-8"))
    if not all(any(r["solver"] == name and r["repetition"] == 0 and r["accepted"]
                   for r in old["runs"]) for name in ("native", "cantera")):
        raise ValueError("Both source profiles must have passed their solver's acceptance")
    case = FlameCase(**old["case"])
    case.slope *= args.factor
    case.curve *= args.factor
    out = args.directory.parent / datetime.now().strftime("refinement_%Y%m%d_%H%M%S")
    out.mkdir(exist_ok=False)
    summary = dict(case=asdict(case), runs=[], options=dict(mesh_factor=args.factor),
                   command=__import__("sys").argv, source=str(args.directory),
                   source_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                   measurement="Seeded finer-mesh diagnostic; not cold-start timing")
    print(out.resolve(), flush=True)
    for name in ("native", "cantera"):
        with np.load(args.directory / f"{name}_0.npz") as data:
            previous = {k: data[k].copy() for k in data.files}
        if name == "native":
            previous["x"] = pack_state(previous["u"], previous["T"], previous["Y"])
            result = _v2_solve(case, prev_v2_data=previous,
                solve_option_overrides=dict(max_refine_passes=15, max_total_time_s=90.,
                    lag_multicomponent_transport=True))
            result["accepted"] = bool(result["ok"] and result["report"]["final_accepted"])
            result["Finf"] = result["report"]["Finf_final"]
            result["stages"] = [result.pop("report")]
            result.pop("x")
        else:
            gas = ct.Solution(case.mech)
            gas.TP = case.T_in, case.P
            gas.set_equivalence_ratio(case.phi, case.fuel, case.oxidizer)
            flame = ct.FreeFlame(gas, grid=previous["z"])
            flame.transport_model = case.transport_model
            flame.soret_enabled = case.soret_enabled
            flame.set_initial_guess()
            loc = (previous["z"] - previous["z"][0]) / np.ptp(previous["z"])
            flame.set_profile("T", loc, previous["T"])
            flame.set_profile("velocity", loc, previous["u"])
            for k, species in enumerate(gas.species_names):
                flame.set_profile(species, loc, previous["Y"][k])
            flame.fixed_temperature = case.T_in + .25 * (previous["T"].max() - case.T_in)
            flame.set_refine_criteria(ratio=case.ratio, slope=case.slope,
                                      curve=case.curve, prune=case.prune)
            start = time.perf_counter()
            flame.solve(loglevel=0, auto=False, refine_grid=True)
            result = dict(time_s=time.perf_counter() - start, accepted=True,
                          Su=float(flame.velocity[0]), n_points=len(flame.grid),
                          width=float(np.ptp(flame.grid)), z=flame.grid,
                          T=flame.T, Y=flame.Y, u=flame.velocity)
        profiles = {k: result.pop(k) for k in ("z", "T", "Y", "u")}
        np.savez_compressed(out / f"{name}_0.npz", **profiles)
        result.update(solver=name, repetition=0)
        summary["runs"].append(result)
        (out / "summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
        print(json.dumps(json_safe({k: v for k, v in result.items() if k not in ("stages", "species_names")})), flush=True)


if __name__ == "__main__":
    main()
