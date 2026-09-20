"""Observe Cantera auto stages through public callbacks, without changing them."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import cantera as ct
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pressure-atm", type=float, default=10.)
    parser.add_argument("--mechanism", default="gri30.yaml")
    args = parser.parse_args()
    gas = ct.Solution(args.mechanism)
    gas.TP = 300., args.pressure_atm * ct.one_atm
    gas.set_equivalence_ratio(1., "H2", "O2:1,N2:3.76")
    flame = ct.FreeFlame(gas, width=.03)
    flame.transport_model = "multicomponent"
    flame.soret_enabled = True
    flame.set_refine_criteria(ratio=2.5, slope=.04, curve=.08, prune=.003)
    flame.set_max_grid_points(flame.flame, 1600)
    events = []
    start = time.perf_counter()
    def record(kind, value):
        events.append(dict(kind=kind, elapsed_s=time.perf_counter() - start,
                           n_points=len(flame.grid), transport=flame.transport_model,
                           soret=flame.soret_enabled, energy=flame.energy_enabled,
                           width=float(np.ptp(flame.grid)), dt=float(value)))
        return 0.
    # FreeFlame.solve chains the user steady callback after its width check;
    # failed width checks are thus absent from these successful steady events.
    flame.set_steady_callback(lambda v: record("steady", v))
    flame.set_time_step_callback(lambda v: record("time_step", v))
    start = time.perf_counter()
    flame.solve(loglevel=0, auto=True)
    elapsed = time.perf_counter() - start
    stats = {name: getattr(flame, name) for name in
             ("grid_size_stats", "eval_count_stats", "eval_time_stats", "jacobian_count_stats",
              "jacobian_time_stats", "time_step_stats")}
    out = Path("resultados/soret_native_validation") / datetime.now().strftime("cantera_trace_%Y%m%d_%H%M%S.json")
    result = dict(version=ct.__version__, arguments=vars(args), wall_s=elapsed, Su=float(flame.velocity[0]),
                  stats=stats, events=events, n_points=len(flame.grid), width=float(np.ptp(flame.grid)),
                  caveat="Callback timestamps include callback overhead; CPU statistics are not wall time. "
                         "First event in a new transport stage is after its first success, not its exact start.")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(out.resolve())
    print(json.dumps({k:v for k,v in result.items() if k != "events"}))
    print(json.dumps([e for e in events if e["kind"] == "steady"]))


if __name__ == "__main__":
    main()
