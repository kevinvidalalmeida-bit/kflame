"""Summarize saved Newton/PTC failures without treating rejection as a solution."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def audit(path):
    source = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for run in source["runs"]:
        phases = []
        for stage in run.get("stages", []):
            for trace in stage.get("solver_trace", []):
                history = trace["history"]
                transients = [h for h in history if h.get("phase") == "transient"]
                inner = [i for h in history for i in h.get("iterations", [])]
                phases.append(dict(
                    label=trace["label"], n_points=trace["n_points"], width=trace["width"],
                    energy=trace["energy"], transport=trace["transport"], soret=trace["soret"],
                    successful_pseudo_steps=sum(bool(h["ok"]) for h in transients),
                    rejected_pseudo_steps=sum(not h["ok"] for h in transients),
                    schemes=dict(Counter(h["scheme"] for h in transients)),
                    newton_statuses=dict(Counter(i["status"] for i in inner)),
                    last_event={k: v for k, v in history[-1].items() if k != "iterations"}
                               if history else {},
                    rescues=[h for h in history if h.get("phase") in ("ptc_stagnation_rescue", "ptc_budget_rescue")],
                ))
        records.append(dict(
            solver=run["solver"], repetition=run["repetition"], accepted=run["accepted"],
            time_s=run.get("time_s"), Su=run.get("Su"), Finf=run.get("Finf"),
            phases=phases,
            stage_times=[dict(transport=s.get("transport_model"), total_s=s.get("total_time_s"),
                              target_corrector_s=s.get("target_corrector_time_s"))
                         for s in run.get("stages", [])],
        ))
    return dict(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                case=source["case"], options=source["options"], command=source["command"],
                source_sha256_by_module=source.get("source_sha256"), records=records,
                warning="Tracing adds overhead. These are diagnostic observations, not paired timing statistics. "
                        "Pseudo-step success is not steady convergence; numerical acceptance is not an independent mesh study.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
