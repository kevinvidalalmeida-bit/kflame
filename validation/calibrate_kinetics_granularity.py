"""Measure serial/parallel crossover by workload, not flame pressure."""
import json
import time
from unittest.mock import patch
from types import SimpleNamespace
from benchmark_cold_strategies import ROOT, backend_module, np, ct, _resolve_mechanism
from cold_strategy_kernels import build_candidates


def main():
    kernels = build_candidates()
    case = SimpleNamespace(mech=_resolve_mechanism('gri30.yaml'), transport_model='mixture-averaged', soret_enabled=False)
    backend = backend_module.NativeSpeciesBackend(SimpleNamespace(case=case, P=ct.one_atm))
    gas = ct.Solution(case.mech)
    gas.TP = 300, ct.one_atm
    gas.set_equivalence_ratio(1., 'CH4', 'O2:1,N2:3.76')
    fresh = gas.Y.copy()
    gas.equilibrate('HP')
    captured = []
    rows = []
    for n in (1, 2, 4, 8, 16, 32, 64, 256):
        ts = np.linspace(500, 2300, n)
        y = (fresh[:, None]*(1-np.linspace(0, 1, n)) + gas.Y[:, None]*np.linspace(0, 1, n)).copy()
        with patch.object(backend_module, '_eval_thermo_kinetics_sparse_numba_core', lambda *a: captured.append(a)):
            backend.eval_grid_thermo_kinetics_into(ts, y, np.empty_like(y), np.empty_like(y))
        args = captured.pop()
        for name in ('baseline', 'serial'):
            kernels[name](*args)  # all layouts compiled before any measurement
        times = dict(baseline=[], serial=[])
        for repetition in range(9):
            for name in (('baseline', 'serial') if repetition % 2 == 0 else ('serial', 'baseline')):
                start = time.perf_counter()
                for _ in range(50):
                    kernels[name](*args)
                times[name].append((time.perf_counter()-start)/50)
        rows.append(dict(nodes=n, medians_s={key: float(np.median(value)) for key, value in times.items()}, raw_s=times))
    # One threshold fit to the whole measured workload range, equally weighted
    # in relative cost; not selected from a favorable flame benchmark.
    thresholds = [0]+[r['nodes'] for r in rows]
    cost = lambda threshold: sum(r['medians_s']['serial' if r['nodes'] <= threshold else 'baseline'] /
                                 r['medians_s']['baseline'] for r in rows)
    cutoff = min(thresholds, key=cost)
    result = dict(policy='9 alternating samples of 50 kernel calls; separate layout warmup',
                  selected_serial_cutoff=cutoff, rows=rows,
                  source_sha256=kernels['serial'].audit_sha256)
    path = ROOT/'validation/kinetics_granularity_calibration.json'
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
