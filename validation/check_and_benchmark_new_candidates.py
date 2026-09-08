"""Check both mechanisms before timing; retain compilation cost separately."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
from pathlib import Path
import json
import time
import sys
import benchmark_cold_strategies as bench
from cold_generated_chemistry import GeneratedChemistry
from cold_exact_residual import exact_residual_context


def main():
    trace = []
    generated = GeneratedChemistry(bench.backend_module._eval_thermo_kinetics_sparse_numba_core, trace)
    started = time.perf_counter()
    checks = bench.kernel_checks(dict(
        baseline=bench.backend_module._eval_thermo_kinetics_sparse_numba_core,
        generated=generated))
    with exact_residual_context([]):
        pass  # Guarded source rewrite must succeed before a timed flame.
    report = dict(local_checks=checks, generated_trace=trace,
                  check_wall_s=time.perf_counter()-started,
                  note='Generated kernels remain warm in this process; compile time reported here, not hidden in solve.')
    dest = Path(__file__).with_name('new_candidates_local_checks.json')
    dest.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('LOCAL_CHECKS_SAVED', str(dest), flush=True)
    # The main benchmark records its literal command and snapshots the generator.
    bench.main()


if __name__ == '__main__':
    main()
