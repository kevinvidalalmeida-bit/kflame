# Compatibility matrix

KFLAME supports Python 3.11 and newer. The project constrains its CPU
dependencies to versions that work together; it does not upgrade the user's
global Python installation. Use a virtual environment per matrix below.

| Component | Python 3.11 baseline | Python 3.13 current validated stack |
|---|---:|---:|
| Python | 3.11.0 | 3.13.x |
| NumPy | 2.4.6 | 2.4.6 |
| SciPy | 1.17.1 | 1.17.1 |
| Numba | 0.65.1 | 0.67.x |
| llvmlite | 0.47.0 | 0.49.x |
| PyYAML | 6.0.3 | 6.0.3 |
| Cantera (optional reference) | 3.2.0 | 3.2.0 |
| Matplotlib (optional plots) | 3.11.1 | 3.11.2 |
| pytest (optional tests) | 9.0.3 | 9.1.1 |

The NumPy 2.4 and SciPy 1.17 lines work with both supported Numba lines.
`pip install .` resolves the CPU stack; `pip install ".[reference,test]"` adds
the reference and testing tools.

KFLAME is CPU/Numba by design. CuPy/GPU perturbation experiments are retained
only as a rejected strategy in `DECISIONES_DESCARTADAS.md`: their transfer and
dispatch costs did not justify a second numerical backend for this workload.

The native/reference timing protocol is
`benchmarks/benchmark_native_vs_cantera.py`. It measures subprocess wall time,
records first-use JIT separately, and compares cold, unseeded FGM and
individual flames with fixed CPU thread counts.
