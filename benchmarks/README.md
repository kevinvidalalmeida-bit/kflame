# Performance experiments

Install the package in editable mode before running these scripts from the
repository root. They are intentionally separate from the installed solver.

- `benchmark_fgm_batches.py --output-root tmp/fgm-batches --pairs 3`: complete
  five-flame FGM comparison of vectorized and legacy perturbation preparation.
- `benchmark_thermal_reuse.py --output tmp/thermal.json --pairs 3`: native thermal
  reuse versus the general evaluator; original isolated prototype is archived.
- `benchmark_transport_cache.py --output tmp/transport.json --pairs 3`: exact
  pair/conductivity caches versus recomputing the same native values.
- `profile_native_stages.py --output tmp/stages.json`: inclusive and exclusive
  cProfile costs; instrumentation times are not production speed benchmarks.
- `benchmark_native_dependencies.py --output tmp/molecular.json`: historical
  multicomponent molecular-fit cache ablation. It does not disable the newer
  mixture-averaged fit cache and should not be called a complete cache ablation.
- `check_native_all_stages.py --help`: compare saved integration results and
  emit a compact native-only audit. It needs the specified previous outputs.

For example: `python benchmarks/benchmark_fgm_batches.py --output-root tmp/batch-check`.
Use a new output folder, fixed thread settings and no concurrent heavy jobs.
Include every failure and distinguish cold solves from cached regenerations.
Three timing pairs are a local screening result, not a universal speed guarantee.
Historical results live in `docs/validation/`; local raw profiles stay ignored.
