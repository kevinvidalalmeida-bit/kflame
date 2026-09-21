# Performance experiments

Install the package in editable mode before running these scripts from the
repository root. They are intentionally separate from the installed solver.

- `benchmark_fgm_batches.py --output-root tmp/fgm-batches --pairs 3`: complete
  five-flame FGM comparison of vectorized and legacy perturbation preparation.
- `benchmark_thermal_reuse.py --output tmp/thermal.json --pairs 3`: native thermal
  reuse versus the general evaluator; original isolated prototype is archived.
- `benchmark_molecular_fit_cache.py --output tmp/molecular.json --pairs 3`:
  multicomponent molecular-fit cache versus recomputing the same native values;
  this does not disable the separate mixture-averaged cache.
- `benchmark_transport_data_cache.py --output tmp/transport.json --pairs 3`:
  pair/conductivity data caches versus recomputing the same native values.
- `profile_native_stages.py --output tmp/stages.json`: inclusive and exclusive
  cProfile costs; instrumentation times are not production speed benchmarks.
- `benchmark_native_vs_cantera.py`: matched KFLAME/Cantera FGM and individual
  cold-flame timing; specify the two Python executables to compare.

For example: `python benchmarks/benchmark_fgm_batches.py --output-root tmp/batch-check`.
Use a new output folder, fixed thread settings and no concurrent heavy jobs.
Include every failure and distinguish cold solves from cached regenerations.
Three timing pairs are a local screening result, not a universal speed guarantee.
Historical results live in `docs/validation/`; local raw profiles stay ignored.

`../tools/audit_native_outputs.py --help` is an audit utility, not a benchmark:
it summarizes supplied native integration outputs with Cantera blocked.
