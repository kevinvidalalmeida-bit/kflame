# KFLAME — native free flames and FGM tables

KFLAME solves one-dimensional premixed free flames and builds flamelet-generated
manifold (FGM) tables in mixture-fraction/progress-variable coordinates (Z, c).
The production solver is CPU-native: **Cantera is not a runtime dependency**.
Initialization, NASA thermodynamics, reaction rates, molecular transport,
multicomponent/Soret transport and tabulation run locally.

Start with [examples/example.py](examples/example.py) for one flame or
[examples/example_fgm.py](examples/example_fgm.py) for a quick adaptive FGM table, or
[examples/example_fgm_adaptive_map.py](examples/example_fgm_adaptive_map.py) to reproduce
the thesis adaptive FGM map with its validated 44-flamelet schedule. The [public Python API](docs/api.md) exposes physical inputs and mesh
criteria through `kflame.solve_flame` and `kflame.generate_fgm`.
CPU/Numba is the production implementation; the rejected GPU strategy is
recorded in `DECISIONES_DESCARTADAS.md`.

## Install and run

Python 3.11 or later is required. Python 3.13 is recommended for new setups;
the supported CPU dependency matrix is recorded in
[compatibility](docs/compatibility.md). From a clone of this repository:

```sh
python -m pip install .
python -m kflame fgm --phi-values 0.7,0.9,1,1.1,1.4 --save-raw-profiles
```

The first execution compiles Numba kernels and can take noticeably longer.
Outputs go to `runs/fgm/`, never into the installed package.

```sh
# Force a cold sweep without saved profile seeds
python -m kflame fgm --phi-values 0.9,1,1.1 --disable-seed-cache --parallel-workers 1

# Native H2/Soret benchmark with native mixture-averaged bootstrap
python -m kflame soret --bootstrap --bootstrap-mesh-factor 2 --pressure-atm 10

# Export a generated table
python -m kflame export --npz runs/fgm/RUN_NAME/fgm_table.npz --all-species --single-flamelets
```

Use `python -m kflame COMMAND --help` for the full options. The installed `kflame`
console command is equivalent. Other commands are `refine-table`,
`validate-table`, `plot`, `compare` and `reference-fgm`.

Plotting and reference comparisons are separate optional installations:

```sh
python -m pip install ".[plot]"
python -m kflame plot --run-dir runs/fgm/RUN_NAME

python -m pip install ".[reference]"
python -m kflame compare --loglevel 1
```

`compare`, `reference-fgm`, `--transport-backend cantera-reference` and
`soret --no-native-only` explicitly select Cantera-dependent validation.
They are not silent fallback paths.

## Repository layout

```text
src/kflame/
  flame/        equations, nonlinear solver and adaptive mesh
  chemistry/    native thermodynamics, kinetics, transport and bundled data
  fgm/          generation, tabulation, export, plotting and validation
  reference/    optional Cantera comparisons
  benchmarks/   standalone Soret benchmark
tests/          regression tests and no-Cantera audit guard
benchmarks/     reproducible performance experiments, outside production
tools/          audit utilities for supplied outputs, outside production
examples/       ready-to-edit single-flame and FGM input files
docs/           architecture, migration and dated validation evidence
DECISIONES_DESCARTADAS.md   retained negative results and design decisions
```

Manuscripts, thesis figures, research-only scripts, archived experiments and
their data are kept locally under `.local/research/`, excluded from Git and
from the package. Generated `runs/`, `tmp/` and caches are also excluded.
The legacy loose-script entry points have been replaced by installed package
imports and commands; see [migration](docs/migration.md).

## Performance and numerical scope

The solver defaults to analytic block-tridiagonal Jacobians with frozen
transport coefficients, including thermal derivatives and partial column
refreshes during continuation. Strict acceptance and mesh checks, pivoted LU,
damped Newton and PTC with backward-Euler rescue remain in place.
Use `--no-analytic-spatial` for the finite-difference alternative, or add
`--analytic-chemistry` for the hybrid chemical route.
[Validation and timings](docs/validation/ANALYTIC_SPATIAL_20260921.md).

A historical three-pair CH4/GRI30 five-flame FGM test measured a median reduction from
18.44 to 15.89 seconds (13.8%) for the new perturbation preparation. Tables
were bit-identical; one pair was slower, so this is not a universal speed claim.
[Protocol and results](docs/validation/FGM_BATCH_VECTORIZATION_20260920.md).

Saved native seeds accelerate repeated sweeps; parallel workers are enabled
automatically only when every requested seed is available. A seeded regeneration
is not a cold-start benchmark. Thread defaults respect explicit environment
overrides. Seed identities include mechanism contents, not just filenames.

GRI30 and H2/O2 YAML inputs are included. Other NASA-7 ideal-gas mechanisms
need compatibility and validation checks; arbitrary Cantera features are not
claimed to be supported. The historical Bilger reference-stream convention
remains unchanged and is recorded in output metadata. Software independence
does not establish mesh-independent physical accuracy; see
[architecture](docs/architecture.md) and
[independence audit](docs/validation/NATIVE_ALL_STAGES_20260920.md).

## Development and validation

```sh
python -m pip install -e ".[reference,test]"
python -m unittest discover -s tests -v
python -m pytest -q
```

CI tests both Linux and Windows, plus a built distribution with Cantera and the
retired transport-fit archive blocked. Instructions for native-only validation
and benchmark protocols are in [CONTRIBUTING.md](CONTRIBUTING.md) and
[benchmarks/README.md](benchmarks/README.md).

The [refactoring audit](docs/validation/REFACTOR_AUDIT_20260920.md) records
preserved interfaces, exact before/after comparisons and remaining warnings.

Bundled mechanisms, collision data and adapted formulations retain their
[third-party attribution](src/kflame/chemistry/data/README.md).
An original-code redistribution license has not yet been selected.
