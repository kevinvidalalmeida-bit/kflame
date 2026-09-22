# Public Python interface

`import kflame` exposes `solve_flame` and `generate_fgm`. The examples are
ready-to-edit user input files. Both functions select the CPU-native solver.
Jacobian aging, linear algebra, PTC and damping remain internal defaults.
The default Jacobian now uses analytic thermal and spatial blocks with frozen
transport coefficients; see the [validation report](validation/ANALYTIC_SPATIAL_20260921.md).

The organization follows Cantera's separation of
[mixture state, transport and free-flame refinement](https://cantera.org/3.2/examples/python/onedim/adiabatic_flame.html).
KFLAME uses two keyword-only functions and dictionaries/paths for results.

| Input | Meaning and supported scope |
|---|---|
| `mechanism` | Bundled `gri30.yaml`, `h2o2.yaml`, or compatible NASA-7 ideal-gas YAML path; species come from the file |
| `temperature`, `pressure` | Fresh inlet K and constant pressure Pa |
| `phi`, `fuel`, `oxidizer` | Single-flame equivalence ratio and mole-basis streams; default phi=1 |
| `X` or `Y` | Alternative direct single-flame composition, string or dictionary; nonnegative amounts normalized internally |
| `diluent`, `dilution` | Optional single-flame dilution; final mole fraction of specified diluent, 0 <= dilution < 1 |
| `phis` | FGM: at least two positive, strictly increasing equivalence ratios; streams define mixture fraction |
| `width`, `initial_points` | Initial domain in metres and initial node count; final domain can expand automatically |
| `grid` | Single-flame explicit initial nodes, increasing from 0 to width; adaptive refinement remains active |
| `ratio`, `slope`, `curve`, `prune`, `max_points` | Adaptive spatial mesh criteria and node budget; no loss of acceptance at the cap |
| `transport`, `soret` | Mixture-averaged or multicomponent, each with Soret enabled or disabled; molar-gradient transport |
| `rtol`, `atol` | Single-flame steady weighted convergence tolerances; transient and final residual guards remain internal |
| `max_time` | Solver time budget per flame, seconds |
| `species`, `plots`, `output` | Species for CSV/plots, figure generation and new output directory |
| `export` | FGM: complete FlameMaster/CSV export, all mechanism species |

The physical problem is an adiabatic premixed freely propagating flame with
fixed fresh inlet state, constant pressure and zero-gradient species outlet.
Burners, imposed mass flow, radiation, wall losses, ionized transport and
arbitrary boundary types are not exposed. Reject unsupported keywords and
invalid physical combinations explicitly. Selecting output species never
reduces the chemical mechanism. Direct X/Y, arbitrary initial grids and steady
tolerance overrides belong to single flames; the FGM wrapper uses validated
equivalence-ratio sweeps and production tolerances. Dilute FGM streams by
including the diluent in the fuel/oxidizer composition.

`solve_flame` returns arrays `z, T, u, Y, rho, cp_mass, conductivity, qdot`,
species names, `Su`, acceptance report, runtime and output Path. `Y` has shape
(species, nodes). All fields remain available in NPZ. A failed numerical or
mesh certificate raises an error after saving diagnostics. `generate_fgm`
returns the output Path only after checking every flame and the table.

Figures require `pip install ".[plot]"`; Cantera is needed only for the
independent reference tests (`.[reference,test]`). Soret mixture-averaged uses
the distinct Cantera 3.2 mixture closure, not multicomponent coefficients with
a renamed flag. Its adapted formulation retains attribution in the bundled
transport license.
