# KFLAME public API and mixture-averaged Soret — 2026-09-20

The installed module and console command are `kflame`; software is KFLAME.
The remote repository is `kevinvidalalmeida-bit/kflame`. Historical dated
reports retain their original paths and hashes. CPU/Numba remains the solver;
CuPy has no package extra or production branch.

## User entry points and organization

`examples/example.py` and `examples/example_fgm.py` execute using only the
public `kflame.solve_flame` and `kflame.generate_fgm` imports. Inputs, supported
boundaries, units and output files are documented in `../api.md`. No numerical
developer settings appear in the example inputs. CLI entry points still work.

The API reuses the existing native solver, FGM generator, export and plotting
code. Composition/grid/tolerance data are carried by the existing FlameCase,
including through the native transport bootstrap. No new stateful hierarchy
was added. Direct X and Y produce exactly equal z/T/u/Y profiles in the
H2/O2 custom-grid check; diluted input also completes successfully.

Shared benchmark settings and summaries moved to the existing Soret module;
independent ablation scripts retain their distinct hypotheses. Cache scripts
have descriptive names; output audits are under `tools/`. Redundant requirements
entry files were removed; dependencies are declared in `pyproject.toml`.
Old build products and package metadata are retained in local maintenance
storage, excluded from Git. Thesis files and figure scripts remain in
`.local/research/`; `DECISIONES_DESCARTADAS.md` remains tracked.

The real Numba contiguity warning in the Schur product was corrected with
`ascontiguousarray`, preserving the values and block operation. The existing
Schur/dense/reference test passes. This change has no quantified timing claim.

## New mixture-averaged Soret closure

Thermal diffusion follows
[Cantera 3.2 MixTransport](https://github.com/Cantera/cantera/blob/v3.2.0/src/transport/MixTransport.cpp)
with native viscosity, binary diffusion and C-star fits. Its pair contributions
receive the mixture mass-flux correction. The residual applies the thermal
term after the ordinary corrected mixture flux, as in
[Flow1D](https://github.com/Cantera/cantera/blob/v3.2.0/src/oneD/Flow1D.cpp).
Coefficients and the face temperature scale are frozen consistently in scalar,
Python-batched and compiled local Jacobians. BSD attribution is retained.

Tests compare coefficients with Cantera and the optional reference backend in
16 states: GRI30 and H2/O2, 1 and 10 atm, fresh/reactive/random/pure compositions,
300–2400 K. All pass rtol=2e-8 and atol=2e-14 kg/(m s); coefficient mass sums
are below 1e-18. Full/scalar/Python/compiled local residuals agree at all four
test nodes, including a negative Newton trial species.

Cold H2/O2 at 10 atm initially failed with Soret active from the first iterate.
The existing native mixture-without-Soret bootstrap was extended to supply a
seed for this closure, followed by the complete requested Soret corrector.
Acceptance, final mesh criteria and residual guards were not relaxed.

## Complete-flame validation

H2/air, phi=1, 300 K, width initially 0.03 m, ratio=2.5, slope=.04, curve=.08,
prune=.003. Both solvers adapt their meshes/domains independently. All four
native solves and Cantera comparisons succeeded with the bootstrap strategy.

| Mechanism | atm | KFLAME Su, m/s | Cantera Su, m/s | Relative Su difference | E2(T) | Largest E2(Y), active species |
|---|---:|---:|---:|---:|---:|---:|
| H2/O2 | 1 | 2.12743314 | 2.12502062 | 0.1135% | 0.0581% | 1.2553% |
| H2/O2 | 10 | 1.32895989 | 1.32625238 | 0.2041% | 0.0112% | 0.7702% |
| GRI30 | 1 | 2.12727895 | 2.12495798 | 0.1092% | 0.0544% | 0.5997% |
| GRI30 | 10 | 1.32919981 | 1.32628440 | 0.2198% | 0.0092% | 0.8529% |

Profiles are aligned at 25% of the common temperature rise, interpolated onto
the union of overlapping meshes and compared using spatial trapezoidal L2
norms. Active reference species peak at >=1e-5. Maximum composition sum error
is 2.51e-10. These results establish implementation agreement for these cases,
not mesh-independent accuracy or an arbitrary mechanism/pressure guarantee.

Reproduce: `python tools/validate_mixture_soret.py --output tmp/new-soret-check`.
Saved profiles and case summaries from this audit remain in
`tmp/mix_soret_bootstrap_validation_20260920/`; the failed initial campaign
remains in `tmp/mix_soret_validation_20260920/`. Timings are single observations
and included overlapping test activity; no speedup is inferred from them.

The native-only public FGM integration with H2/O2, phi=(.9,1.1), Soret mixture
transport and 241 adaptive c coordinates accepts both rows and validates the
table with Cantera imports blocked. The default CH4 five-flame example also
accepts all rows, exports all 53 species, individual flamelets and CSV, and
generates the established FGM figures. The single-flame example creates an
inspected four-panel PNG/PDF with a dedicated adaptive-cell-spacing panel.

The interrupted CPU/GPU campaign is not a completed benchmark. Its completed
first FGM CPU pair measured 16.164 s (native Python 3.11) vs 64.189 s (Cantera
3.11), with native 3.13 at 16.161 s. Those are single observations. The earlier
13.8% vectorization result retains its own paired protocol and attribution.
