# Cold-start strategy audit — 2026-09-06

Scope: test every strategy proposed in the preceding review, with and without
Soret, preserving the existing production solver as the baseline. Experiments
live in `validation/`; no candidate is enabled in the production pipeline.

## Candidates and required evidence

1. `thermal`: share exact NASA/Arrhenius/equilibrium/Troe temperature factors
   within the FD Jacobian batch. Concentrations, mixture properties, third bodies
   and composition-dependent falloff are reevaluated. Cache construction and
   temperature grouping are inside the measured solve.
2. `blocked`: limit temporary T/Y perturbation batches to approximately 2 MiB.
   This bounds temporary storage, not the full Jacobian output. It is an isolated
   memory-layout ablation, independent of the thermal candidate.
   `streamed` extends this to direct Jacobian assembly: each batch is consumed
   into the final tridiagonal blocks before the next is evaluated, with no global
   concatenation of perturbed thermochemistry outputs. Its four-case trial is complete.
3. `powers`: share equilibrium concentration powers for distinct delta-nu.
   Grouping cost is included. The chemical expression is unchanged.
4. `serial` and `granularity`: first measure the serial reference, then fit one
   serial/parallel crossover to kernel workloads. No pressure-specific policy.
5. `mesh-budget`: interpolation discrepancy supplies an exploratory intermediate
   nonlinear-error budget. It is NOT a proven spatial error estimator. Unchanged
   refinement and a strict final correction are required. Raw final metrics are
   checked again outside the experimental acceptance hook.
6. `two-grid`: two-grid FAS correction before the standard post-refinement solve.
   The coarse forcing is F_H(R x_h) - R F_h(x_h); restriction retains boundaries
   and the anchor. Full transport residuals decide fine-grid descent. This is a
   two-grid prototype, not an implemented recursive multigrid solver.

## Protocol

- No saved flame seeds. Full-stack warmup per variant is discarded as a seed.
- Cold flame initialization is distinct from cold JIT compilation.
- Initial screening followed by seven alternating repetitions for candidates
  considered for adoption; preserve failures rather than hiding them.
- GRI30, 300 K, phi=1, initial domain 0.03 m, unchanged expansion policy.
- CH4 mixture-averaged/no Soret and H2 multicomponent/Soret at 1 and 10 atm.
- Refinement slope/curve 0.04/0.08 at 1 atm and 0.01/0.02 at 10 atm, shared by
  all variants in each case. Residual guard 1e4 and final weighted norm <=1.
- Compare profiles and meshes, not just flame speed and acceptance messages.
- Within-V2 ablation times do not constitute new V2–Cantera measurements.

## Evidence so far

- `resultados/cold_strategies/20260906_012047`: local kernel checks passed for
  h2o2 and GRI30, at 1 and 10 atm, six temperatures including the NASA branch
  boundary, three progress states, and positive/negative-trace Newton states.
  Thermal, powers and serial variants matched the baseline outputs exactly in
  these checks. This is not full validation of all possible mechanisms.
- `resultados/cold_strategies/20260906_012112`: completed one-pair screening at
  1 atm, CH4/no Soret and H2/Soret. All ten measured solves accepted; same mesh
  and fields. Fully serial execution was slower in both cases.
- `resultados/cold_strategies/20260906_012403`: seven-pair campaign completed for
  baseline/thermal/blocked/powers at 1 atm. All 56 measured flames accepted and
  have bitwise-equal saved fields relative to the baseline. Thermal median:
  CH4 4.1501 -> 3.8242 s; H2/Soret 6.0027 -> 5.4141 s. Paired ratio confidence
  intervals exclude one. Blocked and powers intervals include one.
- `validation/kinetics_granularity_calibration.json`: the measured crossover
  selected cutoff zero (parallel at every tested workload). Adding dispatch is
  therefore not an algorithmic acceleration on this machine.
- `resultados/cold_strategies/20260906_013037`: aborted harness run, NOT a solver
  regression. An additional absolute mass-flow threshold of 1e-6 rejected the
  unchanged baseline (its diagnostic is 1.2356e-6). That new threshold was not
  part of the existing acceptance policy and has been removed from the harness.
  Conservation remains recorded and compared, not silently rounded or hidden.
- `resultados/cold_strategies/20260906_013216`: one-pair 1-atm screening of
  granularity, mesh-budget and two-grid. All measured states pass the unchanged
  final norm/residual limits and aligned-profile accuracy gates. Two-grid is
  slower in both cases; mesh-budget shows no clear gain. Granularity's cutoff
  zero makes it a same-algorithm control: timing fluctuation is not a speedup.
- `validation/cold_strategy_audit_20260906.json`: complete first-stage statistics
  and aligned-profile gates (0.1% Su, 0.5% T, 2% active Y, 5% heat release).
- `resultados/cold_strategies/20260906_013450`: 10-atm screening launched for
  baseline/thermal/blocked/powers/mesh-budget/two-grid, CH4 and H2/Soret.
  It has now completed: all twelve measured flames passed final acceptance and
  profile gates. The thermal screening times were CH4 26.4978 -> 24.5435 s and
  H2/Soret 48.6431 -> 45.8889 s. These are single pairs, not confirmation.
- `resultados/cold_strategies/20260906_105753`: seven-pair confirmation at 10 atm,
  with exact source snapshots. CH4 is complete: baseline median 23.6311 s,
  thermal 24.6731 s, paired ratio interval [0.8894, 1.1200]. This does NOT support
  global adoption. Large temporal drift makes the promising screening insufficient.
  H2/Soret completed: 51.4042/47.4205 s, paired ratio interval [0.9046, 1.0169].
  That interval also includes one. No global promotion.
- The full unit suite completed successfully: 60 tests, including source rewrite
  guards, preservation of the FAS fixed point and anchor, strict final norm gates,
  and a test forbidding a relaxed budget on an unmarked grid.

The first round is complete. Direct streamed assembly completed in campaign
`20260906_111817`, without general savings and with identical saved profiles.
All six measured campaigns are consolidated in `cold_strategy_audit_20260906.json`;
122 measured profiles passed `cold_artifact_integrity_20260906.json`.
See `COLD_FOLLOWUP_20260906.md` for the separate second round.

### Terminal checkpoint

Both confirmation and streamed-assembly processes finished successfully. No
production implementation was promoted. Inconclusive and negative outcomes,
particularly the CH4/10-atm confirmation, remain part of the final decision.

## Interpretation safeguards

- All fourteen measured thermal runs in the seven-pair 1-atm campaign have
  successful precompute counts equal to the number of full Jacobian builds
  (84 for CH4, 55+12 for the H2 bootstrap/final stages). The results were not
  obtained by silently falling back to the original per-column path.
- Median Jacobian thermochemistry time decreased from 0.9914 to 0.6962 s in
  CH4 and from 1.3112 to 0.9679 s in H2/Soret. LU counts/algorithm are unchanged.
  Instrumented times are nested; do not sum them as exclusive wall-time shares.
- The two-grid prototype did perform corrections: nine of nine accepted fine
  residual descents in CH4, nine of fourteen in H2/Soret, with no recorded
  exceptions. Their added cost and nonlinear trajectory outweighed the benefit
  at 1 atm. This does not disprove other multigrid formulations.
- The only production-adjacent change in this audit is ignoring regenerable
  `resultados/cold_strategies/` data in Git. Full solver/kernel experiments remain
  outside `V2/`; existing user edits have not been reset or overwritten.

## Second-round terminal checkpoint

See `COLD_FOLLOWUP_20260906.md`: all five follow-up campaigns finished,
52 measured profiles independently audited, 65 unit tests passed. The
compiled-LU screening (`20260906_165412`) preserves bitwise profiles but
does not establish general savings; CH4/10 atm is slower in the single
pair. Dependencies show a favorable three-pair interval in two of four
cases, not global evidence. Transport action and the tested homotopy are
not adopted. No production source changed in this round, no fresh Cantera
timing claim, and no benchmark process remains running.

### Subsequent user-requested archive decision

All candidates, including inconclusive dependencies and compiled-LU, are now
archived rather than active follow-ups. The default benchmark builds/checks
only the baseline kernel; nonbaseline variants require `--reproduce-archived`
for historical reproduction. No data or source snapshots were deleted.
67 tests pass, including the two new archive-policy guards. New untested
proposals are listed separately in `NEXT_IMPROVEMENTS.md`.
