# Package migration — 20 September 2026

Install with `python -m pip install .` (or `-e .` for development). No source
directory needs to be added to PYTHONPATH. The numerical solver retains its
KFLAME lineage in output metadata; the import namespace is now `kflame`.

| Previous entry point | Replacement |
|---|---|
| FGM/scripts/generate_fgm_tables_native.py | `python -m kflame fgm` |
| FGM/scripts/export_fgm_to_flamelet.py | `python -m kflame export` |
| FGM/scripts/plot_fgm_figures.py | `python -m kflame plot` |
| FGM/scripts/refine_fgm_schedule.py | `python -m kflame refine-table` |
| FGM/scripts/validate_fgm_holdouts.py | `python -m kflame validate-table` |
| Legacy `V2/benchmark_soret_native.py` | `python -m kflame soret` |
| Legacy `V2/run_saved_comparison.py` | `python -m kflame compare` |
| FGM/scripts/generate_fgm_tables_cantera.py | `python -m kflame reference-fgm` |
| Legacy `V2/config.py`, `problem.py`, `solver.py` | `kflame.flame.config`, `.problem`, `.solver` |
| Legacy `V2/materiales/*.py` | `kflame.chemistry.*` (short names without `_native`) |
| validation/*.py benchmarks | `benchmarks/` |
| validation/*.md and *.json | `docs/validation/` |
| Legacy `V2/PIPELINE.md` | `docs/architecture.md` |

Historical reports may retain the former `V2` label, dated path strings and source
hashes; those describe the experiment's snapshot, not the renamed files. Use
the installed commands above for new runs. Some historical raw campaigns were
already local-only; compact public summaries are not replacements for raw data.

The first call after migration recompiles Numba at the new module locations.
Old cached flame seeds are not assumed portable across mechanism path changes;
the first sweep regenerates them. Physical equations, acceptance and refinement
criteria have not been relaxed. Generated output defaults now use `runs/`.

## Local research material

The manuscript, bibliography and thesis figures were moved together to
`.local/research/thesis/`. The latest PDF outputs are under
`.local/research/output/`. Thesis analysis/figure scripts are in
`.local/research/scripts/`, their saved FGM campaigns in
`.local/research/FGM/resultados/`, and other results/evidence alongside them.
The standalone root draft is preserved in `.local/research/drafts/`.

These files are excluded from the current Git tree and the wheel. They were
preserved, not deleted. The deprecated transport-fit archive and isolated
prototype are also retained locally. A source/manuscript snapshot before this
migration is in `.local/maintenance/before-layout/` on the maintainer's machine.
Existing Git history is unchanged and can still contain earlier thesis files.
