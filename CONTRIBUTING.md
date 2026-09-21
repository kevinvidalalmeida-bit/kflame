# Contributing

Install the package in editable mode with `python -m pip install -e ".[reference,test]"`.
Run `python -m unittest discover -s tests -v` or `python -m pytest -q`.
Use fully qualified `kflame` imports: do not mutate sys.path inside production code.
Runtime dependencies and optional extras are declared in `pyproject.toml`.
User scripts should import `kflame.solve_flame` or `kflame.generate_fgm`; see
`examples/` and `docs/api.md`. Developer solver tuning stays out of examples.

Keep Cantera in explicit reference tools. For an independence check, install
only `python -m pip install .` in a fresh virtual environment and set PYTHONPATH
to the absolute `tests/no_cantera` directory before running tests or commands.
The guard rejects both Cantera imports and the retired exported-fit archive.
Reference comparisons skip explicitly when the optional dependency is absent.

Performance changes need paired whole-workflow timings, excluded/declared JIT
warmups, identical input seeds, fixed thread counts, all acceptance outcomes,
and physical-field comparisons. A faster isolated kernel is not sufficient.
Read `DECISIONES_DESCARTADAS.md` before reintroducing an archived strategy.
Do not change acceptance thresholds to make a benchmark pass.

Keep generated results in `runs/` or `tmp/`. Manuscripts, research-only plots,
private data and maintenance backups belong in `.local/`, which is not published.
Never use `git add -f` on these directories. Moving a tracked file out of the
current tree does not erase its earlier Git history.

Build a distribution with `python -m pip wheel . --no-deps --wheel-dir dist`.
Test that wheel in a fresh environment, including bundled YAML/collision data,
outside the checkout's working directory. Do not publish to a package registry
or select an original-code license without the maintainer's decision.
