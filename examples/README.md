# User examples

Install from the repository root with `python -m pip install ".[plot]"`.
Run `python examples/example.py` or `python examples/example_fgm.py`.
Edit the keyword arguments in the example; import only `kflame`.

Every run creates a fresh folder under `runs/flame/` or `runs/fgm/`. These
outputs are ignored by Git. Specify `output='runs/my_case'` for a chosen name;
an existing directory is rejected to preserve previous calculations.

The single-flame example writes all species in `flame.npz`, selected species
in `profiles.csv`, solver diagnostics in `metadata.json`, and a four-panel
PDF/PNG showing temperature, heat release, species and adaptive mesh spacing.
The FGM example retains the established serif/Computer Modern figure style,
Z-c heatmaps, species curves, flame speed and adaptive progress-grid figures.
It exports the complete table and individual flamelets in FlameMaster format
and CSV. See [the public API](../docs/api.md) for units and supported inputs.
