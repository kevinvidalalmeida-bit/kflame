# User examples

Install from the repository root with `python -m pip install ".[plot]"`.
Run `python examples/example.py` for one flame or
`python examples/example_fgm.py` for a quick five-flamelet FGM table.

`python examples/example_fgm_adaptive_map.py` recreates the adaptive FGM map
used in the thesis. It solves the validated 44-row schedule: five requested
CH4/air flamelets plus 39 bridge flamelets. The generated
`fgm_mapa_adaptativo.pdf` contains:

- CO2 and CO profiles for the rows actually solved;
- qdot and absolute omega_c reconstructed only with the table's bilinear
  interpolant in `(Z*, c)`;
- dotted vertical lines at every solved row, so interpolation is not presented
  as an extra flamelet calculation.

The bridge schedule was selected with the leave-one-out maximum relative L2
interpolation defect at a 1% target. To derive a new bridge proposal from any
certified table, run:

```sh
python -m kflame refine-table \
  --table runs/fgm/RUN_NAME/fgm_table.npz \
  --output runs/fgm/RUN_NAME/next_schedule.json \
  --target-defect 0.01 --max-bridges 99
```

Pass the resulting JSON to `kflame fgm --phi-schedule-json ...`, certify the
new table, and repeat the defect check until no bridge is proposed.

Every run creates a fresh folder under `runs/flame/` or `runs/fgm/`, ignored by
Git. Specify `output='runs/my_case'` for a chosen name; an existing directory
is rejected to preserve previous calculations.

The single-flame example writes all species in `flame.npz`, selected species
in `profiles.csv`, solver diagnostics in `metadata.json`, and a four-panel
PDF/PNG showing temperature, heat release, species and adaptive mesh spacing.
The quick FGM example exports the complete table, individual flamelets and
figures. See [the public API](../docs/api.md) for units and supported inputs.
