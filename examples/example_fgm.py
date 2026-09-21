"""Run with: python examples/example_fgm.py; results under runs/fgm/."""
import kflame


if __name__ == '__main__':
    directory = kflame.generate_fgm(
        mechanism='gri30.yaml',
        temperature=300.0, pressure=101325.0,
        fuel='CH4', oxidizer='O2:1, N2:3.76',
        phis=(0.7, 0.9, 1.0, 1.1, 1.4),
        width=0.03, initial_points=8,
        transport='mixture-averaged', soret=False,
        ratio=2.5, slope=0.04, curve=0.08, prune=0.003,
        max_points=1600, max_time=180.0,
        species=('CO2', 'H2O'),             # Species in the established FGM plots
        plots=True, export=True,            # Adaptive-c figures, NPZ, CSV and .fla
        output=None,                        # Automatic runs/fgm/<timestamp>/
    )
    print(f'FGM tables, raw profiles and figures: {directory}')
