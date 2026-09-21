"""Run with: python examples/example.py (install KFLAME with the plot extra)."""
from pathlib import Path

import kflame


if __name__ == '__main__':
    flame = kflame.solve_flame(
        mechanism='gri30.yaml',
        temperature=300.0,                 # K
        pressure=101325.0,                  # Pa
        phi=1.0,
        fuel='CH4',
        oxidizer='O2:1, N2:3.76',           # Mole-basis stream, including diluent
        # Alternatively replace phi/fuel/oxidizer with ONE of:
        # X={'CH4': 1, 'O2': 2, 'N2': 7.52},  # Mole amounts, normalized internally
        # Y={'CH4': .055, 'O2': .22, 'N2': .725},  # Mass amounts
        # diluent='N2', dilution=0.1,        # Optional final mole fraction
        width=0.03,                        # Initial domain [m]; may expand
        initial_points=8,                  # Adaptive refinement follows
        # grid=[0, .003, .009, .012, .015, .021, .027, .03],
        transport='mixture-averaged',       # Or 'multicomponent'
        soret=False,                       # Supported with either model
        ratio=2.5, slope=0.04, curve=0.08, prune=0.003,
        max_points=1600,
        rtol=1e-4, atol=1e-9,
        max_time=180.0,                     # Time budget [s]
        species=('CH4', 'O2', 'CO2', 'H2O', 'OH'),
        plots=True,
        # Each run must use a fresh directory to protect previous results.
        output=None,                       # Automatic runs/flame/<timestamp>/
    )
    print(f"KFLAME: Su = {flame['Su']:.8f} m/s, {len(flame['z'])} adaptive nodes")
    print(f"Results: {Path(flame['output'])}")
