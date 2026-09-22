"""Recreate the adaptive FGM map used in the thesis.

Run from the repository root after installing ``.[plot]``. The schedule has
five requested CH4/air flamelets and 39 certified bridge flamelets. It was
selected with a 1% leave-one-out interpolation-defect criterion.
"""
from pathlib import Path

from kflame.fgm.generate import main as generate_fgm
from kflame.fgm.plot import main as plot_fgm


if __name__ == "__main__":
    schedule = Path(__file__).with_name("fgm_adaptive_map_schedule.json")
    run_dir = generate_fgm([
        "--mech", "gri30.yaml",
        "--fuel", "CH4",
        "--oxidizer", "O2:1,N2:3.76",
        "--phi-schedule-json", str(schedule),
        "--width", "0.03",
        "--ratio", "2.5", "--slope", "0.04", "--curve", "0.08", "--prune", "0.003",
        "--max-grid-points", "1600",
        "--max-flame-time-s", "300",
        "--n-c", "241", "--c-fine", "2001", "--refine-bias", "5",
        "--progress-species", "CO2:1.0,H2O:1.0,CO:1.0,H2:0.5",
        "--indicator-species", "CH4,O2,CO2,H2O,CO,H2,OH",
        "--indicator-weight-grad", "1.0", "--indicator-weight-conc", "0.6",
        "--indicator-weight-temp", "0.8", "--indicator-weight-qdot", "0.4",
        "--save-raw-profiles", "--disable-seed-cache", "--parallel-workers", "1",
    ])
    plot_fgm(["--run-dir", str(run_dir), "--adaptive-map"])
    print(f"Thesis-style adaptive map: {run_dir / 'fgm_mapa_adaptativo.pdf'}")
