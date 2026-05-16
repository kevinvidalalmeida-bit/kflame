from pathlib import Path

from config import FlameCase
from cantera_ref import solve_free_flame, save_reference_npz, save_case_json, summarize_solution


def main():
    case = FlameCase(
        mech="gri30.yaml",
        fuel="CH4",
        oxidizer="O2:1.0, N2:3.76",
        phi=1.0,
        T_in=300.0,
        P=101325.0,
        width=0.03,
        transport_model="mixture-averaged",
        ratio=3.0,
        slope=0.07,
        curve=0.14,
        prune=0.0,
        loglevel=1,
        auto=True,
    )

    flame, data = solve_free_flame(case)

    outdir = Path("outputs/reference")
    save_reference_npz(data, outdir / "freeflame_ref.npz")
    save_case_json(data, outdir / "freeflame_ref_meta.json")

    print(summarize_solution(data))


if __name__ == "__main__":
    main()