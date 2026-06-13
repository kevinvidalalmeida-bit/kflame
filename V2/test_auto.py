import time
import numpy as np
from config import FlameCase
from problem import FreeFlameProblem
from species_backend import SpeciesBackend
from solver import solve_free_flame, SolveOptions
from state import unpack_state

def main():
    case = FlameCase(
        mech="gri30.yaml",
        fuel="CH4",
        oxidizer="O2:1.0, N2:3.76",
        phi=1.0, T_in=300.0, P=101325.0, width=0.03,
        transport_model="mixture-averaged",
        flux_gradient_basis="molar",
        soret_enabled=False,
        ratio=10.0, slope=0.8, curve=0.8, prune=-0.1,
    )
    
    opts = SolveOptions(
        verbose=True,
        refine_ratio=case.ratio,
        refine_slope=case.slope,
        refine_curve=case.curve,
        refine_prune=case.prune,
    )
    
    p2 = FreeFlameProblem(case, n_points=8)
    p2.backend = SpeciesBackend(p2)
    
    t0 = time.perf_counter()
    x_sol, ok, rpt = solve_free_flame(p2, options=opts)
    t_total = time.perf_counter() - t0
    
    print(f"\n  converged    = {ok}")
    print(f"  n_points     = {p2.n_points}")
    print(f"  tiempo [s]   = {t_total:.1f}")

if __name__ == "__main__":
    main()
