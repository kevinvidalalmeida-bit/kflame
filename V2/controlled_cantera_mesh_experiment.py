"""
controlled_cantera_mesh_experiment.py

Experimento controlado para aislar ecuaciones vs numerica del solver:
- Usa la malla final de Cantera (auto=True) y su ancho final.
- Resuelve nuestro sistema en ESA malla fija (sin refine/expand).
- Ejecuta dos arranques:
  A) desde perfil final de Cantera
  B) desde nuestra conjetura inicial lineal en la misma malla
"""
from __future__ import annotations

import time
import numpy as np
import cantera as ct

from config import FlameCase
from problem import FreeFlameProblem
from species_backend import SpeciesBackend
from solver_cantera_auto import SolveOptions, _hybrid_newton
from residual import residual
from state import pack_state, unpack_state


def cantera_reference(case: FlameCase, loglevel: int = 0):
    gas = ct.Solution(case.mech)
    gas.TP = case.T_in, case.P
    gas.set_equivalence_ratio(case.phi, case.fuel, case.oxidizer)

    flame = ct.FreeFlame(gas, width=case.width)
    flame.transport_model = case.transport_model
    if hasattr(flame, "soret_enabled"):
        flame.soret_enabled = bool(getattr(case, "soret_enabled", False))
    if hasattr(flame, "flux_gradient_basis"):
        flame.flux_gradient_basis = str(getattr(case, "flux_gradient_basis", "molar"))
    flame.set_refine_criteria(
        ratio=case.ratio, slope=case.slope, curve=case.curve, prune=case.prune
    )

    t0 = time.perf_counter()
    flame.solve(loglevel=int(loglevel), auto=True)
    t_solve = time.perf_counter() - t0

    z = np.asarray(flame.grid, dtype=float)
    u = np.asarray(flame.velocity, dtype=float)
    T = np.asarray(flame.T, dtype=float)
    Y = np.asarray(flame.Y, dtype=float)
    return z, u, T, Y, t_solve


def run_hybrid_on_fixed_mesh(problem, opts: SolveOptions, x0: np.ndarray):
    x0 = problem.reset_bad_values(np.asarray(x0, dtype=float))
    _, T0, _ = unpack_state(x0, problem.n_points, problem.n_species)
    problem.solve_energy = True
    problem.setup_fixed_temperature(T_profile=T0)

    t0 = time.perf_counter()
    x_sol, ok, hist = _hybrid_newton(problem, x0, opts, label="controlled")
    dt = time.perf_counter() - t0

    u, T, Y = unpack_state(x_sol, problem.n_points, problem.n_species)
    out = {
        "ok": bool(ok),
        "time_s": float(dt),
        "steps": int(len(hist)),
        "Su": float(u[0]),
        "Finf": float(np.linalg.norm(residual(x_sol, problem), ord=np.inf)),
        "u": u,
        "T": T,
        "Y": Y,
    }
    return x_sol, out


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
        flux_gradient_basis="molar",
        soret_enabled=False,
        ratio=10.0,
        slope=0.8,
        curve=0.8,
        prune=-0.1,
    )

    z_ct, u_ct, T_ct, Y_ct, t_ct = cantera_reference(case, loglevel=0)
    Su_ct = float(u_ct[0])

    case_fix = FlameCase(
        mech=case.mech,
        fuel=case.fuel,
        oxidizer=case.oxidizer,
        phi=case.phi,
        T_in=case.T_in,
        P=case.P,
        width=float(z_ct[-1] - z_ct[0]),
        transport_model=case.transport_model,
        flux_gradient_basis=case.flux_gradient_basis,
        soret_enabled=case.soret_enabled,
        ratio=case.ratio,
        slope=case.slope,
        curve=case.curve,
        prune=case.prune,
    )

    problem = FreeFlameProblem(case_fix, n_points=int(z_ct.size))
    problem.z = z_ct.copy()
    problem.n_points = int(z_ct.size)
    problem.width = float(z_ct[-1] - z_ct[0])
    problem.backend = SpeciesBackend(problem)

    opts = SolveOptions(verbose=False)

    # A) Start from Cantera profile
    x_ct = pack_state(u_ct, T_ct, Y_ct)
    _, res_a = run_hybrid_on_fixed_mesh(problem, opts, x_ct)

    # B) Start from our linear guess (same fixed mesh)
    x_guess = problem.make_initial_guess(u_left_guess=opts.u_left_guess)
    _, res_b = run_hybrid_on_fixed_mesh(problem, opts, x_guess)

    res_a["T_linf_vs_ct"] = float(np.max(np.abs(res_a["T"] - T_ct)))
    res_a["u_linf_vs_ct"] = float(np.max(np.abs(res_a["u"] - u_ct)))
    res_a["Y_linf_vs_ct"] = float(np.max(np.abs(res_a["Y"] - Y_ct)))

    res_b["T_linf_vs_ct"] = float(np.max(np.abs(res_b["T"] - T_ct)))
    res_b["u_linf_vs_ct"] = float(np.max(np.abs(res_b["u"] - u_ct)))
    res_b["Y_linf_vs_ct"] = float(np.max(np.abs(res_b["Y"] - Y_ct)))

    print("=== CONTROLLED MESH/WIDTH TEST ===")
    print(
        f"Cantera: n={z_ct.size}, width={z_ct[-1]-z_ct[0]:.6f}, "
        f"Su={Su_ct:.9f}, t={t_ct:.2f}s"
    )

    print("Case A (start=Cantera profile):")
    print(
        f"  ok={res_a['ok']}  t={res_a['time_s']:.2f}s  steps={res_a['steps']}  "
        f"Su={res_a['Su']:.9f}  dSu={res_a['Su']-Su_ct:+.9e}  Finf={res_a['Finf']:.6e}"
    )
    print(
        f"  Linf: dT={res_a['T_linf_vs_ct']:.6e}  du={res_a['u_linf_vs_ct']:.6e}  "
        f"dY={res_a['Y_linf_vs_ct']:.6e}"
    )

    print("Case B (start=our linear guess):")
    print(
        f"  ok={res_b['ok']}  t={res_b['time_s']:.2f}s  steps={res_b['steps']}  "
        f"Su={res_b['Su']:.9f}  dSu={res_b['Su']-Su_ct:+.9e}  Finf={res_b['Finf']:.6e}"
    )
    print(
        f"  Linf: dT={res_b['T_linf_vs_ct']:.6e}  du={res_b['u_linf_vs_ct']:.6e}  "
        f"dY={res_b['Y_linf_vs_ct']:.6e}"
    )


if __name__ == "__main__":
    main()

