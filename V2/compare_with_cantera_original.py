# -*- coding: cp1252 -*-
"""
compare_with_cantera_original.py

Compara el residual propio (nuevo layout por-punto) con el de Cantera
evaluado en la misma malla y solución convergida.

Uso:
    python compare_with_cantera_original.py [--run-ours] [--ct-loglevel N]
"""
from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import replace

import numpy as np
import cantera as ct

from config import FlameCase
from problem import FreeFlameProblem
from species_backend import SpeciesBackend
from residual import residual
from state import pack_state, unpack_state


# ---------------------------------------------------------------------------
#  Solución de referencia con Cantera
# ---------------------------------------------------------------------------
def _cantera_solve(case: FlameCase, ct_loglevel: int = 1):
    gas = ct.Solution(case.mech)
    gas.TP = case.T_in, case.P
    gas.set_equivalence_ratio(case.phi, case.fuel, case.oxidizer)

    flame = ct.FreeFlame(gas, width=case.width)
    flame.transport_model = case.transport_model
    if hasattr(flame, "soret_enabled"):
        flame.soret_enabled = bool(getattr(case, "soret_enabled", False))
    if hasattr(flame, "flux_gradient_basis"):
        flame.flux_gradient_basis = str(getattr(case, "flux_gradient_basis", "molar"))
    flame.set_refine_criteria(ratio=case.ratio, slope=case.slope,
                              curve=case.curve, prune=case.prune)
    t0 = time.perf_counter()
    flame.solve(loglevel=int(ct_loglevel), auto=True)
    t_solve = time.perf_counter() - t0

    z = np.asarray(flame.grid, dtype=float)
    u = np.asarray(flame.velocity, dtype=float)
    T = np.asarray(flame.T, dtype=float)
    Y = np.asarray(flame.Y, dtype=float)
    mdot = float(flame.density[0] * flame.velocity[0])

    # Extraer residual de Cantera
    t_ev0 = time.perf_counter()
    flame.eval(rdt=0.0)
    dom_idx = flame.domain_index("flame")
    n_comp = flame.flame.n_components
    n_pts = flame.flame.n_points
    comp_names = [flame.flame.component_name(i) for i in range(n_comp)]

    R_ct = np.empty((n_comp, n_pts), dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for i in range(n_comp):
            for j in range(n_pts):
                R_ct[i, j] = float(flame.work_value(dom_idx, i, j))
    t_eval = time.perf_counter() - t_ev0

    return comp_names, R_ct, z, u, T, Y, mdot, t_solve, t_eval


# ---------------------------------------------------------------------------
#  Residual propio en la malla de Cantera
# ---------------------------------------------------------------------------
def _ours_residual(case: FlameCase, z, u, T, Y, mdot):
    problem = FreeFlameProblem(case, n_points=int(z.size))
    problem.z = np.asarray(z, dtype=float).copy()
    problem.n_points = int(z.size)
    problem.width = float(z[-1] - z[0])
    problem.u_fixed = np.asarray(u, dtype=float).copy()
    problem.T_fixed = np.asarray(T, dtype=float).copy()
    problem.Y_ref = np.asarray(Y, dtype=float).copy()
    problem.mdot_fixed = float(mdot)
    problem.backend = SpeciesBackend(problem)
    problem.solve_energy = True
    problem.setup_fixed_temperature(T_profile=problem.T_fixed)

    x = pack_state(np.asarray(u), np.asarray(T), np.asarray(Y))
    t0 = time.perf_counter()
    F = residual(x, problem)
    t_eval = time.perf_counter() - t0

    # Reshape a (n_vars_per_point, n_pts) para comparación fácil
    nv = problem.n_vars_per_point
    F_m = F.reshape(problem.n_points, nv).T   # (nv, n_pts)
    return problem, F_m, t_eval


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-ours", action="store_true",
                        help="Ejecutar también nuestro solver auto")
    parser.add_argument("--ct-loglevel", type=int, default=1,
                        help="Nivel de log de Cantera (0=silencioso, 1+=detallado)")
    args = parser.parse_args()

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

    print("Resolviendo con Cantera...")
    comp_names, Rc, z, u, T, Y, mdot, t_ct, t_ct_ev = _cantera_solve(
        case, ct_loglevel=args.ct_loglevel
    )
    ct_Finf = float(np.max(np.abs(Rc)))
    ct_su = float(u[0]) if u.size else float("nan")
    ct_n_points = int(z.size)

    print("\n=== Cantera (resumen) ===")
    print("  converged    = True")
    print(f"  n_points     = {ct_n_points}")
    print(f"  Su (~u[0])   = {ct_su:.6f} m/s")
    print(f"  ||F||inf       = {ct_Finf:.6e}")
    print(f"  tiempo [s]   = {t_ct:.1f}")

    print("Evaluando residual propio en malla de Cantera...")
    problem, Ro, t_ours = _ours_residual(case, z, u, T, Y, mdot)

    def inf_int(v):
        return float(np.max(np.abs(v[1:-1]))) if v.size > 2 else float(np.max(np.abs(v)))

    print("\n=== Comparación de residuales ===")
    print(f"  n_points          = {problem.n_points}")
    print(f"  ||F_ct||inf         = {np.max(np.abs(Rc)):.6e}")
    print(f"  ||F_ours||inf       = {np.max(np.abs(Ro)):.6e}")
    print(f"\n  Cantera solve [s] = {t_ct:.3f}")
    print(f"  Cantera eval  [s] = {t_ct_ev:.3f}")
    print(f"  Ours eval     [s] = {t_ours:.4f}")

    # Continuidad
    i_u = comp_names.index("velocity") if "velocity" in comp_names else None
    if i_u is not None:
        print(f"\n  continuidad  ct interior inf = {inf_int(Rc[i_u, :]):.6e}")
    print(f"  continuidad  ours int  inf = {inf_int(Ro[0, :]):.6e}")

    # Energía
    i_T = comp_names.index("T") if "T" in comp_names else None
    if i_T is not None:
        print(f"  energía      ct interior inf = {inf_int(Rc[i_T, :]):.6e}")
    print(f"  energía      ours int  inf = {inf_int(Ro[1, :]):.6e}")

    # Especies top 10
    rows = []
    for k, sp in enumerate(problem.species_names):
        if sp in comp_names:
            ic = comp_names.index(sp)
            rows.append((max(inf_int(Rc[ic, :]), inf_int(Ro[2+k, :])),
                         sp, inf_int(Rc[ic, :]), inf_int(Ro[2+k, :])))
    rows.sort(reverse=True)
    print("\n  Top especies por residual interior:")
    for _, sp, rc, ro in rows[:10]:
        print(f"    {sp:8s}  cantera={rc:.3e}   ours={ro:.3e}")

    if args.run_ours:
        from solver_cantera_auto import solve_free_flame, SolveOptions
        print("\n=== Nuestro solver (malla propia) ===")
        case_ours = replace(case, width=float(z[-1] - z[0]))
        p2 = FreeFlameProblem(case_ours, n_points=8)
        p2.backend = SpeciesBackend(p2)
        # Usar los mismos criterios de refinamiento definidos en `case`
        # para comparar Cantera vs solver propio en condiciones equivalentes.
        opts = SolveOptions(
            verbose=True,
            refine_ratio=case_ours.ratio,
            refine_slope=case_ours.slope,
            refine_curve=case_ours.curve,
            refine_prune=case_ours.prune,
        )
        t0 = time.perf_counter()
        x_sol, ok, rpt = solve_free_flame(p2, options=opts)
        t_total = time.perf_counter() - t0
        u_s, T_s, _ = unpack_state(x_sol, p2.n_points, p2.n_species)
        print(f"\n  converged    = {ok}")
        print(f"  n_points     = {p2.n_points}")
        print(f"  width [m]    = {p2.width:.6f}")
        print(f"  Su (~u[0])   = {u_s[0]:.6f} m/s")
        print(f"  ||F||inf       = {rpt['Finf_final']:.6e}")
        print(f"  tiempo [s]   = {t_total:.1f}")
        print("\n=== Comparaci�n Cantera vs Nuestro ===")
        print(f"  dSu [m/s]    = {u_s[0] - ct_su:+.6e}")
        print(f"  dn_points    = {p2.n_points - ct_n_points:+d}")
        print(f"  d||F||inf      = {rpt['Finf_final'] - ct_Finf:+.6e}")
        print(f"  dtiempo [s]  = {t_total - t_ct:+.2f}")


if __name__ == "__main__":
    main()





