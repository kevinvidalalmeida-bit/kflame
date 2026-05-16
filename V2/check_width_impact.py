# -*- coding: utf-8 -*-
"""
check_width_impact.py - Diagnóstico: ¿es el ancho del dominio la causa
principal de la diferencia con Cantera?

Ejecuta nuestro solver con width=0.12 (el dominio final de Cantera)
y compara la velocidad de llama y los perfiles.
"""
import time
import numpy as np
import cantera as ct

from config import FlameCase
from problem import FreeFlameProblem
from species_backend import SpeciesBackend
from solver_cantera_auto import SolveOptions, solve_free_flame
from state import unpack_state


def main():
    # 1. Resolver con Cantera (referencia)
    print("=" * 60)
    print("  CHECK: Impacto del ancho del dominio")
    print("=" * 60)

    case_ct = FlameCase(
        mech="gri30.yaml", fuel="CH4",
        oxidizer="O2:1.0, N2:3.76", phi=1.0,
        T_in=300.0, P=101325.0, width=0.03,
        transport_model="mixture-averaged",
        flux_gradient_basis="molar",
        ratio=10.0, slope=0.8, curve=0.8, prune=-0.1,
    )

    print("\n1) Cantera referencia (width=0.03, auto=True)...")
    gas = ct.Solution(case_ct.mech)
    gas.TP = case_ct.T_in, case_ct.P
    gas.set_equivalence_ratio(case_ct.phi, case_ct.fuel, case_ct.oxidizer)
    flame = ct.FreeFlame(gas, width=case_ct.width)
    flame.transport_model = case_ct.transport_model
    flame.set_refine_criteria(ratio=10.0, slope=0.8, curve=0.8, prune=-0.1)
    flame.solve(loglevel=0, auto=True)

    Su_ct = float(flame.velocity[0])
    width_ct = float(flame.grid[-1] - flame.grid[0])
    n_pts_ct = int(flame.grid.size)

    print(f"   Su = {Su_ct:.6f} m/s")
    print(f"   Width final = {width_ct:.4f} m")
    print(f"   n_points = {n_pts_ct}")

    # 2. Nuestro solver con width=0.03 (original)
    print("\n2) Nuestro solver con width=0.03...")
    case_narrow = FlameCase(
        mech="gri30.yaml", fuel="CH4",
        oxidizer="O2:1.0, N2:3.76", phi=1.0,
        T_in=300.0, P=101325.0, width=0.03,
        transport_model="mixture-averaged",
        flux_gradient_basis="molar",
        ratio=10.0, slope=0.8, curve=0.8, prune=-0.1,
    )
    p_narrow = FreeFlameProblem(case_narrow, n_points=5)
    p_narrow.backend = SpeciesBackend(p_narrow)
    opts = SolveOptions(verbose=False, refine_ratio=10.0, refine_slope=0.8,
                        refine_curve=0.8, refine_prune=-0.1,
                        linear_solver="jfnk", jfnk_fallback_to_lu=True)
    t0 = time.perf_counter()
    x_narrow, ok_narrow, _ = solve_free_flame(p_narrow, options=opts)
    t_narrow = time.perf_counter() - t0
    u_n, T_n, Y_n = unpack_state(x_narrow, p_narrow.n_points, p_narrow.n_species)

    print(f"   Converged = {ok_narrow}")
    print(f"   Su = {u_n[0]:.6f} m/s")
    print(f"   Width = {p_narrow.width:.4f} m")
    print(f"   n_points = {p_narrow.n_points}")
    print(f"   Tiempo = {t_narrow:.1f} s")

    # 3. Nuestro solver con width=0.12 (el dominio final de Cantera)
    print(f"\n3) Nuestro solver con width={width_ct:.4f} (dominio de Cantera)...")
    case_wide = FlameCase(
        mech="gri30.yaml", fuel="CH4",
        oxidizer="O2:1.0, N2:3.76", phi=1.0,
        T_in=300.0, P=101325.0, width=width_ct,
        transport_model="mixture-averaged",
        flux_gradient_basis="molar",
        ratio=10.0, slope=0.8, curve=0.8, prune=-0.1,
    )
    p_wide = FreeFlameProblem(case_wide, n_points=8)
    p_wide.backend = SpeciesBackend(p_wide)
    opts_w = SolveOptions(verbose=False, refine_ratio=10.0, refine_slope=0.8,
                          refine_curve=0.8, refine_prune=-0.1,
                          linear_solver="jfnk", jfnk_fallback_to_lu=True,
                          max_total_time_s=120.0)
    t0 = time.perf_counter()
    x_wide, ok_wide, _ = solve_free_flame(p_wide, options=opts_w)
    t_wide = time.perf_counter() - t0
    u_w, T_w, Y_w = unpack_state(x_wide, p_wide.n_points, p_wide.n_species)

    print(f"   Converged = {ok_wide}")
    print(f"   Su = {u_w[0]:.6f} m/s")
    print(f"   Width = {p_wide.width:.4f} m")
    print(f"   n_points = {p_wide.n_points}")
    print(f"   Tiempo = {t_wide:.1f} s")

    # 4. Resumen comparativo
    print("\n" + "=" * 60)
    print("  RESUMEN COMPARATIVO")
    print("=" * 60)
    print(f"  {'Método':<25} {'Su [m/s]':>12} {'dSu [m/s]':>12} {'Width [m]':>10} {'n_pts':>6}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10} {'-'*6}")
    print(f"  {'Cantera (ref)':<25} {Su_ct:>12.6f} {'---':>12} {width_ct:>10.4f} {n_pts_ct:>6}")
    if ok_narrow:
        dSu_n = u_n[0] - Su_ct
        print(f"  {'Nuestro (w=0.03)':<25} {u_n[0]:>12.6f} {dSu_n:>+12.6f} {p_narrow.width:>10.4f} {p_narrow.n_points:>6}")
    else:
        print(f"  {'Nuestro (w=0.03)':<25} {'NO CONV':>12}")
    if ok_wide:
        dSu_w = u_w[0] - Su_ct
        print(f"  {'Nuestro (w=0.12)':<25} {u_w[0]:>12.6f} {dSu_w:>+12.6f} {p_wide.width:>10.4f} {p_wide.n_points:>6}")
    else:
        print(f"  {'Nuestro (w=0.12)':<25} {'NO CONV':>12}")

    if ok_narrow and ok_wide:
        print(f"\n  Mejora por expandir dominio:")
        print(f"    Error Su con w=0.03: {abs(u_n[0] - Su_ct)/Su_ct*100:.3f}%")
        print(f"    Error Su con w=0.12: {abs(u_w[0] - Su_ct)/Su_ct*100:.3f}%")


if __name__ == "__main__":
    main()
