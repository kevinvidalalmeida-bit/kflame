# -*- coding: utf-8 -*-
"""
check_refinement_impact.py - Diagnóstico: verificar si el solver con más
refinamiento converge mejor a Cantera.
"""
import time
import numpy as np
import cantera as ct

from config import FlameCase
from problem import FreeFlameProblem
from species_backend import SpeciesBackend
from solver_cantera_auto import SolveOptions, solve_free_flame
from state import unpack_state
from residual import residual


def main():
    print("=" * 60)
    print("  CHECK: Impacto del refinamiento y comparación detallada")
    print("=" * 60)

    # 1. Cantera referencia
    print("\n1) Cantera referencia...")
    gas = ct.Solution("gri30.yaml")
    gas.TP = 300.0, 101325.0
    gas.set_equivalence_ratio(1.0, "CH4", "O2:1.0, N2:3.76")
    flame = ct.FreeFlame(gas, width=0.03)
    flame.transport_model = "mixture-averaged"
    flame.set_refine_criteria(ratio=10.0, slope=0.8, curve=0.8, prune=-0.1)
    flame.solve(loglevel=0, auto=True)

    z_ct = np.asarray(flame.grid)
    u_ct = np.asarray(flame.velocity)
    T_ct = np.asarray(flame.T)
    Su_ct = float(u_ct[0])
    width_ct = float(z_ct[-1] - z_ct[0])
    print(f"   Su = {Su_ct:.6f} m/s, width = {width_ct:.4f} m, n = {z_ct.size}")

    # 2. Nuestro solver con width=0.03 y refinamiento más estricto
    print("\n2) Nuestro solver con width=0.03, refinamiento estricto...")
    case = FlameCase(
        mech="gri30.yaml", fuel="CH4",
        oxidizer="O2:1.0, N2:3.76", phi=1.0,
        T_in=300.0, P=101325.0, width=0.03,
        transport_model="mixture-averaged",
        flux_gradient_basis="molar",
        ratio=3.0, slope=0.6, curve=0.6, prune=0.05,
    )
    p = FreeFlameProblem(case, n_points=8)
    p.backend = SpeciesBackend(p)
    opts = SolveOptions(
        verbose=False,
        refine_ratio=3.0, refine_slope=0.6, refine_curve=0.6, refine_prune=0.05,
        refine_max_points=200,
        linear_solver="jfnk", jfnk_fallback_to_lu=True,
        max_total_time_s=180.0,
    )
    t0 = time.perf_counter()
    x_sol, ok, rpt = solve_free_flame(p, options=opts)
    t_solve = time.perf_counter() - t0
    u_s, T_s, Y_s = unpack_state(x_sol, p.n_points, p.n_species)

    Finf = float(np.linalg.norm(residual(x_sol, p), ord=np.inf))
    print(f"   Converged = {ok}")
    print(f"   Su = {u_s[0]:.6f} m/s")
    print(f"   n_points = {p.n_points}")
    print(f"   ||F||inf = {Finf:.4e}")
    print(f"   Tiempo = {t_solve:.1f} s")

    # 3. Comparaciones detalladas
    print("\n" + "=" * 60)
    print("  COMPARACIÓN DETALLADA")
    print("=" * 60)

    dSu = u_s[0] - Su_ct
    print(f"\n  dSu = {dSu:+.6f} m/s ({abs(dSu)/Su_ct*100:.3f}%)")

    # Interpolar nuestros perfiles a la malla de Cantera para comparar
    u_ours_ct = np.interp(z_ct, p.z, u_s)
    T_ours_ct = np.interp(z_ct, p.z, T_s)

    # Pero antes normalizar por el ancho del dominio
    # Nuestro dominio es [0, 0.03], Cantera es [0, 0.12]
    # Escalar z para comparar frentes
    z_norm_ct = (z_ct - z_ct[0]) / (z_ct[-1] - z_ct[0])
    z_norm_ours = (p.z - p.z[0]) / (p.z[-1] - p.z[0])

    # Interpolar en coordenada normalizada
    u_ours_norm = np.interp(z_norm_ct, z_norm_ours, u_s)
    T_ours_norm = np.interp(z_norm_ct, z_norm_ours, T_s)

    # Errores
    err_T = np.abs(T_ours_norm - T_ct)
    err_u = np.abs(u_ours_norm - u_ct)

    print(f"\n  En coordenada normalizada (z/width):")
    print(f"    ||T_ours - T_ct||_inf  = {np.max(err_T):.2f} K")
    print(f"    ||T_ours - T_ct||_mean = {np.mean(err_T):.2f} K")
    print(f"    ||u_ours - u_ct||_inf  = {np.max(err_u):.4f} m/s")
    print(f"    ||u_ours - u_ct||_mean = {np.mean(err_u):.4f} m/s")

    # Comparar T_max
    print(f"\n  T_max Cantera  = {T_ct.max():.1f} K")
    print(f"  T_max Nuestro  = {T_s.max():.1f} K")
    print(f"  T_min Cantera  = {T_ct.min():.1f} K")
    print(f"  T_min Nuestro  = {T_s.min():.1f} K")

    # Posición del frente (donde T = 0.5*(T_in + T_max))
    T_mid = 0.5 * (300.0 + T_ct.max())
    j_front_ct = np.argmin(np.abs(T_ct - T_mid))
    j_front_ours = np.argmin(np.abs(T_s - T_mid))
    z_front_ct = z_ct[j_front_ct] / width_ct
    z_front_ours = p.z[j_front_ours] / p.width
    print(f"\n  Posición del frente (z/width donde T ≈ T_mid):")
    print(f"    Cantera  z/w = {z_front_ct:.4f}")
    print(f"    Nuestro  z/w = {z_front_ours:.4f}")


if __name__ == "__main__":
    main()
