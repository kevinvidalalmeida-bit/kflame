"""
checkpoint_comparison.py

Este script ejecuta una llama libre 1D premezclada usando:
  1) Cantera Nativo (C++)
  2) Implementación V2 (Python) - Usando backend nativo para propiedades si se requiere.

Luego compara los resultados (tiempo de ejecución, velocidad de la llama Su,
y error máximo en los perfiles de T y u interpolados a una malla común).
"""

import time
import numpy as np
import cantera as ct

from config import FlameCase
from problem import FreeFlameProblem
from solver import solve_free_flame, SolveOptions
from state import unpack_state

# El backend original o el nativo
try:
    from materiales.species_backend_native import NativeSpeciesBackend as BackendToUse
    backend_name = "Nativo (Vectorizado)"
except ImportError:
    from species_backend import SpeciesBackend as BackendToUse
    backend_name = "Cantera Wrapper"


def run_cantera(case: FlameCase):
    print(f"\n--- Ejecutando Cantera (C++) ---")
    gas = ct.Solution(case.mech)
    gas.TP = case.T_in, case.P
    gas.set_equivalence_ratio(case.phi, case.fuel, case.oxidizer)

    flame = ct.FreeFlame(gas, width=case.width)
    flame.transport_model = case.transport_model
    flame.set_refine_criteria(ratio=case.ratio, slope=case.slope,
                              curve=case.curve, prune=case.prune)

    t0 = time.perf_counter()
    flame.solve(loglevel=0, auto=True)
    t1 = time.perf_counter()

    z = flame.grid
    T = flame.T
    u = flame.velocity
    Y = flame.Y
    Su = u[0]

    print(f"  Tiempo        : {t1 - t0:.3f} s")
    print(f"  Puntos malla  : {len(z)}")
    print(f"  Velocidad Su  : {Su * 100:.2f} cm/s")

    return z, T, u, Y, (t1 - t0), Su


def run_v2(case: FlameCase):
    print(f"\n--- Ejecutando V2 (Python - {backend_name}) ---")

    # Configuramos el problema V2 inicial con malla gruesa
    p = FreeFlameProblem(case, n_points=8)
    p.backend = BackendToUse(p)

    opts = SolveOptions(
        verbose=True,
        refine_ratio=case.ratio,
        refine_slope=case.slope,
        refine_curve=case.curve,
        refine_prune=case.prune,
        max_total_time_s=300.0,
    )

    t0 = time.perf_counter()
    x_sol, ok, rpt = solve_free_flame(p, options=opts)
    t1 = time.perf_counter()

    z = p.z
    u, T, Y = unpack_state(x_sol, p.n_points, p.n_species)
    Su = u[0]

    if not ok:
        print("  ADVERTENCIA: El solver V2 no convergió.")

    print(f"  Tiempo        : {t1 - t0:.3f} s")
    print(f"  Puntos malla  : {len(z)}")
    print(f"  Velocidad Su  : {Su * 100:.2f} cm/s")

    return z, T, u, Y, (t1 - t0), Su


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
        ratio=5.0, slope=0.2, curve=0.2, prune=0.02, # Mismos criterios refinamiento
    )

    print("=" * 60)
    print(" COMPARACIÓN DE PUNTOS DE CONTROL: CANTERA vs V2 PYTHON ")
    print("=" * 60)

    # Ejecutar ambos
    # z_ct, T_ct, u_ct, Y_ct, t_ct, Su_ct = run_cantera(case)
    z_v2, T_v2, u_v2, Y_v2, t_v2, Su_v2 = run_v2(case)

    print("\n" + "=" * 60)
    print(" RESUMEN DE LA COMPARACIÓN")
    print("=" * 60)

    # 1. Velocidad de quemado
    # err_Su = abs(Su_v2 - Su_ct) / abs(Su_ct) * 100
    # print(f" Velocidad Su Cantera : {Su_ct * 100:6.2f} cm/s")
    print(f" Velocidad Su V2      : {Su_v2 * 100:6.2f} cm/s")
    # print(f" Error Relativo Su    : {err_Su:6.3f} %")

    # 2. Tiempos
    # speedup = t_ct / t_v2 if t_v2 > 0 else float('inf')
    # print(f"\n Tiempo Cantera       : {t_ct:6.3f} s")
    print(f" Tiempo V2            : {t_v2:6.3f} s")
    # print(f" Speedup V2           : {speedup:6.2f}x")

    # 3. Diferencia perfiles (interpolando V2 a malla Cantera)
    # T_v2_interp = np.interp(z_ct, z_v2, T_v2)
    # u_v2_interp = np.interp(z_ct, z_v2, u_v2)

    # err_T_max = np.max(np.abs(T_ct - T_v2_interp))
    # err_u_max = np.max(np.abs(u_ct - u_v2_interp))

    # print(f"\n Error Máx Absoluto T : {err_T_max:6.2f} K")
    # print(f" Error Máx Absoluto u : {err_u_max:6.4f} m/s")

    # Guardar resultados opcionalmente
    print("\n(Resultados no guardados en .npz - Modificar script si se requiere)")

if __name__ == "__main__":
    main()
