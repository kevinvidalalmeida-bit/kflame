"""
main_adaptive_test.py – Solver de especies con MALLA ADAPTATIVA.

Estrategia (igual que Cantera Sim1D):
  1. Resolver en malla gruesa (subsample=4, ~42 puntos)
  2. Refinar la malla segun criterios de gradiente y curvatura
  3. Interpolar la solucion a la nueva malla
  4. Resolver de nuevo
  5. Repetir hasta que la malla no cambie
"""
import time
import numpy as np

from config import FlameCase
from problem import FreeFlameProblem
from state import pack_temperature_species, unpack_temperature_species
from species_backend import SpeciesBackend
from mesh import AdaptiveRefiner
from residual_flame import (
    residual_flame,
    residual_flame_transient,
)
from newton import hybrid_steady_transient_newton
from jacobian import banded_jacobian_flame


REF_PATH = "outputs/reference/freeflame_ref.npz"


def solve_on_grid(problem, x0, tol=5e-5, verbose=True):
    """Resuelve el problema acoplado (T + especies) en la malla actual."""
    x_sol, ok, hist = hybrid_steady_transient_newton(
        residual_flame,
        residual_flame_transient,
        x0, problem,
        steady_max_iter=1000, transient_max_iter=8,
        max_cycles=15, tol=tol, jac_eps=1e-8, alpha_min=1e-8,
        time_step=1e-5, time_step_sequence=(2, 5, 10, 20),
        time_step_grow=2.0, time_step_shrink=0.5,
        use_sparse=True, verbose=verbose,
        jacobian_fn=banded_jacobian_flame
    )
    return x_sol, ok


def main():
    case = FlameCase(
        mech="gri30.yaml", fuel="CH4", oxidizer="O2:1.0, N2:3.76",
        phi=1.0, T_in=300.0, P=101325.0, width=0.03,
    )

    t_total = time.perf_counter()

    # ============================================================
    #  NIVEL 0: Malla gruesa (subsample=4)
    # ============================================================
    print("=" * 65)
    print("  NIVEL 0: Malla gruesa (subsample=4)")
    print("=" * 65)

    problem = FreeFlameProblem(case, n_points=8)
    problem.use_reference_grid(REF_PATH, subsample=4)
    problem.backend = SpeciesBackend(problem)

    Y_ref = problem.Y_ref.copy()
    data = np.load(REF_PATH, allow_pickle=True)
    z_ref = np.asarray(data["z"])
    T_ref = np.asarray(data["T"])
    
    Y_ind_ref = Y_ref[:-1, :]
    # Usar perfil inicial de T de la referencia para converger rápidamente
    T0 = problem.T_fixed.copy()
    
    # Vector de estado x: [T, Y1, ..., Y_{Ns-1}]
    x0 = pack_temperature_species(T0, Y_ind_ref)

    t0 = time.perf_counter()
    x_sol, ok = solve_on_grid(problem, x0, verbose=True)
    t1 = time.perf_counter()

    T_sol, Y_sol = unpack_temperature_species(x_sol, problem.n_points, problem.n_species)
    F = residual_flame(x_sol, problem)
    
    err_Y = np.max(np.abs(Y_sol - Y_ref)) if problem.Y_ref is not None else float("nan")
    err_T = np.max(np.abs(T_sol - T_ref))
    err = max(err_Y, err_T / 2000.0) # Error normalizado para T

    print(f"\n  Nivel 0: {problem.n_points} pts, ||F||={np.linalg.norm(F, np.inf):.2e}, "
          f"err_Y={err_Y:.4e}, err_T={err_T:.1f}K, t={t1-t0:.1f}s, ok={ok}")

    # ============================================================
    #  REFINAMIENTO ADAPTATIVO
    # ============================================================
    refiner = AdaptiveRefiner(
        ratio=3.0,    # max ratio de celdas
        slope=0.1,    # criterio de gradiente (Cantera default: 0.1)
        curve=0.2,    # criterio de curvatura (Cantera default: 0.2)
        prune=0.02,   # umbral para eliminar puntos
        grid_min=1e-8,
        max_points=300,
    )

    level = 1
    while True:
        print(f"\n{'=' * 65}")
        print(f"  REFINAMIENTO -> NIVEL {level}")
        print(f"{'=' * 65}")

        # Construir perfiles para el refiner
        T_sol, Y_full = unpack_temperature_species(x_sol, problem.n_points, problem.n_species)
        profiles = {"T": T_sol}
        for name in ["CH4", "O2", "H2O", "CO2", "CO", "H2", "OH"]:
            if name in problem.species_names:
                k = problem.species_names.index(name)
                profiles[name] = Y_full[k, :]

        z_old = problem.z.copy()
        z_new, changed, n_ins, n_rem = refiner.refine(
            z_old, profiles, all_Y=Y_full
        )

        if not changed:
            print("  Sin cambios en la malla. Convergido!")
            break

        print(f"  {len(z_old)} pts -> {len(z_new)} pts "
              f"(+{n_ins} insertados, -{n_rem} eliminados)")

        # Interpolar solucion y perfiles a la nueva malla
        Y_new, T_new, u_new = refiner.interpolate_solution(
            z_old, z_new,
            Y_full,           # (nsp, n_old)
            problem.T_fixed,  # (n_old,)
            problem.u_fixed,  # (n_old,)
        )

        # Interpolar referencia tambien
        data = np.load(REF_PATH, allow_pickle=True)
        z_ref_full = np.asarray(data["z"])
        Y_ref_full = np.asarray(data["Y"])
        T_ref_full = np.asarray(data["T"])
        u_ref_full = np.asarray(data["u"])

        # Actualizar el problema con la nueva malla
        problem.z = z_new
        problem.n_points = len(z_new)
        problem.T_fixed = np.interp(z_new, z_ref_full, T_ref_full)
        problem.u_fixed = np.interp(z_new, z_ref_full, u_ref_full)
        problem.Y_ref = np.empty((problem.n_species, len(z_new)))
        for k in range(problem.n_species):
            problem.Y_ref[k, :] = np.interp(z_new, z_ref_full, Y_ref_full[k, :])
        problem.backend = SpeciesBackend(problem)

        # 3. Interpolar la solución al nuevo z_new
        n_ind = problem.n_ind_species
        T_new = np.interp(z_new, z_old, T_sol)
        Y_ind_new = np.zeros((n_ind, len(z_new)))
        for i in range(n_ind):
            Y_ind_new[i, :] = np.interp(z_new, z_old, Y_full[i, :])
        
        # Clip y normalización simple
        Y_ind_new = np.clip(Y_ind_new, 1e-15, 1.0)
        for j in range(len(z_new)):
            s = Y_ind_new[:, j].sum()
            if s > 0.99:
                Y_ind_new[:, j] *= 0.99 / s
        x_sol = pack_temperature_species(T_new, Y_ind_new)

        # Resolver en la nueva malla
        t0 = time.perf_counter()
        x_sol, ok = solve_on_grid(problem, x_sol, verbose=True)
        t1 = time.perf_counter()

        T_sol, Y_sol = unpack_temperature_species(x_sol, problem.n_points, problem.n_species)
        F = residual_flame(x_sol, problem)
        
        err_Y = np.max(np.abs(Y_sol - problem.Y_ref))
        err_T = np.max(np.abs(T_sol - problem.T_fixed)) # Nota: T_fixed ahora almacena el T_ref interpolado
        
        print(f"\n  Nivel {level}: {problem.n_points} pts, "
              f"||F||={np.linalg.norm(F, np.inf):.2e}, "
              f"err_Y={err_Y:.4e}, err_T={err_T:.1f}K, t={t1-t0:.1f}s, ok={ok}")

        level += 1

    # ============================================================
    #  RESULTADO FINAL
    # ============================================================
    elapsed = time.perf_counter() - t_total
    T_sol, Y_sol = unpack_temperature_species(x_sol, problem.n_points, problem.n_species)
    err_Y = np.max(np.abs(Y_sol - problem.Y_ref))
    err_T = np.max(np.abs(T_sol - problem.T_fixed))
    
    print("\nRESUMEN FINAL")
    print(f"Malla final  : {problem.n_points} puntos")
    print(f"Error max Y  : {err_Y:.4e}")
    print(f"Error max T  : {err_T:.1f} K")
    print("=" * 65)
    
    # 5. Guardar resultados
    out_path = "outputs/flame_coupled_adaptive.npz"
    np.savez(out_path, z=problem.z, T=T_sol, Y=Y_sol)


if __name__ == "__main__":
    main()
