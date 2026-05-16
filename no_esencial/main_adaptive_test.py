"""
main_adaptive_test.py – Solver de especies con MALLA ADAPTATIVA.

Versión ajustada para ser consistente con:
  * bc.py corregido
  * residual_species.py corregido
  * main_species_frozen_test.py corregido

Estrategia:
-----------
  1. Cargar la referencia de Cantera
  2. Resolver en una malla gruesa inicial (subsample de la referencia)
  3. Refinar la malla según gradiente/curvatura
  4. Interpolar la solución al nuevo z
  5. Reconstruir perfiles frozen sobre la nueva malla
  6. Resolver de nuevo
  7. Repetir hasta que la malla no cambie

IMPORTANTE:
-----------
Este script sigue resolviendo un subproblema "frozen":
  - T(z) fijo
  - mdot fijo
  - comparación contra referencia de Cantera

No es todavía un free-flame solver completamente independiente.
"""

from __future__ import annotations

import time
import numpy as np

from config import FlameCase
from problem import FreeFlameProblem
from state import pack_species, unpack_species
from species_backend import SpeciesBackend
from mesh import AdaptiveRefiner
from residual_species import (
    residual_species_frozen,
    residual_species_frozen_transient,
)
from newton import hybrid_steady_transient_newton


REF_PATH = "outputs/reference/freeflame_ref.npz"


# ---------------------------------------------------------------------------
#  Utilidades
# ---------------------------------------------------------------------------
def _clip_and_renormalize_independent(
    Y_ind: np.ndarray,
    max_sum: float = 0.99,
) -> np.ndarray:
    """
    Recorta Y_ind a valores no negativos y garantiza que la suma de especies
    independientes no exceda max_sum en cada nodo.
    """
    Y_ind = np.asarray(Y_ind, dtype=float).copy()
    Y_ind = np.clip(Y_ind, 0.0, None)

    for j in range(Y_ind.shape[1]):
        s = float(np.sum(Y_ind[:, j]))
        if s > max_sum and s > 0.0:
            Y_ind[:, j] *= max_sum / s

    return Y_ind


def _residual_report(F: np.ndarray, problem) -> dict:
    """
    Separa el residual por bloques:
      - left
      - right
      - interior
    """
    n_ind = int(problem.n_ind_species)
    n_pts = int(problem.n_points)

    Fm = np.asarray(F, dtype=float).reshape((n_ind, n_pts), order="C")

    return {
        "left_inf": float(np.max(np.abs(Fm[:, 0]))),
        "right_inf": float(np.max(np.abs(Fm[:, -1]))),
        "interior_inf": float(np.max(np.abs(Fm[:, 1:-1]))) if n_pts > 2 else 0.0,
    }


def _species_error_report(Y_sol: np.ndarray, Y_ref: np.ndarray, species_names: list[str]) -> dict:
    """
    Error máximo absoluto por especie para algunas especies relevantes.
    """
    out = {}
    for nm in ["CH4", "O2", "H2O", "CO2", "CO", "H2", "OH", "N2"]:
        if nm in species_names:
            k = species_names.index(nm)
            out[nm] = float(np.max(np.abs(Y_sol[k, :] - Y_ref[k, :])))
    return out


def solve_on_grid(problem, xY0, tol=5e-5, verbose=True):
    """
    Resuelve el problema frozen de especies en la malla actual.
    """
    xY_sol, ok, hist = hybrid_steady_transient_newton(
        residual_species_frozen,
        residual_species_frozen_transient,
        xY0,
        problem,
        steady_max_iter=1000,
        transient_max_iter=8,
        max_cycles=15,
        tol=tol,
        jac_eps=1e-8,
        alpha_min=1e-8,
        time_step=1e-4,
        time_step_sequence=(2, 5, 10, 20),
        time_step_grow=2.0,
        time_step_shrink=0.5,
        use_sparse=True,
        verbose=verbose,
    )
    return xY_sol, ok, hist


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    case = FlameCase(
        mech="gri30.yaml",
        fuel="CH4",
        oxidizer="O2:1.0, N2:3.76",
        phi=1.0,
        T_in=300.0,
        P=101325.0,
        width=0.03,
    )

    # Cargar referencia completa una sola vez
    ref_data = np.load(REF_PATH, allow_pickle=True)
    z_ref_full = np.asarray(ref_data["z"], dtype=float)
    T_ref_full = np.asarray(ref_data["T"], dtype=float)
    Y_ref_full = np.asarray(ref_data["Y"], dtype=float)
    u_ref_full = np.asarray(ref_data["u"], dtype=float)
    mdot_ref = float(ref_data["mdot_in"])

    t_total = time.perf_counter()

    # ============================================================
    #  NIVEL 0: Malla gruesa
    # ============================================================
    print("=" * 72)
    print("  NIVEL 0: Malla gruesa (subsample=4)")
    print("=" * 72)

    problem = FreeFlameProblem(case, n_points=8)
    problem.use_reference_grid(REF_PATH, subsample=4)
    problem.backend = SpeciesBackend(problem)

    # Asegurar coherencia explícita del frozen problem
    problem.mdot_fixed = mdot_ref

    Y_ref = problem.Y_ref.copy()

    rng = np.random.default_rng(42)
    pert = 1.0 + 0.02 * (rng.random(Y_ref[:-1, :].shape) - 0.5)
    Y0_ind = Y_ref[:-1, :] * pert
    Y0_ind = _clip_and_renormalize_independent(Y0_ind, max_sum=0.99)
    xY = pack_species(Y0_ind)

    # Diagnóstico inicial
    xY_ref = pack_species(Y_ref[:-1, :])
    F_ref = residual_species_frozen(xY_ref, problem)
    rep_ref = _residual_report(F_ref, problem)

    print(f"n_points        = {problem.n_points}")
    print(f"n_species       = {problem.n_species}")
    print(f"n_ind_species   = {problem.n_ind_species}")
    print(f"n_unknowns      = {problem.n_ind_species * problem.n_points}")
    print(f"mdot_fixed      = {problem.mdot_fixed:.8f}")
    print(f"T_fix min/max   = {problem.T_fixed.min():.3f} / {problem.T_fixed.max():.3f}")

    print("\n=== Residual en referencia sobre malla inicial ===")
    print(f"||F(Y_ref)||inf = {np.linalg.norm(F_ref, np.inf):.6e}")
    print(f"left_inf        = {rep_ref['left_inf']:.6e}")
    print(f"right_inf       = {rep_ref['right_inf']:.6e}")
    print(f"interior_inf    = {rep_ref['interior_inf']:.6e}")

    # Resolver nivel 0
    t0 = time.perf_counter()
    xY, ok, hist = solve_on_grid(problem, xY, verbose=True)
    t1 = time.perf_counter()

    Y_sol = unpack_species(xY, problem.n_points, problem.n_species)
    F = residual_species_frozen(xY, problem)
    rep = _residual_report(F, problem)
    err = np.max(np.abs(Y_sol - Y_ref)) if problem.Y_ref is not None else float("nan")

    print(f"\nNivel 0: {problem.n_points} pts")
    print(f"  converged     = {ok}")
    print(f"  steps         = {len(hist)}")
    print(f"  ||F||inf      = {np.linalg.norm(F, np.inf):.6e}")
    print(f"  left_inf      = {rep['left_inf']:.6e}")
    print(f"  right_inf     = {rep['right_inf']:.6e}")
    print(f"  interior_inf  = {rep['interior_inf']:.6e}")
    print(f"  max|Y-Yref|   = {err:.6e}")
    print(f"  t             = {t1 - t0:.2f} s")

    # ============================================================
    #  REFINAMIENTO ADAPTATIVO
    # ============================================================
    refiner = AdaptiveRefiner(
        ratio=3.0,
        slope=0.1,
        curve=0.2,
        prune=0.02,
        grid_min=1e-8,
        max_points=300,
    )

    level = 1
    while True:
        print(f"\n{'=' * 72}")
        print(f"  REFINAMIENTO -> NIVEL {level}")
        print(f"{'=' * 72}")

        # Solución actual reconstruida completa
        Y_full = unpack_species(xY, problem.n_points, problem.n_species)

        # Perfiles usados por el refiner
        profiles = {"T": np.asarray(problem.T_fixed, dtype=float)}
        for name in ["CH4", "O2", "H2O", "CO2", "CO", "H2", "OH"]:
            if name in problem.species_names:
                k = problem.species_names.index(name)
                profiles[name] = Y_full[k, :]

        z_old = problem.z.copy()

        z_new, changed, n_ins, n_rem = refiner.refine(
            z_old,
            profiles,
            all_Y=Y_full,
        )

        if not changed:
            print("Sin cambios en la malla. Refinamiento convergido.")
            break

        print(f"{len(z_old)} pts -> {len(z_new)} pts (+{n_ins}, -{n_rem})")

        # --------------------------------------------------------
        # Interpolación de la solución actual
        # --------------------------------------------------------
        Y_new, T_new_dummy, u_new_dummy = refiner.interpolate_solution(
            z_old,
            z_new,
            Y_full,
            problem.T_fixed,
            problem.u_fixed,
        )

        Y_ind_new = _clip_and_renormalize_independent(Y_new[:-1, :], max_sum=0.99)
        xY = pack_species(Y_ind_new)

        # --------------------------------------------------------
        # Actualizar el frozen problem en la nueva malla
        # --------------------------------------------------------
        problem.z = np.asarray(z_new, dtype=float)
        problem.n_points = int(len(z_new))
        problem.width = float(problem.z[-1] - problem.z[0])

        # Interpolar perfiles frozen desde la referencia completa
        problem.T_fixed = np.interp(problem.z, z_ref_full, T_ref_full)
        problem.u_fixed = np.interp(problem.z, z_ref_full, u_ref_full)
        problem.mdot_fixed = mdot_ref

        problem.Y_ref = np.empty((problem.n_species, problem.n_points), dtype=float)
        for k in range(problem.n_species):
            problem.Y_ref[k, :] = np.interp(problem.z, z_ref_full, Y_ref_full[k, :])

        problem.backend = SpeciesBackend(problem)

        # --------------------------------------------------------
        # Resolver en la nueva malla
        # --------------------------------------------------------
        xY_ref = pack_species(problem.Y_ref[:-1, :])
        F_ref = residual_species_frozen(xY_ref, problem)
        rep_ref = _residual_report(F_ref, problem)

        print("Residual de referencia en nueva malla:")
        print(f"  ||F(Y_ref)||inf = {np.linalg.norm(F_ref, np.inf):.6e}")
        print(f"  left_inf        = {rep_ref['left_inf']:.6e}")
        print(f"  right_inf       = {rep_ref['right_inf']:.6e}")
        print(f"  interior_inf    = {rep_ref['interior_inf']:.6e}")

        t0 = time.perf_counter()
        xY, ok, hist = solve_on_grid(problem, xY, verbose=True)
        t1 = time.perf_counter()

        Y_sol = unpack_species(xY, problem.n_points, problem.n_species)
        F = residual_species_frozen(xY, problem)
        rep = _residual_report(F, problem)
        err = np.max(np.abs(Y_sol - problem.Y_ref))

        print(f"\nNivel {level}: {problem.n_points} pts")
        print(f"  converged     = {ok}")
        print(f"  steps         = {len(hist)}")
        print(f"  ||F||inf      = {np.linalg.norm(F, np.inf):.6e}")
        print(f"  left_inf      = {rep['left_inf']:.6e}")
        print(f"  right_inf     = {rep['right_inf']:.6e}")
        print(f"  interior_inf  = {rep['interior_inf']:.6e}")
        print(f"  max|Y-Yref|   = {err:.6e}")
        print(f"  t             = {t1 - t0:.2f} s")

        level += 1

    # ============================================================
    #  RESULTADO FINAL
    # ============================================================
    elapsed = time.perf_counter() - t_total
    Y_sol = unpack_species(xY, problem.n_points, problem.n_species)
    F = residual_species_frozen(xY, problem)
    rep = _residual_report(F, problem)

    print(f"\n{'=' * 72}")
    print(f"  RESULTADO FINAL (t_total={elapsed:.2f} s)")
    print(f"{'=' * 72}")
    print(f"n_points        = {problem.n_points}")
    print(f"||F||inf        = {np.linalg.norm(F, np.inf):.6e}")
    print(f"left_inf        = {rep['left_inf']:.6e}")
    print(f"right_inf       = {rep['right_inf']:.6e}")
    print(f"interior_inf    = {rep['interior_inf']:.6e}")
    print(f"sum(Y)[0]       = {Y_sol[:, 0].sum():.12f}")
    print(f"sum(Y)[mid]     = {Y_sol[:, problem.n_points // 2].sum():.12f}")
    print(f"sum(Y)[-1]      = {Y_sol[:, -1].sum():.12f}")
    print(f"min(Y)          = {Y_sol.min():.6e}")
    print(f"max(Y)          = {Y_sol.max():.6e}")
    print(f"max|Y-Yref|     = {np.max(np.abs(Y_sol - problem.Y_ref)):.6e}")

    by_species = _species_error_report(Y_sol, problem.Y_ref, problem.species_names)
    for nm, val in by_species.items():
        print(f"  {nm:5s}: {val:.4e}")

    # Guardado opcional del resultado final
    out_path = "outputs/species_frozen_adaptive.npz"
    np.savez(
        out_path,
        z=problem.z,
        T_fixed=problem.T_fixed,
        u_fixed=problem.u_fixed,
        mdot_fixed=problem.mdot_fixed,
        Y=Y_sol,
        Y_ref=problem.Y_ref,
        species_names=np.array(problem.species_names, dtype=object),
    )
    print(f"\nGuardado en: {out_path}")

    print("\nNota:")
    print("Este resultado sigue siendo un problema frozen de especies con")
    print("T(z) y mdot fijados desde la referencia de Cantera.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())