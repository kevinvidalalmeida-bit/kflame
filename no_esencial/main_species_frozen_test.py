"""
main_species_frozen_test.py – Test del solver de especies (frozen).

Versión ajustada para ser consistente con:
  * bc.py corregido
  * residual_species.py corregido

Qué prueba este script:
-----------------------
1) Carga una solución de referencia de Cantera
2) Usa su malla adaptada (subsampleada)
3) Congela T(z) y mdot
4) Resuelve solo las especies independientes
5) Reporta:
   - residual total
   - residual en frontera izquierda / derecha / interior
   - conservación de suma(Y)
   - error respecto a la referencia

IMPORTANTE:
-----------
Como las BC ya no son:
    Y(0) = Y_in
    dY/dz = 0
sino balances convectivo-difusivos, el valor de ||F(Y_ref)|| ya no debe
interpretarse como "error de discretización puro" de una forma ingenua,
sino como "mismatch residual del problema frozen discretizado usando la
referencia de Cantera como campo de prueba".
"""

from __future__ import annotations

import time
import numpy as np

from config import FlameCase
from problem import FreeFlameProblem
from state import pack_species, unpack_species
from species_backend import SpeciesBackend
from residual_species import (
    residual_species_frozen,
    residual_species_frozen_transient,
)
from newton import hybrid_steady_transient_newton


REF_PATH = "outputs/reference/freeflame_ref.npz"


# ---------------------------------------------------------------------------
#  Utilidades
# ---------------------------------------------------------------------------
def _clip_and_renormalize_independent(Y_ind: np.ndarray, max_sum: float = 0.99) -> np.ndarray:
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
      - left  : nodo j=0
      - right : nodo j=N-1
      - interior : nodos j=1..N-2
    """
    n_ind = int(problem.n_ind_species)
    n_pts = int(problem.n_points)

    Fm = np.asarray(F, dtype=float).reshape((n_ind, n_pts), order="C")

    rep = {
        "left_inf": float(np.max(np.abs(Fm[:, 0]))),
        "right_inf": float(np.max(np.abs(Fm[:, -1]))),
        "interior_inf": float(np.max(np.abs(Fm[:, 1:-1]))) if n_pts > 2 else 0.0,
    }
    return rep


def _species_error_report(Y_sol: np.ndarray, Y_ref: np.ndarray, species_names: list[str]) -> dict:
    """
    Error máximo absoluto por especie para algunas especies de interés.
    """
    out = {}
    for nm in ["CH4", "O2", "H2O", "CO2", "CO", "H2", "OH", "N2"]:
        if nm in species_names:
            k = species_names.index(nm)
            out[nm] = float(np.max(np.abs(Y_sol[k, :] - Y_ref[k, :])))
    return out


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

    problem = FreeFlameProblem(case, n_points=8)

    # Usar malla de referencia adaptada de Cantera y perfiles frozen
    # subsample=2 => ~80 puntos, aún razonable con Jacobiano sparse
    problem.use_reference_grid(REF_PATH, subsample=2)
    problem.backend = SpeciesBackend(problem)

    print("=" * 68)
    print("  SOLVER ESPECIES FROZEN (BC convectivo-difusivas + Jacobiano sparse)")
    print("=" * 68)
    print(f"n_points        = {problem.n_points}")
    print(f"n_species       = {problem.n_species}")
    print(f"n_ind_species   = {problem.n_ind_species}")
    print(f"n_unknowns      = {problem.n_ind_species * problem.n_points}")
    print(f"mdot_fixed      = {problem.mdot_fixed:.8f}")
    print(f"T_fix min/max   = {problem.T_fixed.min():.3f} / {problem.T_fixed.max():.3f}")

    # ------------------------------------------------------------
    # Referencia
    # ------------------------------------------------------------
    Y_ref = problem.Y_ref.copy()
    xY_ref = pack_species(Y_ref[:-1, :])

    F_ref = residual_species_frozen(xY_ref, problem)
    rep_ref = _residual_report(F_ref, problem)

    print("\n=== Residual en la solución de referencia ===")
    print(f"||F(Y_ref)||inf = {np.linalg.norm(F_ref, ord=np.inf):.6e}")
    print(f"left_inf        = {rep_ref['left_inf']:.6e}")
    print(f"right_inf       = {rep_ref['right_inf']:.6e}")
    print(f"interior_inf    = {rep_ref['interior_inf']:.6e}")

    # ------------------------------------------------------------
    # Initial guess perturbado
    # ------------------------------------------------------------
    rng = np.random.default_rng(42)
    pert = 1.0 + 0.02 * (rng.random(Y_ref[:-1, :].shape) - 0.5)
    Y0_ind = Y_ref[:-1, :] * pert
    Y0_ind = _clip_and_renormalize_independent(Y0_ind, max_sum=0.99)

    xY0 = pack_species(Y0_ind)
    F0 = residual_species_frozen(xY0, problem)
    rep0 = _residual_report(F0, problem)

    print("\n=== Initial guess perturbado ===")
    print(f"||F(Y0)||inf    = {np.linalg.norm(F0, ord=np.inf):.6e}")
    print(f"left_inf        = {rep0['left_inf']:.6e}")
    print(f"right_inf       = {rep0['right_inf']:.6e}")
    print(f"interior_inf    = {rep0['interior_inf']:.6e}")

    # ------------------------------------------------------------
    # Resolución
    # ------------------------------------------------------------
    print("\n--- Resolviendo ---\n")
    t0 = time.perf_counter()

    xY_sol, ok, hist = hybrid_steady_transient_newton(
        residual_species_frozen,
        residual_species_frozen_transient,
        xY0,
        problem,
        steady_max_iter=15,
        transient_max_iter=8,
        max_cycles=15,
        tol=5e-5,
        jac_eps=1e-8,
        alpha_min=1e-8,
        time_step=1e-4,
        time_step_sequence=(2, 5, 10, 20),
        time_step_grow=2.0,
        time_step_shrink=0.5,
        use_sparse=True,
        verbose=True,
    )

    elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------
    # Postproceso
    # ------------------------------------------------------------
    F_sol = residual_species_frozen(xY_sol, problem)
    rep_sol = _residual_report(F_sol, problem)
    Y_sol = unpack_species(xY_sol, problem.n_points, problem.n_species)

    print("\n" + "=" * 68)
    print(f"  RESULTADO FINAL (t = {elapsed:.2f} s)")
    print("=" * 68)
    print(f"converged       = {ok}")
    print(f"newton_steps    = {len(hist)}")
    print(f"||F||inf        = {np.linalg.norm(F_sol, ord=np.inf):.6e}")
    print(f"left_inf        = {rep_sol['left_inf']:.6e}")
    print(f"right_inf       = {rep_sol['right_inf']:.6e}")
    print(f"interior_inf    = {rep_sol['interior_inf']:.6e}")

    print("\n=== Chequeo de normalización ===")
    print(f"sum(Y)[0]       = {Y_sol[:, 0].sum():.12f}")
    print(f"sum(Y)[mid]     = {Y_sol[:, problem.n_points // 2].sum():.12f}")
    print(f"sum(Y)[-1]      = {Y_sol[:, -1].sum():.12f}")
    print(f"min(Y)          = {Y_sol.min():.6e}")
    print(f"max(Y)          = {Y_sol.max():.6e}")

    if problem.Y_ref is not None:
        err_full = np.max(np.abs(Y_sol - problem.Y_ref))
        err_ind = np.max(np.abs(Y_sol[:-1, :] - problem.Y_ref[:-1, :]))

        print("\n=== Error respecto a referencia Cantera ===")
        print(f"max|Y - Y_ref|          = {err_full:.6e}")
        print(f"max|Y_ind - Y_ref_ind|  = {err_ind:.6e}")

        by_species = _species_error_report(Y_sol, problem.Y_ref, problem.species_names)
        for nm, val in by_species.items():
            print(f"  {nm:5s}: {val:.4e}")

    # ------------------------------------------------------------
    # Información interpretativa
    # ------------------------------------------------------------
    print("\n=== Nota ===")
    print("Este test resuelve un problema 'frozen' con T(z) y mdot fijados")
    print("desde una referencia de Cantera. Por tanto, no es una llama libre")
    print("independiente completa, sino un subproblema de especies.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())