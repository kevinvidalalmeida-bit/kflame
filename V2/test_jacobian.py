"""
test_jacobian.py – Verificación numérica de los Jacobianos.

Prueba que:
  1. El Jacobiano estacionario sea correcto (dirección aleatoria).
  2. El Jacobiano transitorio sea correcto (igual, con x_old distinto).
  3. La actualización diagonal sea exacta:
       J_t[n,n] == J_ss[n,n] - mask[n] * rdt   para todo n.

Uso:
    python test_jacobian.py
"""
from __future__ import annotations

import numpy as np
import cantera as ct

from config import FlameCase
from problem import FreeFlameProblem
from species_backend import SpeciesBackend
from equations import residual
from equations import (banded_jacobian, build_jacobian_steady,
                      build_jacobian_transient, update_transient)
from state import pack_state, build_transient_mask


def _cantera_reference(case):
    gas = ct.Solution(case.mech)
    gas.TP = case.T_in, case.P
    gas.set_equivalence_ratio(case.phi, case.fuel, case.oxidizer)
    flame = ct.FreeFlame(gas, width=case.width)
    flame.transport_model = case.transport_model
    flame.set_refine_criteria(ratio=case.ratio, slope=case.slope,
                              curve=case.curve, prune=case.prune)
    flame.solve(loglevel=0, auto=True)
    z = np.asarray(flame.grid, dtype=float)
    u = np.asarray(flame.velocity, dtype=float)
    T = np.asarray(flame.T, dtype=float)
    Y = np.asarray(flame.Y, dtype=float)
    return z, u, T, Y


def _build_problem(case, z, u, T, Y):
    problem = FreeFlameProblem(case, n_points=len(z))
    problem.z = z.copy()
    problem.n_points = len(z)
    problem.width = float(z[-1] - z[0])
    problem.backend = SpeciesBackend(problem)
    problem.solve_energy = True
    problem.setup_fixed_temperature(T_profile=T)
    return problem


def steady_fun(x, prob):
    return residual(x, prob, rdt=0.0)


def main():
    case = FlameCase(
        mech="gri30.yaml", fuel="CH4",
        oxidizer="O2:1.0, N2:3.76", phi=1.0,
        T_in=300.0, P=101325.0, width=0.03,
        transport_model="mixture-averaged",
    )

    print("Obteniendo malla de referencia de Cantera...")
    z, u, T, Y = _cantera_reference(case)
    problem = _build_problem(case, z, u, T, Y)
    x = pack_state(u, T, Y)

    rng = np.random.default_rng(42)
    v = rng.normal(size=x.size)
    v /= np.linalg.norm(v)
    eps = 1e-7

    # ----------------------------------------------------------------
    #  1. Jacobiano estacionario
    # ----------------------------------------------------------------
    print("\n[1] Jacobiano estacionario (dirección-v test)...")
    F0 = steady_fun(x, problem)
    J_ss, ss_diag = build_jacobian_steady(steady_fun, x, problem)
    F1 = steady_fun(x + eps * v, problem)
    lhs = (F1 - F0) / eps
    rhs = J_ss @ v
    err_abs = float(np.linalg.norm(lhs - rhs, ord=np.inf))
    err_rel = err_abs / max(float(np.linalg.norm(lhs, ord=np.inf)), 1e-30)
    print(f"  ||lhs-rhs||∞ = {err_abs:.4e}   err_rel = {err_rel:.4e}")
    assert err_rel < 0.01, f"Jacobiano estacionario incorrecto: err_rel={err_rel:.4e}"
    print("  ✓ OK")

    # ----------------------------------------------------------------
    #  2. Jacobiano transitorio (actualización diagonal)
    # ----------------------------------------------------------------
    print("\n[2] Verificando que J_t[n,n] = J_ss[n,n] - mask[n]*rdt ...")
    rdt = 1e6
    mask = build_transient_mask(problem.n_points, problem.n_species,
                                solve_energy=True)
    J_t = update_transient(J_ss, mask, rdt)
    diag_ss = J_ss.diagonal()
    diag_t  = J_t.diagonal()
    expected_diag = diag_ss - mask * rdt
    err_diag = float(np.max(np.abs(diag_t - expected_diag)))
    print(f"  max|diag_t - expected|  = {err_diag:.4e}")
    assert err_diag < 1e-6 * max(float(np.max(np.abs(expected_diag))), 1.0), \
        "Actualización diagonal incorrecta"
    n_diff = int(np.sum(mask))
    print(f"  Variables diferenciales  = {n_diff} / {mask.size}")
    print("  ✓ OK")

    # ----------------------------------------------------------------
    #  3. Jacobiano transitorio (dirección-v test)
    # ----------------------------------------------------------------
    print("\n[3] Jacobiano transitorio (dirección-v test)...")
    x_old = x + 1e-6 * rng.normal(size=x.size)

    def ts_fun(xn, prob):
        return residual(xn, prob, rdt=rdt, x_old=x_old)

    F0_ts = ts_fun(x, problem)
    J_t_full = build_jacobian_transient(steady_fun, x, problem, rdt)
    F1_ts = ts_fun(x + eps * v, problem)
    lhs_ts = (F1_ts - F0_ts) / eps
    rhs_ts = J_t_full @ v
    err_ts = float(np.linalg.norm(lhs_ts - rhs_ts, ord=np.inf))
    err_ts_rel = err_ts / max(float(np.linalg.norm(lhs_ts, ord=np.inf)), 1e-30)
    print(f"  ||lhs-rhs||∞ = {err_ts:.4e}   err_rel = {err_ts_rel:.4e}")
    assert err_ts_rel < 0.01, f"Jacobiano transitorio incorrecto: err_rel={err_ts_rel:.4e}"
    print("  ✓ OK")

    print("\n✓ Todos los tests pasaron.")


if __name__ == "__main__":
    main()
