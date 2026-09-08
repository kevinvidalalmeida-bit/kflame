"""Local equivalence and homotopy endpoint checks before full timing."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
from pathlib import Path
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'V2'),
               str(Path(__file__).resolve().parents[1]/'V2/materiales')]
import json
import numpy as np
import cantera as ct
from config import FlameCase
from problem import FreeFlameProblem
from species_backend_native import NativeSpeciesBackend
from run_saved_comparison import _resolve_mechanism
import equations
import solver
from cold_strategy_kernels import build_candidates
from cold_dependencies import precompute
from cold_transport_candidates import build_action, action_flux, blend_context


def main():
    result = []
    kernels = build_candidates()
    action = build_action()
    shared_action = build_action(shared=True)
    for mechanism in ('h2o2.yaml', 'gri30.yaml'):
        for pressure in (1, 10):
            case = FlameCase(mech=_resolve_mechanism(mechanism), fuel='H2',
                oxidizer='O2:1,N2:3.76', phi=1., T_in=300., P=pressure*ct.one_atm,
                width=.03, transport_model='multicomponent', soret_enabled=True)
            problem = FreeFlameProblem(case, n_points=8)
            problem.assume_finite_y = True
            problem.jacobian_mode = 'block_tridiag'
            problem.precompute_jacobian_thermo = True
            problem.solve_energy = True
            backend = problem.backend = NativeSpeciesBackend(problem)
            gas = ct.Solution(case.mech)
            gas.TP = 300, case.P
            gas.set_equivalence_ratio(1, case.fuel, case.oxidizer)
            fresh = gas.Y.copy()
            gas.equilibrate('HP')
            burned = gas.Y.copy()
            t = np.array([300., 500., 999.9, 1000.1, 1400., 1900., 2200., 2400.])
            y = fresh[:, None] + (burned-fresh)[:, None]*np.linspace(0, 1, t.size)
            y[gas.species_index('H'), 2] = -1e-8
            dz = np.diff(problem.z)
            x = np.column_stack((np.ones(t.size), t, y.T)).ravel()
            cache = equations.build_local_jacobian_cache(x, problem)
            args = (x, problem, cache, 1e-5, 1e-10, 1e-5)
            reference = equations._precompute_block_tridiag_center_thermo(*args)
            trace = []
            candidate = precompute(*args, thermal=kernels['thermal'], records=trace)
            for a, b in zip(reference, candidate):
                np.testing.assert_allclose(a, b, rtol=5e-12, atol=1e-10)
            face_data = backend.eval_multicomponent_face_transport(.5*(t[:-1]+t[1:]), .5*(y[:, :-1]+y[:, 1:]))
            rho, lam, wmix, multi, thermal = face_data
            expected = equations._multicomponent_flux(y[:, :-1], y[:, 1:], t[:-1], t[1:],
                rho, wmix, multi, thermal, dz, backend.W)
            actual_lam, actual = action_flux(action, backend, y[:, :-1], y[:, 1:], t[:-1], t[1:], dz)
            np.testing.assert_allclose(actual_lam, lam, rtol=2e-10, atol=1e-12)
            np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-11)
            shared_lam, shared_flux = action_flux(shared_action, backend, y[:, :-1], y[:, 1:], t[:-1], t[1:], dz)
            np.testing.assert_allclose(shared_lam, lam, rtol=2e-10, atol=1e-12)
            np.testing.assert_allclose(shared_flux, expected, rtol=2e-8, atol=1e-11)
            backend.soret_enabled = False
            without_lam, without_flux = action_flux(action, backend, y[:, :-1], y[:, 1:], t[:-1], t[1:], dz)
            without_expected = equations._multicomponent_flux(y[:, :-1], y[:, 1:], t[:-1], t[1:],
                rho, wmix, multi, np.zeros_like(thermal), dz, backend.W)
            np.testing.assert_allclose(without_flux, without_expected, rtol=2e-8, atol=1e-11)
            np.testing.assert_allclose(without_lam, lam, rtol=2e-10, atol=1e-12)
            backend.soret_enabled = True
            # Unchanged state endpoints: mixture and full residual/J must match
            # the corresponding endpoint of the blended operator.
            mix_case = solver.copy.copy(case)
            mix_case.transport_model, mix_case.soret_enabled = 'mixture-averaged', False
            mix_problem = FreeFlameProblem(mix_case, n_points=8)
            mix_backend = NativeSpeciesBackend(mix_problem)
            problem.lag_multicomponent_transport = False
            expected_full = equations.residual(x, problem)
            with blend_context(problem, mix_backend, 1., []):
                np.testing.assert_array_equal(solver.residual(x, problem), expected_full)
            with blend_context(problem, mix_backend, .5, []) as _:
                jac, diagonal = solver.build_jacobian_steady(lambda q, p: solver.residual(q, p), x, problem)
                assert np.isfinite(jac.diag).all()
                np.testing.assert_array_equal(jac.diagonal(), diagonal)
            result.append(dict(mechanism=mechanism, pressure_atm=pressure,
                dependency_max_abs=max(float(np.max(abs(a-b))) for a, b in zip(reference, candidate)),
                flux_max_abs=float(np.max(abs(actual-expected))),
                shared_flux_max_abs=float(np.max(abs(shared_flux-expected))),
                flux_without_soret_max_abs=float(np.max(abs(without_flux-without_expected))),
                flux_mass_sum_max=float(np.max(abs(actual.sum(axis=0)))),
                conductivity_max_abs=float(np.max(abs(actual_lam-lam)))))
            print(result[-1], flush=True)
    destination = Path(__file__).with_name('cold_followup_local_checks.json')
    destination.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
