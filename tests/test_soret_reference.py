from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import builtins
from contextlib import nullcontext

try:
    import cantera as ct
except ImportError:
    ct = None
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

from kflame.flame.equations import (  # noqa: E402
    _multicomponent_flux,
    build_jacobian_steady,
    build_local_jacobian_cache,
    factorize,
    residual,
    residual_local_rows,
    residual_local_rows_batch_perturbed,
    solve_linear,
)
from kflame.flame.problem import FreeFlameProblem  # noqa: E402
if ct is not None:
    from kflame.reference.backend import SpeciesBackend  # noqa: E402
from kflame.chemistry.backend import NativeSpeciesBackend  # noqa: E402
from kflame.chemistry.mechanism import ONE_ATM, resolve_mechanism  # noqa: E402
from kflame.flame.state import pack_state  # noqa: E402
from kflame.flame.solver import SolveOptions, JacobianState, newton_solve, _acceptance_status, _solve_auto_stages, _refine_and_solve, _hybrid_newton  # noqa: E402


def _case(*, soret: bool, transport: str = "multicomponent"):
    return SimpleNamespace(
        mech="gri30.yaml",
        fuel="H2",
        oxidizer="O2:1.0, N2:3.76",
        phi=1.0,
        T_in=300.0,
        P=ONE_ATM,
        width=0.03,
        transport_model=transport,
        flux_gradient_basis="molar",
        soret_enabled=soret,
        cantera_seed_grid=False,
        adaptive_grid=False,
    )


class SoretReferenceTransportTests(unittest.TestCase):
    def test_removed_experimental_options_fail_explicitly(self):
        for name, value in (("transient_scheme", "ptc-auto"),
                            ("newton_fixed_weights", True)):
            with self.subTest(option=name), self.assertRaises(TypeError):
                SolveOptions(**{name: value})
        case = _case(soret=False, transport="mixture-averaged")
        case.upwind_factor = .5
        with self.assertRaisesRegex(ValueError, 'upwind'):
            FreeFlameProblem(case)

    def test_damping_and_convergence_use_current_state_scales(self):
        p = SimpleNamespace(n_points=2, n_species=0, solve_energy=True)
        x = np.array([1., 300., 2., 400.])
        target = x + 1.
        jac = JacobianState(J=np.eye(4), lu=True, ss_diag=np.ones(4), last_rdt=0., age=0)
        with patch("kflame.flame.equations.residual", side_effect=lambda state, *a, **kw: state-target), \
             patch("kflame.flame.solver.solve_linear", side_effect=lambda lu, rhs: rhs), \
             patch("kflame.flame.solver.bound_step_limit", return_value=(1., "")), \
             patch("kflame.flame.solver.weighted_norm", side_effect=[10., 2.]) as norms:
            _, ok, history, _ = newton_solve(None, x, p, max_iter=1, jac_state=jac)
        self.assertFalse(ok)  # contraction is not convergence when norm > 1
        self.assertEqual(history[-1]["s1"], 2.)
        np.testing.assert_array_equal(norms.call_args_list[0].args[1], x)
        np.testing.assert_array_equal(norms.call_args_list[1].args[1], target)

    def test_ptc_budget_is_bounded_without_extra_experimental_budget(self):
        p = FreeFlameProblem(_case(soret=True), n_points=4)
        p.solve_energy = True
        x = p.make_initial_guess()
        opts = SolveOptions(verbose=False,
                            time_step_sequence=(1,), max_time_step_count=2)
        calls = []
        def newton(fun, state, problem, **kw):
            calls.append(kw)
            if not kw["rdt"]:
                return state, False, [{"status": "no_damp", "normF": 10.}], None
            ptc = kw["residual_damping"]
            return state, not ptc, [{"status": "step" if ptc else "ok", "normF": 10., "s1": .1}], None
        with patch("kflame.flame.solver.newton_solve", side_effect=newton), \
             patch("kflame.flame.solver._residual_inf", return_value=10.):
            _, ok, history = _hybrid_newton(p, x, opts)
        self.assertFalse(ok)
        self.assertEqual([c["max_iter"] for c in calls if c["rdt"]], [1, 1])
        self.assertEqual(history[-1]["reason"], "max_time_step_count")
        self.assertEqual(history[-1]["nsteps_total"], 2)

    def test_rejected_ptc_uses_full_be_without_certifying_a_pseudo_step(self):
        p = FreeFlameProblem(_case(soret=True), n_points=4)
        p.solve_energy = True
        x = p.make_initial_guess()
        opts = SolveOptions(verbose=False,
                            time_step_sequence=(1,), max_time_step_count=1,
                            trace_solver=True)
        calls = []
        def newton(fun, state, problem, **kw):
            calls.append(kw)
            if kw["rdt"] == 0:
                return state, False, [{"status": "no_damp", "normF": 10.}], None
            if kw["residual_damping"]:
                return state, False, [{"status": "no_damp", "normF": 10., "s1": 2.}], None
            return state, True, [{"status": "ok", "normF": 10., "s1": .1}], None
        with patch("kflame.flame.solver.newton_solve", side_effect=newton), \
             patch("kflame.flame.solver._residual_inf", return_value=10.):
            actual, ok, history = _hybrid_newton(p, x, opts)
        self.assertFalse(ok)  # successful pseudo-steps are not a steady certificate
        self.assertEqual([c["max_iter"] for c in calls if c["rdt"] > 0], [1, 50])
        self.assertIs(p._solver_trace[0]["history"], history)
        np.testing.assert_array_equal(actual, x)

    def test_progressing_ptc_does_not_switch_to_be(self):
        p = FreeFlameProblem(_case(soret=True), n_points=4)
        p.solve_energy = True
        x = p.make_initial_guess()
        opts = SolveOptions(verbose=False,
                            time_step_sequence=(1,), max_time_step_count=2)
        calls = []
        def newton(fun, state, problem, **kw):
            calls.append(kw)
            return state, False, [{"status": "step" if kw["rdt"] else "no_damp",
                                   "normF": 10., "s1": 2.}], None
        with patch("kflame.flame.solver.newton_solve", side_effect=newton), \
             patch("kflame.flame.solver._residual_inf", return_value=5.):
            _, ok, history = _hybrid_newton(p, x, opts)
        self.assertFalse(ok)
        self.assertEqual([c["max_iter"] for c in calls if c["rdt"] > 0], [1, 1])
        self.assertTrue(all(h["scheme"] == "PTC-SER" for h in history if h["phase"] == "transient"))

    @unittest.skipIf(ct is None, 'Optional independent Cantera reference')
    def test_hydrogen_thermochemistry_at_low_and_high_pressure(self) -> None:
        for mechanism in ("gri30.yaml", "h2o2.yaml"):
            case = _case(soret=True)
            case.mech = str(Path(ct.get_data_directories()[-1]) / mechanism)
            for pressure in (ct.one_atm, 10 * ct.one_atm):
                case.P = pressure
                native = NativeSpeciesBackend(SimpleNamespace(case=case, P=pressure))
                gas = ct.Solution(case.mech)
                gas.TP = 300., pressure
                gas.set_equivalence_ratio(1., "H2", "O2:1,N2:3.76")
                fresh = gas.Y.copy()
                gas.equilibrate("HP")
                burnt = gas.Y.copy()
                temperature = np.array([300., 500., 800., 1200., 1800., 2400.])
                y = fresh[:, None] * (1 - np.linspace(0, 1, 6)) + burnt[:, None] * np.linspace(0, 1, 6)
                omega, enthalpy = np.empty_like(y), np.empty_like(y)
                rho, cp = native.eval_grid_thermo_kinetics_into(temperature, y, omega, enthalpy)
                for j in range(temperature.size):
                    with self.subTest(mechanism=mechanism, pressure=pressure, temperature=temperature[j]):
                        gas.TPY = temperature[j], pressure, y[:, j]
                        reference = gas.net_production_rates * gas.molecular_weights
                        np.testing.assert_allclose(rho[j], gas.density, rtol=1e-10)
                        np.testing.assert_allclose(cp[j], gas.cp_mass, rtol=1e-10)
                        np.testing.assert_allclose(enthalpy[:, j], gas.partial_molar_enthalpies, rtol=1e-10, atol=1e-6)
                        np.testing.assert_allclose(omega[:, j], reference, rtol=1e-7,
                                                   atol=1e-9 * max(1., np.max(np.abs(reference))))

    def test_restored_coarse_grid_is_not_mesh_convergence(self) -> None:
        p = FreeFlameProblem(_case(soret=True), n_points=4)
        x = p.make_initial_guess()
        z_old = p.z.copy()
        z_new = np.sort(np.append(z_old, .5 * (z_old[1] + z_old[2])))
        refiner = SimpleNamespace(refine=lambda *args, **kw: (z_new, True, 1, 0))
        accepted = dict(Finf=0.0, Finf_limit=1e4, residual_accepted=True,
                        weighted_step_norm=0.0, weighted_step_norm_limit=1.0,
                        weighted_step_accepted=True, residual_guard_inf=1e4,
                        residual_guard_accepted=True, criterion="cantera", accepted=True)
        with patch("kflame.flame.solver.AdaptiveRefiner", return_value=refiner), \
             patch("kflame.flame.solver._refresh_backend"), \
             patch("kflame.flame.solver._acceptance_status", return_value=accepted), \
             patch("kflame.flame.solver._hybrid_newton", side_effect=lambda prob, state, opts, **kw:
                   (state, False, [{"status": "max_time_step_count"}])):
            actual, ok, log = _refine_and_solve(p, x, SolveOptions(require_grid_convergence=True), None)
        self.assertFalse(ok)
        self.assertTrue(log[-1]["restored"])
        self.assertFalse(log[-1]["grid_converged"])
        self.assertTrue(np.isnan(p.last_weighted_step_norm))
        np.testing.assert_array_equal(actual, x)
        np.testing.assert_array_equal(p.z, z_old)

    def test_domain_check_only_uses_a_solved_energy_balance(self) -> None:
        p = SimpleNamespace(n_points=3, n_species=1, setup_fixed_temperature=lambda **kw: None)
        x = pack_state(np.ones(3), np.array([300., 700., 1000.]), np.ones((1, 3)))
        checks = []
        modes = []
        def hybrid(problem, state, opts, **kw):
            modes.append(problem.solve_energy)
            ok = len(modes) != 1  # A fails; B and C succeed
            if ok and kw["steady_callback"] is not None:
                kw["steady_callback"](state)
            return state, ok, [{"status": "ok" if ok else "failed"}]
        with patch("kflame.flame.solver._hybrid_newton", side_effect=hybrid):
            _, ok, _ = _solve_auto_stages(p, x, SolveOptions(), None, refine_grid=False,
                                         width_check=lambda state: checks.append(p.solve_energy))
        self.assertTrue(ok)
        self.assertEqual(modes, [True, False, True])
        self.assertEqual(checks, [True])

    def test_final_certificate_forces_exact_transport(self) -> None:
        """A tiny lagged correction cannot hide a large full residual."""
        problem = SimpleNamespace(last_weighted_step_norm=1e-6,
                                  lag_multicomponent_transport=True)
        options = SolveOptions(acceptance_criterion="cantera", residual_guard_inf=1e4)

        def residual_probe(x, p, *, force_exact_transport=False):
            return np.array([2e4 if force_exact_transport else 0.0])

        with patch("kflame.flame.solver.residual", side_effect=residual_probe) as probe:
            result = _acceptance_status(problem, np.array([1.0]), options)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["Finf"], 2e4)
        self.assertTrue(probe.call_args.kwargs["force_exact_transport"])

    def test_nonfinite_state_or_residual_never_certifies(self) -> None:
        problem = SimpleNamespace(last_weighted_step_norm=0.0)
        for criterion in ("residual", "combined", "cantera"):
            options = SolveOptions(acceptance_criterion=criterion,
                                   final_Finf_limit=float("inf"), residual_guard_inf=float("inf"))
            for state, value in ((1.0, np.nan), (1.0, np.inf), (np.nan, 0.0)):
                with self.subTest(criterion=criterion, state=state, residual=value):
                    with patch("kflame.flame.solver.residual", return_value=np.array([value])):
                        result = _acceptance_status(problem, np.array([state]), options)
                    self.assertFalse(result["accepted"])

    def test_mixture_soret_constructs_native_backend(self) -> None:
        problem = SimpleNamespace(case=_case(soret=True, transport="mixture-averaged"), P=ONE_ATM)
        backend = NativeSpeciesBackend(problem)
        self.assertTrue(backend.soret_enabled)
        self.assertFalse(backend.uses_multicomponent_flux)
        if ct is not None:
            self.assertTrue(SpeciesBackend(problem).soret_enabled)

    @unittest.skipIf(ct is None, 'Optional independent Cantera reference')
    def test_mixture_soret_coefficients_and_mass_closure(self):
        for mechanism in ('h2o2.yaml', 'gri30.yaml'):
            for pressure in (ONE_ATM, 10 * ONE_ATM):
                case = _case(soret=True, transport='mixture-averaged')
                case.mech, case.P = mechanism, pressure
                backend = NativeSpeciesBackend(SimpleNamespace(case=case, P=pressure))
                reference = SpeciesBackend(SimpleNamespace(case=case, P=pressure))
                gas = ct.Solution(mechanism, transport_model='mixture-averaged')
                gas.TP = 300., pressure
                gas.set_equivalence_ratio(1., 'H2', 'O2:1,N2:3.76')
                fresh = gas.Y.copy()
                gas.equilibrate('HP')
                random = np.random.default_rng(20).random(gas.n_species)
                random /= random.sum()
                pure = np.zeros(gas.n_species)
                pure[gas.species_index('N2')] = 1.
                for temperature, y in ((300., fresh), (1000., .5 * (fresh + gas.Y)),
                                       (2400., random), (800., pure)):
                    with self.subTest(mechanism=mechanism, pressure=pressure, temperature=temperature):
                        actual = backend.eval_mixture_thermal_diffusion(np.array([temperature]), y[:, None])[:, 0]
                        gas.TPY = temperature, pressure, y
                        np.testing.assert_allclose(actual, gas.thermal_diff_coeffs, rtol=2e-8, atol=2e-14)
                        expected = reference.eval_mixture_thermal_diffusion(np.array([temperature]), y[:, None])[:, 0]
                        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-14)
                        self.assertLess(abs(actual.sum()), 1e-18)

    def test_mixture_soret_full_scalar_and_batched_residuals(self):
        case = _case(soret=True, transport='mixture-averaged')
        case.mech = 'h2o2.yaml'
        problem = FreeFlameProblem(case, n_points=4)
        problem.backend = NativeSpeciesBackend(problem)
        problem.z = np.array([0., 1e-4, 2e-4, 3e-4])
        blend = np.linspace(0, 1, 4)
        y = problem.Y_in[:, None] * (1 - blend) + problem.Y_eq[:, None] * blend
        # Include Newton's allowed negative trial species.
        y[problem.species_names.index('H'), 1] = -1e-8
        x = pack_state(np.full(4, 2.), np.array([300., 700., 1500., 2200.]), y)
        full = residual(x, problem)
        self.assertIsNone(problem.last_residual_error)
        cache = build_local_jacobian_cache(x, problem)
        nv = problem.n_species + 2
        for center in range(4):
            rows, scalar_base = residual_local_rows(x, problem, center, cache)
            np.testing.assert_allclose(scalar_base, full[rows], rtol=2e-10, atol=2e-7)
            cols = np.arange(center * nv, (center + 1) * nv)
            values = x[cols] + 1e-6 * np.abs(x[cols]) + 1e-9
            problem.use_numba_local_jacobian = True
            rows, compiled = residual_local_rows_batch_perturbed(x, problem, center, cols, values, cache)
            problem.use_numba_local_jacobian = False
            _, python_batch = residual_local_rows_batch_perturbed(x, problem, center, cols, values, cache)
            np.testing.assert_allclose(compiled, python_batch, rtol=2e-10, atol=2e-7)
            for index, col in enumerate(cols):
                trial = x.copy()
                trial[col] = values[index]
                _, scalar = residual_local_rows(trial, problem, center, cache)
                np.testing.assert_allclose(compiled[index], scalar, rtol=2e-10, atol=2e-7)

    @unittest.skipIf(ct is None, 'Optional independent Cantera reference')
    def test_native_backend_matches_multicomponent_soret_face_data(self) -> None:
        """The KFLAME face kernel must not query Cantera after construction."""
        case = _case(soret=True)
        case.mech = str(Path(ct.get_data_directories()[-1]) / "gri30.yaml")
        problem = SimpleNamespace(
            case=case,
            P=ct.one_atm,
            transport_model="multicomponent",
            soret_enabled=True,
            flux_gradient_basis="molar",
        )
        native = NativeSpeciesBackend(problem)
        reference = SpeciesBackend(problem)

        gas = ct.Solution(case.mech)
        gas.TP = case.T_in, case.P
        gas.set_equivalence_ratio(case.phi, case.fuel, case.oxidizer)
        Y = np.repeat(gas.Y[:, None], 3, axis=1)
        T = np.array([700.0, 1200.0, 1800.0])

        actual = native.eval_multicomponent_face_transport(T, Y)
        expected = reference.eval_multicomponent_face_transport(T, Y)
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want, rtol=2.0e-8, atol=2.0e-12)
        self.assertEqual(native.backend_kind, "native")

    def test_native_construction_and_evaluation_without_cantera(self) -> None:
        case = _case(soret=True)
        case.mech = resolve_mechanism('gri30.yaml')
        problem = SimpleNamespace(case=case, P=ONE_ATM)
        original_import = builtins.__import__

        def no_cantera(name, *args, **kwargs):
            if name == "cantera" or name.startswith("cantera."):
                raise AssertionError("Native Soret attempted to import Cantera")
            return original_import(name, *args, **kwargs)

        ct_guard = patch.object(ct, 'Solution', side_effect=AssertionError('Native Soret called Cantera')) if ct is not None else nullcontext()
        with patch("builtins.__import__", side_effect=no_cantera), ct_guard:
            native = NativeSpeciesBackend(problem)
            y = np.zeros((native.n_species, 1)) if hasattr(native, "n_species") else np.zeros((len(native.W), 1))
            y[native.mech.species_names.index("N2")] = 0.8
            y[native.mech.species_names.index("H2")] = 0.2
            actual = native.eval_multicomponent_face_transport(np.array([800.0]), y)
        self.assertTrue(all(np.isfinite(a).all() for a in actual))

    @unittest.skipIf(ct is None, 'Optional independent Cantera reference')
    def test_native_transport_with_a_different_species_set(self) -> None:
        case = _case(soret=True)
        case.mech = str(Path(ct.get_data_directories()[-1]) / "h2o2.yaml")
        problem = SimpleNamespace(case=case, P=ct.one_atm)
        native = NativeSpeciesBackend(problem)
        reference = SpeciesBackend(problem)
        gas = ct.Solution(case.mech)
        gas.TP = 300.0, case.P
        gas.set_equivalence_ratio(1.0, "H2", "O2:1,N2:3.76")
        fresh = gas.Y.copy()
        gas.equilibrate("HP")
        y = np.column_stack((fresh, .5 * (fresh + gas.Y), gas.Y))
        temperature = np.array([300.0, 1200.0, 2400.0])
        for got, want in zip(native.eval_multicomponent_face_transport(temperature, y),
                             reference.eval_multicomponent_face_transport(temperature, y)):
            np.testing.assert_allclose(got, want, rtol=2e-6, atol=2e-12)

    @unittest.skipIf(ct is None, 'Optional independent Cantera reference')
    def test_schur_matches_dense_and_reference_in_reactive_states(self) -> None:
        case = _case(soret=True)
        case.mech = str(Path(ct.get_data_directories()[-1]) / "gri30.yaml")
        problem = SimpleNamespace(case=case, P=ct.one_atm)
        native = NativeSpeciesBackend(problem)
        gas = ct.Solution(case.mech, transport_model="multicomponent")
        gas.TP = 300.0, ct.one_atm
        gas.set_equivalence_ratio(1.0, "H2", "O2:1,N2:3.76")
        fresh = gas.Y.copy()
        gas.equilibrate("HP")
        burnt = gas.Y.copy()
        rng = np.random.default_rng(934)
        random_y = rng.random(len(fresh))
        random_y /= random_y.sum()
        for pressure in (ct.one_atm, 10 * ct.one_atm):
            problem.P = pressure
            native = NativeSpeciesBackend(problem)
            for temperature, y in ((300.0, fresh), (1200.0, .5 * (fresh + burnt)),
                                   (2400.0, burnt), (1700.0, random_y)):
                with self.subTest(pressure=pressure, temperature=temperature):
                    native.multicomponent_transport.linear_solver = "schur"
                    reduced = native.eval_multicomponent_face_transport(np.array([temperature]), y[:, None])
                    native.multicomponent_transport.use_compiled_faces = False
                    python_faces = native.eval_multicomponent_face_transport(np.array([temperature]), y[:, None])
                    native.multicomponent_transport.use_compiled_faces = True
                    for got, want in zip(reduced, python_faces):
                        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)
                    native.multicomponent_transport.linear_solver = "dense"
                    dense = native.eval_multicomponent_face_transport(np.array([temperature]), y[:, None])
                    for got, want in zip(reduced, dense):
                        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)
                    gas.TPY = temperature, pressure, y
                    expected = (gas.density, gas.thermal_conductivity, gas.mean_molecular_weight,
                                gas.multi_diff_coeffs, gas.thermal_diff_coeffs)
                    for got, want in zip(reduced, expected):
                        # Independently fitted collision data and dense/Schur
                        # solves retain the established reference tolerance.
                        np.testing.assert_allclose(np.squeeze(got), want, rtol=2e-6, atol=2e-12)
                    dt = reduced[-1][:, 0]
                    self.assertLess(abs(dt.sum()), 1e-10 * np.linalg.norm(dt) + 1e-14)

    @unittest.skipIf(ct is None, 'Optional independent Cantera reference')
    def test_compiled_and_python_multicomponent_local_perturbations_agree(self) -> None:
        case = _case(soret=True)
        problem = FreeFlameProblem(case, n_points=4)
        problem.backend = SpeciesBackend(problem)
        problem.z = np.array([0.0, 1e-4, 2e-4, 3e-4])
        gas = ct.Solution(case.mech)
        gas.TP = 300, ct.one_atm
        gas.set_equivalence_ratio(1, "H2", "O2:1,N2:3.76")
        fresh = gas.Y.copy()
        gas.equilibrate("HP")
        y = fresh[:, None] * (1 - np.linspace(0, 1, 4)) + gas.Y[:, None] * np.linspace(0, 1, 4)
        x = pack_state(np.full(4, 2.0), np.array([300., 700., 1500., 2200.]), y)
        cache = build_local_jacobian_cache(x, problem)
        nv = problem.n_species + 2
        for center in range(4):
            cols = np.arange(center * nv, (center + 1) * nv)
            values = x[cols] + 1e-6 * np.abs(x[cols]) + 1e-9
            problem.use_numba_local_jacobian = True
            rows, compiled = residual_local_rows_batch_perturbed(x, problem, center, cols, values, cache)
            problem.use_numba_local_jacobian = False
            fallback_rows, fallback = residual_local_rows_batch_perturbed(x, problem, center, cols, values, cache)
            np.testing.assert_array_equal(rows, fallback_rows)
            np.testing.assert_allclose(compiled, fallback, rtol=2e-10, atol=2e-7)
            for i, col in enumerate(cols):
                trial = x.copy()
                trial[col] = values[i]
                scalar_rows, scalar = residual_local_rows(trial, problem, center, cache)
                np.testing.assert_array_equal(rows, scalar_rows)
                np.testing.assert_allclose(compiled[i], scalar, rtol=2e-10, atol=2e-7)

    def test_native_lagged_transport_equals_exact_at_refresh_state(self) -> None:
        """Lagging may reuse a model, never alter it at the refresh state."""
        case = _case(soret=True)
        case.mech = resolve_mechanism('gri30.yaml')
        problem = FreeFlameProblem(case, n_points=4)
        problem.backend = NativeSpeciesBackend(problem)
        problem.use_numba_residual = False
        problem.lag_multicomponent_transport = True
        problem.z = np.array([0.0, 1.0e-4, 2.0e-4, 3.0e-4])
        problem.n_points = int(problem.z.size)

        Y = np.repeat(problem.Y_in[:, None], problem.n_points, axis=1)
        x = pack_state(
            np.full(problem.n_points, 2.0),
            np.array([300.0, 700.0, 1500.0, 2200.0]),
            Y,
        )
        exact = residual(x, problem, force_exact_transport=True)
        build_local_jacobian_cache(x, problem)
        lagged = residual(x, problem)
        np.testing.assert_allclose(lagged, exact, rtol=2.0e-10, atol=2.0e-10)

    @unittest.skipIf(ct is None, 'Optional independent Cantera reference')
    def test_face_flux_matches_public_flow1d_formula_and_mass_closure(self) -> None:
        case = _case(soret=True)
        problem = SimpleNamespace(case=case, P=ct.one_atm, soret_enabled=True,
                                  flux_gradient_basis="molar")
        backend = SpeciesBackend(problem)
        gas = ct.Solution(case.mech)
        gas.TP = case.T_in, case.P
        gas.set_equivalence_ratio(case.phi, case.fuel, case.oxidizer)
        Y_u = gas.Y.copy()
        gas.equilibrate("HP")
        Y_hot = gas.Y.copy()

        T_left = np.array([300.0])
        T_right = np.array([1750.0])
        Y_left = Y_u[:, None]
        Y_right = Y_hot[:, None]
        T_face = 0.5 * (T_left + T_right)
        Y_face = 0.5 * (Y_left + Y_right)
        rho, _lam, Wmix, multi, dthermal = backend.eval_multicomponent_face_transport(
            T_face, Y_face
        )
        dz = np.array([2.5e-5])
        actual = _multicomponent_flux(
            Y_left, Y_right, T_left, T_right, rho, Wmix, multi, dthermal, dz,
            backend.W,
        )

        # Independent transcription of the documented Flow1D face expression.
        W = backend.W
        XL = Y_left / (W[:, None] * np.sum(Y_left / W[:, None], axis=0))
        XR = Y_right / (W[:, None] * np.sum(Y_right / W[:, None], axis=0))
        ordinary = np.einsum("kif,if->kf", multi, W[:, None] * (XR - XL) / dz)
        ordinary *= rho[None, :] * W[:, None] / Wmix[None, :] ** 2
        expected = ordinary - dthermal * (T_right - T_left) / (T_face * dz)[None, :]

        np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=2.0e-13)
        self.assertLess(abs(float(np.sum(actual))), 1.0e-10 * np.linalg.norm(actual) + 1.0e-12)
        self.assertGreater(np.linalg.norm(dthermal * (T_right - T_left) / (T_face * dz)), 0.0)

    @unittest.skipIf(ct is None, 'Optional independent Cantera reference')
    def test_full_and_frozen_local_residual_agree_at_base_state(self) -> None:
        case = _case(soret=True)
        problem = FreeFlameProblem(case, n_points=4)
        problem.backend = SpeciesBackend(problem)
        problem.use_numba_residual = False
        problem.z = np.array([0.0, 1.0e-4, 2.0e-4, 3.0e-4])
        problem.n_points = int(problem.z.size)
        gas = ct.Solution(case.mech)
        gas.TP = case.T_in, case.P
        gas.set_equivalence_ratio(case.phi, case.fuel, case.oxidizer)
        Y = np.repeat(gas.Y[:, None], problem.n_points, axis=1)
        x = pack_state(
            np.full(problem.n_points, 2.0),
            np.array([300.0, 700.0, 1500.0, 2200.0]),
            Y,
        )
        full = residual(x, problem)
        cache = build_local_jacobian_cache(x, problem)
        rows, local = residual_local_rows(x, problem, center_j=2, cache=cache)
        np.testing.assert_allclose(local, full[rows], rtol=2.0e-11, atol=2.0e-11)
        self.assertTrue(np.all(np.isfinite(full)))

    def test_native_signed_trial_residual_matches_frozen_local_base(self) -> None:
        case = _case(soret=True)
        case.mech = resolve_mechanism('gri30.yaml')
        case.P = 10 * ONE_ATM
        problem = FreeFlameProblem(case, n_points=4)
        problem.backend = NativeSpeciesBackend(problem)
        problem.z = np.array([0., 1e-4, 2e-4, 3e-4])
        problem.n_points = 4
        y = np.repeat(problem.Y_in[:, None], 4, axis=1)
        y[problem.species_names.index('H'), :] = -1e-8
        x = pack_state(np.full(4, 2.), np.array([300., 700., 1500., 2200.]), y)
        for compiled in (False, True):
            problem.use_numba_residual = compiled
            full = residual(x, problem, force_exact_transport=True)
            cache = build_local_jacobian_cache(x, problem)
            rows, local = residual_local_rows(x, problem, center_j=2, cache=cache)
            with self.subTest(compiled=compiled):
                np.testing.assert_allclose(local, full[rows], rtol=2e-10, atol=2e-9)
                self.assertTrue(np.all(np.isfinite(full)))

    @unittest.skipIf(ct is None, 'Optional independent Cantera reference')
    def test_multicomponent_soret_block_jacobian_factorizes(self) -> None:
        """The reference closure preserves KFLAME's local block linear algebra."""
        case = _case(soret=True)
        problem = FreeFlameProblem(case, n_points=4)
        problem.backend = SpeciesBackend(problem)
        problem.use_numba_residual = False
        problem.z = np.array([0.0, 1.0e-4, 2.0e-4, 3.0e-4])
        problem.n_points = int(problem.z.size)
        problem.jacobian_mode = "block_tridiag"
        problem.jacobian_rel_perturb = 1.0e-5
        problem.jacobian_abs_perturb = 1.0e-10
        problem.jacobian_threshold = 0.0
        gas = ct.Solution(case.mech)
        gas.TP = case.T_in, case.P
        gas.set_equivalence_ratio(case.phi, case.fuel, case.oxidizer)
        Y = np.repeat(gas.Y[:, None], problem.n_points, axis=1)
        x = pack_state(
            np.full(problem.n_points, 2.0),
            np.array([300.0, 700.0, 1500.0, 2200.0]),
            Y,
        )
        jacobian, _diag = build_jacobian_steady(
            lambda state, prob: residual(state, prob), x, problem
        )
        linear = factorize(jacobian)
        step = solve_linear(linear, -residual(x, problem))
        self.assertTrue(np.all(np.isfinite(step)))


if __name__ == "__main__":
    unittest.main()
