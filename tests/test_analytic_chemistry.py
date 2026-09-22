"""Independent finite differences and Cantera check the native chain rule."""
import os
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pytest
from kflame.chemistry.backend import NativeSpeciesBackend
from kflame.chemistry.analytic import species_partials, _product_partials
from kflame.flame.config import FlameCase
from kflame.flame.problem import FreeFlameProblem
from kflame.flame.equations import residual, build_local_jacobian_cache, build_jacobian_steady


def backend(mech, pressure=101325.):
    p = FreeFlameProblem(FlameCase(mech=mech, fuel='H2', P=pressure), n_points=8)
    p.backend = NativeSpeciesBackend(p)
    return p.backend


def rates(b, t, y):
    om, h = np.empty_like(y), np.empty_like(y)
    b.eval_grid_thermo_kinetics_into(t, y, om, h)
    return om


@pytest.mark.parametrize('mech', ['h2o2.yaml', 'gri30.yaml'])
@pytest.mark.parametrize('pressure', [101325., 1013250., 10132500.])
def test_species_derivatives(mech, pressure):
    b = backend(mech, pressure)
    rng = np.random.default_rng(281)
    t = np.array([350., 999.99, 1000.01, 1800., 2500.])
    y = rng.uniform(.01, 1, (b.n_species, t.size))
    y /= y.sum(axis=0)
    # Signed Newton states, kept away from the nonsmooth zero crossing.
    y[0, 3], y[1, 4] = -1e-4, -1e-4
    jac = species_partials(b, t, y)
    numerical = np.empty_like(jac)
    for j in range(b.n_species):
        step = 1e-6 * np.maximum(abs(y[j]), 1e-4)
        yp, ym = y.copy(), y.copy()
        yp[j] += step
        ym[j] -= step
        numerical[:, :, j] = (rates(b, t, yp)-rates(b, t, ym))/(2*step)
    scale = np.max(abs(jac), axis=2, keepdims=True)
    np.testing.assert_allclose(numerical/np.maximum(scale, 1e-10),
                               jac/np.maximum(scale, 1e-10), atol=3e-5, rtol=3e-5)


def test_zero_product_and_signed_continuation():
    d = np.empty(3)
    assert _product_partials(np.array([0., 2., 3.]), np.array([3, 0, 1, 2]), d) == 0
    np.testing.assert_array_equal(d, [6., 0., 0.])
    assert _product_partials(np.array([-1., -2., 3.]), np.array([3, 0, 1, 2]), d) == 0
    np.testing.assert_array_equal(d, 0.)


def test_cantera_concentration_derivatives():
    ct = pytest.importorskip('cantera')
    b = backend('gri30.yaml', 1013250.)
    gas = ct.Solution('gri30.yaml')
    gas.derivative_settings = {'skip-third-bodies': False, 'skip-falloff': False}
    gas.TPX = 1700., b._P_float, 'CH4:.1,O2:.2,N2:.6,H2O:.05,CO2:.05'
    y = gas.Y[:, None]
    a = species_partials(b, np.array([gas.T]), y)[:, 0]
    dc = gas.net_production_rates_ddCi
    C = gas.concentrations
    s = np.dot(b.invW, y[:, 0])
    expected = b.W[:, None]*b.invW[None, :]*(gas.density*dc - (dc@C)[:, None]/s)
    scale = np.maximum(np.max(abs(expected), axis=1, keepdims=True), 1e-9)
    np.testing.assert_allclose(a/scale, expected/scale, atol=3e-7, rtol=3e-7)


def test_residual_cache_cannot_alias_states_or_backends():
    b = backend('h2o2.yaml')
    p = b.problem
    x = p.make_initial_guess()
    residual(x, p)
    snapshot = p._residual_props_cache['x_snapshot'].copy()
    x[3] += 1e-3  # Does not change any of the old three checksum entries.
    np.testing.assert_array_equal(p._residual_props_cache['x_snapshot'], snapshot)
    hit = build_local_jacobian_cache(x, p)
    p._residual_props_cache = None
    fresh = build_local_jacobian_cache(x, p)
    np.testing.assert_array_equal(hit['omega'], fresh['omega'])
    residual(x, p)
    from unittest.mock import patch
    p.jacobian_backend = NativeSpeciesBackend(p)
    with patch.object(p.jacobian_backend, 'eval_grid_thermo_kinetics_into',
                      wraps=p.jacobian_backend.eval_grid_thermo_kinetics_into) as call:
        build_local_jacobian_cache(x, p)
        assert call.called


@pytest.mark.parametrize('transport,basis,soret', [
    ('mixture-averaged', 'molar', False),
    ('mixture-averaged', 'mass', False),
    ('mixture-averaged', 'molar', True),
    ('multicomponent', 'molar', True),
])
@pytest.mark.parametrize('energy', [True, False])
def test_hybrid_full_blocks(transport, basis, soret, energy):
    p = FreeFlameProblem(FlameCase(mech='gri30.yaml', transport_model=transport,
        flux_gradient_basis=basis, soret_enabled=soret), n_points=8)
    p.backend = NativeSpeciesBackend(p)
    p.solve_energy = energy
    p.jacobian_mode = 'block_tridiag'
    p.precompute_jacobian_thermo = True
    x = p.make_initial_guess()
    p.setup_fixed_temperature(T_profile=x.reshape(p.n_points, -1)[:, 1])
    p.analytic_chemistry = False
    fd, _ = build_jacobian_steady(residual, x, p)
    p.analytic_chemistry = True
    p._profile = {}
    an, _ = build_jacobian_steady(residual, x, p)
    assert p._profile['jacobian_analytic_chemistry']['count'] == 1
    for name in ('lower', 'diag', 'upper'):
        a, f = getattr(an, name), getattr(fd, name)
        scale = np.maximum(np.max(abs(f), axis=2, keepdims=True), 1.)
        np.testing.assert_allclose(a/scale, f/scale, atol=2e-4, rtol=2e-4)


def test_explicit_analytic_errors_are_not_silently_finite_differenced():
    from unittest.mock import patch
    b = backend('h2o2.yaml')
    p = b.problem
    p.analytic_chemistry = p.precompute_jacobian_thermo = True
    x = p.make_initial_guess()
    for mode in ('numba_local', 'block_tridiag', 'banded_lapack'):
        p.jacobian_mode = mode
        with patch('kflame.chemistry.analytic.hybrid_center_properties', side_effect=ValueError('unsupported')):
            with pytest.raises(ValueError, match='unsupported'):
                build_jacobian_steady(residual, x, p)
