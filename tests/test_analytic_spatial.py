"""Direct frozen-transport Jacobian against independently perturbed residuals."""
import os
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pytest
from kflame.flame.config import FlameCase
from kflame.flame.problem import FreeFlameProblem
from kflame.chemistry.backend import NativeSpeciesBackend
from kflame.chemistry.analytic import thermochemical_partials
from kflame.flame.equations import (residual, build_jacobian_steady,
    build_local_jacobian_cache, residual_local_rows)


def test_production_defaults_and_explicit_numerical_override():
    from kflame.flame.solver import SolveOptions
    from kflame.fgm.generate import build_argparser, make_solve_options
    options = SolveOptions()
    assert options.analytic_spatial and options.jacobian_mode == 'block_tridiag'
    parser = build_argparser()
    assert make_solve_options(parser.parse_args([])).analytic_spatial
    assert not make_solve_options(parser.parse_args(['--no-analytic-spatial'])).analytic_spatial
    hybrid = make_solve_options(parser.parse_args(['--no-analytic-spatial','--analytic-chemistry']))
    assert hybrid.analytic_chemistry and not hybrid.analytic_spatial


@pytest.mark.parametrize('mech', ['h2o2.yaml', 'gri30.yaml'])
@pytest.mark.parametrize('pressure', [101325., 1013250., 10132500.])
def test_thermal_partials(mech, pressure):
    p = FreeFlameProblem(FlameCase(mech=mech, fuel='H2', P=pressure), n_points=8)
    b = NativeSpeciesBackend(p)
    rng = np.random.default_rng(89)
    T = np.array([350., 999.9, 1000.1, 1700., 2500.])
    Y = rng.uniform(.01, 1., (p.n_species, T.size))
    Y /= Y.sum(axis=0)
    Y[1, 3] = -1e-4
    drho, dcp, domega, dh = thermochemical_partials(b, T, Y)
    def evaluate(t):
        om, h = np.empty_like(Y), np.empty_like(Y)
        rho, cp = b.eval_grid_thermo_kinetics_into(t, Y, om, h)
        return rho, cp, om, h
    step = .001
    plus, minus = evaluate(T+step), evaluate(T-step)
    for exact, a, c in zip((drho[:, 1], dcp[:, 1], domega[:, :, 1].T, dh), plus, minus):
        numeric = (a-c)/(2*step)
        scale = np.maximum(abs(exact), 1e-5)
        np.testing.assert_allclose(numeric/scale, exact/scale, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize('transport,basis,soret', [
    ('mixture-averaged','molar',False), ('mixture-averaged','mass',False),
    ('mixture-averaged','molar',True), ('multicomponent','molar',True)])
@pytest.mark.parametrize('energy,outlet', [(True,'zero_gradient'), (False,'cantera_flux')])
def test_direct_columns(transport,basis,soret,energy,outlet):
    p = FreeFlameProblem(FlameCase(mech='h2o2.yaml', fuel='H2', transport_model=transport,
        flux_gradient_basis=basis,soret_enabled=soret,outlet_species_bc=outlet), n_points=8)
    p.backend = NativeSpeciesBackend(p)
    p.solve_energy = energy
    x = p.make_initial_guess()
    xr = x.reshape(p.n_points,-1)
    # Smooth interior composition and a reversed-flow node exercise both upwind branches.
    rng = np.random.default_rng(29)
    xr[:,2:] = rng.uniform(.01,1.,xr[:,2:].shape)
    xr[:,2:] /= xr[:,2:].sum(axis=1,keepdims=True)
    xr[:,1] = np.linspace(380.,2100.,p.n_points)
    xr[3,0] = -.4
    p.setup_fixed_temperature(T_profile=xr[:,1])
    p.solve_energy = energy
    p.jacobian_mode = 'block_tridiag'
    p.analytic_spatial = True
    direct,_ = build_jacobian_steady(residual,x,p)
    cache = build_local_jacobian_cache(x,p)
    nv = p.n_species+2
    for col in range(x.size):
        node,v = divmod(col,nv)
        step = 1e-5 if v == 1 else 1e-6*max(abs(x[col]),1e-3)
        xp,xm = x.copy(),x.copy()
        xp[col] += step
        xm[col] -= step
        rows,plus = residual_local_rows(xp,p,node,cache=cache)
        _,minus = residual_local_rows(xm,p,node,cache=cache)
        expected = (plus-minus)/(2*step)
        column = np.zeros(x.size)
        column[col] = 1.
        actual = direct.matvec(column)[rows]
        # Scale by row magnitude across all columns of its three blocks.
        scales = []
        for j in range(max(0,node-1),min(p.n_points,node+2)):
            scale = np.max(abs(direct.diag[j]),axis=1)
            if j>0: scale=np.maximum(scale,np.max(abs(direct.lower[j-1]),axis=1))
            if j<p.n_points-1: scale=np.maximum(scale,np.max(abs(direct.upper[j]),axis=1))
            scales.extend(np.maximum(scale,1.))
        np.testing.assert_allclose(actual/np.array(scales),expected/np.array(scales),atol=3e-6,rtol=3e-6)


def test_direct_does_not_call_perturbed_rates_or_residuals():
    from unittest.mock import patch
    p = FreeFlameProblem(FlameCase(mech='h2o2.yaml',fuel='H2'), n_points=8)
    p.backend = NativeSpeciesBackend(p)
    p.analytic_spatial = True
    p.jacobian_mode = 'block_tridiag'
    x = p.make_initial_guess()
    with patch.object(p.backend, 'eval_jacobian_thermo_kinetics_into', side_effect=AssertionError('FD rates')):
        with patch('kflame.flame.equations.residual_local_rows_batch_perturbed', side_effect=AssertionError('FD residual')):
            j,_ = build_jacobian_steady(residual,x,p)
            assert np.isfinite(j.diag).all()
    # The optional threshold must apply to boundary blocks as well.
    p.jacobian_threshold = 1e300
    filtered,_ = build_jacobian_steady(residual,x,p)
    np.testing.assert_array_equal(filtered.lower,0.)
    np.testing.assert_array_equal(filtered.upper,0.)
    for a,b in zip(j.diag,filtered.diag):
        np.testing.assert_array_equal(np.diag(a),np.diag(b))
        np.testing.assert_array_equal(b-np.diag(np.diag(b)),0.)


def test_analytic_partial_refresh_matches_full_selected_columns():
    from kflame.flame.equations import refresh_block_tridiag_jacobian_columns
    p = FreeFlameProblem(FlameCase(mech='h2o2.yaml',fuel='H2'), n_points=8)
    p.backend = NativeSpeciesBackend(p)
    p.analytic_spatial = True
    p.jacobian_mode = 'block_tridiag'
    x = p.make_initial_guess()
    old,_ = build_jacobian_steady(residual,x,p)
    x.reshape(p.n_points,-1)[2:5,1] += .1
    full,_ = build_jacobian_steady(residual,x,p)
    chosen = np.array([0,3,p.n_points-1])
    part = refresh_block_tridiag_jacobian_columns(residual,x,p,old,chosen)
    for j in range(p.n_points):
        ref = full if j in chosen else old
        np.testing.assert_array_equal(part.diag[j],ref.diag[j])
        if j>0: np.testing.assert_array_equal(part.upper[j-1],ref.upper[j-1])
        if j<p.n_points-1: np.testing.assert_array_equal(part.lower[j],ref.lower[j])
