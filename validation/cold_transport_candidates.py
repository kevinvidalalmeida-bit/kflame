"""Isolated transport-action and homotopy experiments, never production imports."""
from contextlib import contextmanager
from functools import lru_cache
import inspect
import textwrap
import time
from unittest.mock import patch
import numpy as np
from numba import njit
import equations
import solver
import transport_multicomponent_kernels as transport_kernels
from cold_strategy_kernels import replace_once


@lru_cache(maxsize=2)
def build_action(shared=False):
    # Schur elimination needs A^-1 B, not A^-1 itself. This still has K RHSs;
    # avoiding an inverse does NOT remove all cubic work in the thermal solve.
    thermal_source = textwrap.dedent(inspect.getsource(transport_kernels.thermal_system.py_func))
    thermal_source = thermal_source[thermal_source.index('def '):]
    thermal_source = replace_once(thermal_source,
        '        if l00_inverse is None:\n            l00_inverse = np.linalg.solve(l[:n, :n], np.eye(n))\n        ainv_b = l00_inverse @ l01',
        '        ainv_b = np.linalg.solve(l[:n, :n], l01)')
    if shared:
        thermal_source = replace_once(thermal_source, 'l00_inverse, dense):', 'l00_inverse, dense, gradient):')
        thermal_source = replace_once(thermal_source, '        solution = np.linalg.solve(l, rhs)',
            '        solution = np.linalg.solve(l, rhs)\n        response = np.linalg.solve(l00, gradient)')
        thermal_source = replace_once(thermal_source,
            '        ainv_b = np.linalg.solve(l[:n, :n], l01)',
            '        combined = np.empty((n, n+1))\n        combined[:, :n] = l01\n'
            '        combined[:, n] = gradient\n        solved = np.linalg.solve(l[:n, :n], combined)\n'
            '        ainv_b = solved[:, :n]\n        response = solved[:, n]')
        thermal_source = replace_once(thermal_source, '    return dthermal, conductivity',
                                       '    return dthermal, conductivity, response')
    ns = dict(vars(transport_kernels))
    exec(compile(thermal_source, '<transport-action-thermal>', 'exec'), ns)
    ns['thermal_system'] = njit(ns['thermal_system'])
    source = textwrap.dedent(inspect.getsource(transport_kernels.evaluate_faces.py_func))
    source = source[source.index('def '):]
    source = replace_once(source, 'cstar_fit, dense):', 'cstar_fit, dense, gradient, grad_log_t):')
    source = replace_once(source, '    multi = np.empty((faces, n, n))', '    flux = np.empty((n, faces))')
    source = replace_once(source, '        inverse = np.linalg.solve(l00, np.eye(n))',
                          '        inverse = None\n        response = np.linalg.solve(l00, np.ascontiguousarray(gradient[:, face]))')
    source = replace_once(source,
        '            for j in range(n):\n                multi[face, k, j] = prefactor * x[k] / mw[j] * (inverse[k, j] - inverse[k, k])',
        '            density = pressure * mix_weight / (R_UNIV * t)\n'
        '            flux[k, face] = (density * mw[k] / (mix_weight**2) * prefactor * x[k] * response[k]\n'
        '                             - dt[k] * grad_log_t[face])')
    source = replace_once(source, 'return rho, conductivity, wmix, multi.transpose(1, 2, 0), thermal.T',
                          'return conductivity, flux')
    if shared:
        source = replace_once(source,
            '        response = np.linalg.solve(l00, np.ascontiguousarray(gradient[:, face]))\n', '')
        source = replace_once(source, '        dt, lam = thermal_system(',
                               '        dt, lam, response = thermal_system(')
        source = replace_once(source, 'mu, l00, inverse, dense)',
                               'mu, l00, inverse, dense, np.ascontiguousarray(gradient[:, face]))')
    exec(compile(source, '<transport-action-faces>', 'exec'), ns)
    return njit(parallel=True)(ns['evaluate_faces'])


def action_flux(kernel, backend, y_left, y_right, t_left, t_right, dz):
    transport = backend.multicomponent_transport
    t = .5*(t_left+t_right)
    y = .5*(y_left+y_right)
    gradient = (equations._mole_fractions_from_Y(y_right, backend.W)
                - equations._mole_fractions_from_Y(y_left, backend.W))/dz
    # Mole fractions sum to one mathematically. Enforce the corresponding
    # zero-sum gradient to roundoff in its largest component; not a mass clip.
    largest = np.argmax(np.abs(gradient), axis=0)
    for face in range(dz.size):
        gradient[largest[face], face] -= gradient[:, face].sum()
    grad_log_t = (t_right-t_left)/(np.maximum(t, 1e-300)*dz)
    if not backend.soret_enabled:
        grad_log_t = np.zeros_like(grad_log_t)
    return kernel(t, backend._P_float, y, np.asarray(backend.thermo.cp_R(t)),
        transport.mw, transport.inv_w, transport._log_eps_pair, transport.epsilon_k,
        transport.crot, transport.zrot, transport.frot_298, transport._viscosity_fit,
        transport._diffusion_fit, transport._astar_poly, transport._bstar_poly,
        transport._cstar_poly, False, gradient, grad_log_t)


@contextmanager
def action_context(records, shared=False):
    kernel = build_action(shared)
    def evaluate(backend, yl, yr, tl, tr, dz):
        try:
            result = action_flux(kernel, backend, yl, yr, tl, tr, dz)
            records.append(dict(kind='transport_action', faces=int(dz.size), shared_factorization=shared))
            return result
        except Exception as exc:
            records.append(dict(kind='transport_action', exception=repr(exc)))
            raise
    source = inspect.getsource(equations.residual)
    begin = source.index('            face_data = eval_multi_face(T_face, Y_face)')
    end = source.index('        elif hasattr(backend, "eval_faces"):', begin)
    source = source[:begin] + '''            dz_face = z[1:] - z[:-1]
            lam_face, direct_flux = experimental_action(
                backend, Y[:, :-1], Y[:, 1:], T[:-1], T[1:], dz_face)
            flux[:] = direct_flux
''' + source[end:]
    ns = dict(vars(equations), experimental_action=evaluate)
    exec(compile(source, '<transport-action-residual>', 'exec'), ns)
    # Jacobian keeps the existing complete coefficients and exact frozen
    # rational composition updates. Only non-lagged residual transport changes.
    with patch.object(equations, 'residual', ns['residual']), patch.object(solver, 'residual', ns['residual']):
        yield


@contextmanager
def blend_context(problem, mixture_backend, alpha, records):
    original_residual = equations.residual
    original_jacobian = equations.build_jacobian_steady
    @contextmanager
    def mixture():
        with patch.object(problem, 'residual_backend', mixture_backend, create=True), \
             patch.object(problem, 'jacobian_backend', mixture_backend, create=True):
            yield
    def residual(x, prob, **kwargs):
        if prob is not problem:
            return original_residual(x, prob, **kwargs)
        with mixture():
            fm = original_residual(x, prob, **kwargs)
        ft = original_residual(x, prob, **kwargs)
        return (1-alpha)*fm + alpha*ft
    def jacobian(fun, x, prob, eps=1e-5):
        if prob is not problem:
            return original_jacobian(fun, x, prob, eps)
        with mixture():
            jm, _ = original_jacobian(lambda z, p: original_residual(z, p), x, prob, eps)
        jt, _ = original_jacobian(lambda z, p: original_residual(z, p), x, prob, eps)
        for key in ('lower', 'diag', 'upper'):
            setattr(jt, key, (1-alpha)*getattr(jm, key)+alpha*getattr(jt, key))
        records.append(dict(kind='homotopy_jacobian', alpha=alpha, nodes=prob.n_points))
        return jt, jt.diagonal().copy()
    with patch.object(solver, 'residual', residual), patch.object(equations, 'residual', residual), \
         patch.object(solver, 'build_jacobian_steady', jacobian):
        yield


@contextmanager
def homotopy_context(records):
    # Fixed two-interval prototype of a continuous residual homotopy. No
    # pressure-specific policy, no adaptive-step claims, alpha=1 final gate.
    def intermediate(problem, coarse, x, options):
        start = time.perf_counter()
        # Intermediate stage may refine; its failure is reported, never certified.
        opts = solver.replace(options, lag_multicomponent_transport=False)
        with blend_context(problem, coarse.backend, .5, records):
            state, ok, report = solver.solve_free_flame(problem, x0=x, options=opts)
        records.append(dict(kind='homotopy_stage', alpha=.5, accepted=bool(ok),
                            time_s=time.perf_counter()-start,
                            nodes=problem.n_points, Finf=report.get('Finf_final')))
        return state, ok, report
    source = inspect.getsource(solver._solve_transport_bootstrap)
    needle = '    x, ok, final = solve_free_flame(problem, x0=x, options=final_options)'
    replacement = '''    x, bridge_ok, bridge_report = experimental_intermediate(problem, coarse, x, final_options)
    if not bridge_ok:
        return x, False, dict(bridge_report, final_accepted=False,
            reason="homotopy_intermediate_failed", transport_bootstrap=first,
            total_time_s=time.perf_counter()-started)
    final_options = replace(final_options,
        max_total_time_s=max(0.01, opts.max_total_time_s-(time.perf_counter()-started)))
    x, ok, final = solve_free_flame(problem, x0=x, options=final_options)
    final["homotopy_intermediate"] = bridge_report
'''
    source = replace_once(source, needle, replacement)
    ns = dict(vars(solver), experimental_intermediate=intermediate)
    exec(compile(source, '<transport-homotopy-bootstrap>', 'exec'), ns)
    with patch.object(solver, '_solve_transport_bootstrap', ns['_solve_transport_bootstrap']):
        yield
