"""Two-grid FAS correction experiment before a strict fine-grid corrector.

F_H(x_H) = F_H(R x_h) - R F_h(x_h). The tau term preserves fine-grid
fixed points. A coarse Newton correction is approximate; final certification
always belongs to the unmodified fine-grid solver. This is a two-grid prototype,
not a claim of a validated recursive multigrid implementation.
"""
import copy
from contextlib import contextmanager
import time
from unittest.mock import patch
import numpy as np
import equations
import solver


def restriction_indices(n, anchor):
    return np.unique(np.r_[np.arange(0, n, 2), n-1, anchor]).astype(int)


def fas_forcing(coarse_residual_at_injection, restricted_fine_residual):
    return coarse_residual_at_injection - restricted_fine_residual


def correction(problem, x, opts, deadline):
    start = time.perf_counter()
    record = dict(kind='two_grid', nodes=problem.n_points, accepted_correction=False)
    n, nv = problem.n_points, problem.n_species+2
    if n < 7 or not problem.solve_energy:
        return x, dict(record, reason='no_coarse_level', time_s=time.perf_counter()-start)
    keep = restriction_indices(n, problem.j_fixed)
    coarse = copy.copy(problem)
    # Never share mutable solver caches or profile counters with the fine grid.
    for key, value in vars(problem).items():
        if isinstance(value, np.ndarray):
            setattr(coarse, key, value.copy())
    for key in list(vars(coarse)):
        if key.startswith('_'):
            delattr(coarse, key)
    coarse.z = problem.z[keep].copy()
    coarse.n_points = keep.size
    coarse.j_fixed = int(np.flatnonzero(keep == problem.j_fixed)[0])
    coarse.T_profile_fixed = None
    coarse.lag_multicomponent_transport = False
    solver._refresh_backend(coarse)
    xc = np.asarray(x).reshape(n, nv)[keep].copy().ravel()
    fine_f = equations.residual(x, problem, force_exact_transport=True)
    coarse_f = equations.residual(xc, coarse, force_exact_transport=True)
    tau = fas_forcing(coarse_f, fine_f.reshape(n, nv)[keep].ravel())
    initial = xc.copy()
    # Build the derivative of F_H, NOT an FD matrix with a shifted base residual.
    fun = lambda state, prob: equations.residual(state, prob, force_exact_transport=True)
    if deadline and time.perf_counter() >= deadline:
        return x, dict(record, reason='deadline', time_s=time.perf_counter()-start)
    jac = equations._block_tridiag_jacobian_local(fun, xc, coarse, eps=opts.jac_eps)
    lu = equations.factorize(jac)
    steps = 0
    for _ in range(2):
        if deadline and time.perf_counter() >= deadline:
            break
        g = fun(xc, coarse)-tau
        step = equations.solve_linear(lu, -g)
        bound, _ = solver.bound_step_limit(xc, step, coarse)
        for level in range(7):
            alpha = min(1., bound)*2.**(-.5*level)
            trial = xc+alpha*step
            test = fun(trial, coarse)-tau
            if np.isfinite(test).all() and np.linalg.norm(test, np.inf) < np.linalg.norm(g, np.inf):
                xc = trial
                steps += 1
                break
        else:
            break
    dc = (xc-initial).reshape(keep.size, nv)
    delta = np.column_stack([np.interp(problem.z, coarse.z, dc[:, v]) for v in range(nv)]).ravel()
    bound, _ = solver.bound_step_limit(x, delta, problem)
    base_norm = float(np.linalg.norm(fine_f, np.inf))
    for level in range(7):
        alpha = min(1., bound)*2.**(-.5*level)
        trial = x+alpha*delta
        test = equations.residual(trial, problem, force_exact_transport=True)
        if np.isfinite(test).all() and np.linalg.norm(test, np.inf) < base_norm:
            return trial, dict(record, accepted_correction=True, alpha=alpha, coarse_steps=steps,
                               fine_residual_before=base_norm, fine_residual_after=float(np.linalg.norm(test, np.inf)),
                               time_s=time.perf_counter()-start)
    return x, dict(record, reason='no_fine_descent', coarse_steps=steps, time_s=time.perf_counter()-start)


@contextmanager
def two_grid_context(records):
    original = solver._hybrid_newton
    def hybrid(problem, x, opts, label='', steady_callback=None, deadline=None):
        if label.startswith('Post-refine'):
            try:
                x, record = correction(problem, x, opts, deadline)
            except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                # A failed coarse correction is visible and never accepted.
                record = dict(kind='two_grid', accepted_correction=False, exception=repr(exc))
            records.append(record)
        return original(problem, x, opts, label=label, steady_callback=steady_callback, deadline=deadline)
    with patch.object(solver, '_hybrid_newton', hybrid):
        yield
