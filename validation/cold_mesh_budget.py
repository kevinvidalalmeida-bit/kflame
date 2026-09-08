"""Exploratory interpolation-error budget; final acceptance is never relaxed.

Restriction/prolongation discrepancy is an indicator, NOT a proven truncation
error bound. At each Newton call the convergence threshold can be enlarged only
on a mesh which the unchanged production refiner still marks for modification.
When no changes are marked, an ordinary strict correction is mandatory.
"""
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import solver
from state import unpack_state


def indicator(problem, x, opts):
    n, nv = problem.n_points, problem.n_species+2
    if n < 5 or not problem.solve_energy or not np.isfinite(x).all():
        return 1., False
    u, t, y = unpack_state(x, n, problem.n_species)
    profiles = solver.build_freeflame_refiner_profiles(problem, u, t, y)
    refiner = solver.AdaptiveRefiner(ratio=opts.refine_ratio, slope=opts.refine_slope,
                                    curve=opts.refine_curve, prune=opts.refine_prune,
                                    grid_min=opts.refine_grid_min, max_points=opts.refine_max_points)
    _, changed, _, _ = refiner.refine(problem.z, profiles, all_Y=y, j_fixed=problem.j_fixed)
    keep = np.unique(np.r_[np.arange(0, n, 2), n-1, problem.j_fixed]).astype(int)
    values = np.asarray(x).reshape(n, nv)
    reconstructed = np.column_stack([np.interp(problem.z, problem.z[keep], values[keep, v])
                                     for v in range(nv)])
    eta = solver.weighted_norm((values-reconstructed).ravel(), x, problem)
    return float(np.clip(.1*eta, 1., 10.)) if changed else 1., bool(changed)


@contextmanager
def mesh_budget_context(records):
    original_newton = solver.newton_solve
    original_hybrid = solver._hybrid_newton
    original_accept = solver._acceptance_status
    active = {}

    def newton(fun, x, problem, *args, **kwargs):
        opts = active.get(id(problem))
        steady = float(kwargs.get('rdt', 0.)) == 0.
        budget = 1.
        if opts is not None and steady:
            budget, _ = indicator(problem, x, opts)
            kwargs['tol'] = min(float(kwargs.get('tol', opts.tol))*budget, 10.)
        output = original_newton(fun, x, problem, *args, **kwargs)
        if opts is not None and steady and budget > 1:
            # A loose correction can change the marking: do not return it as
            # final when the updated grid no longer needs refinement.
            _, changed = indicator(problem, output[0], opts)
            records.append(dict(kind='mesh_budget', nodes=problem.n_points, budget=budget,
                                marked_after=changed, nonlinear_ok=bool(output[1])))
            if output[1] and not changed:
                kwargs['tol'] = opts.tol
                kwargs['jac_state'] = output[3]
                output = original_newton(fun, output[0], problem, *args, **kwargs)
        problem._audit_spatial_budget = budget
        return output

    def hybrid(problem, x, opts, *args, **kwargs):
        active[id(problem)] = opts
        try:
            return original_hybrid(problem, x, opts, *args, **kwargs)
        finally:
            active.pop(id(problem), None)

    def acceptance(problem, x, opts):
        status = original_accept(problem, x, opts)
        if status['accepted']:
            return status
        budget = float(getattr(problem, '_audit_spatial_budget', 1.))
        if budget > 1 and status['finite_state_and_residual'] and status['residual_guard_accepted']:
            _, changed = indicator(problem, x, opts)
            if changed and status['weighted_step_norm'] <= budget*status['weighted_step_norm_limit']:
                # This permits another mesh cycle, never a final table row.
                status = dict(status, accepted=True, audit_intermediate_only=True)
        return status

    with patch.object(solver, 'newton_solve', newton), \
         patch.object(solver, '_hybrid_newton', hybrid), \
         patch.object(solver, '_acceptance_status', acceptance):
        yield
