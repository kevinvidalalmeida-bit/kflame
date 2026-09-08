"""One overlapping nonlinear block sweep on the first cold grid only.

Two-node local corrections hold all other nodes fixed and evaluate the full
nonlinear residual. A frozen local Jacobian supplies up to two modified Newton
corrections. There is no Krylov solve, coarse grid, or changed target equation.
"""
from contextlib import contextmanager
import inspect
import time
from unittest.mock import patch
import numpy as np
from scipy.linalg.lapack import dgetrf, dgetrs
import equations
import solver
from cold_strategy_kernels import replace_once


def local_matrix(jac, first):
    """Exact two-node diagonal submatrix, including neighbour coupling."""
    return np.block([[jac.diag[first], jac.upper[first]],
                     [jac.lower[first], jac.diag[first+1]]])


def sweep(problem, x, opts, records, deadline=None, global_descent=False):
    started = time.perf_counter()
    original = np.asarray(x).copy()
    current = original.copy()
    record = dict(kind='nonlinear_blocks', nodes=int(problem.n_points),
                  local_trials=0, local_updates=0, singular_blocks=0,
                  local_factorizations=0, accepted=False,
                  global_descent=bool(global_descent), accepted_merits=[])
    def finish(value, reason):
        record.update(reason=reason, time_s=time.perf_counter()-started)
        records.append(record)
        return value
    def expired():
        return deadline is not None and time.perf_counter() >= deadline
    if expired():
        return finish(original, 'deadline')
    f = solver.residual(current, problem, force_exact_transport=True)
    if not np.isfinite(f).all():
        return finish(original, 'nonfinite_initial')
    # Frozen row scales define only the experimental merit function. They do
    # not alter any equation, norm tolerance or the final certification gate.
    scale = np.maximum(np.abs(f), 1.)
    merit0 = float(np.linalg.norm(f/scale))
    record['merit_initial'] = merit0
    if merit0 == 0:
        return finish(original, 'already_root')
    try:
        jac, _ = solver.build_jacobian_steady(
            lambda v, p: solver.residual(v, p, force_exact_transport=True),
            current, problem, eps=opts.jac_eps)
        if not isinstance(jac, equations.BlockTridiagJacobian):
            return finish(original, 'unsupported_matrix')
        nv = jac.block_size
        for j in range(jac.n_blocks-1):
            if expired():
                return finish(original, 'deadline')
            sl = slice(j*nv, (j+2)*nv)
            lu, piv, info = dgetrf(local_matrix(jac, j))
            record['local_factorizations'] += 1
            if info > 0:
                record['singular_blocks'] += 1
                continue
            if info < 0:
                raise RuntimeError('Invalid local LU arguments')
            for _ in range(2):
                direction, info = dgetrs(lu, piv, -f[sl])
                if info != 0 or not np.isfinite(direction).all():
                    break
                step = np.zeros_like(current)
                step[sl] = direction
                alpha, _ = solver.bound_step_limit(current, step, problem)
                alpha = min(1., alpha)
                local0 = np.linalg.norm(f[sl]/scale[sl])
                improved = False
                for _ in range(4):
                    if expired():
                        return finish(original, 'deadline')
                    if alpha < opts.alpha_min:
                        break
                    trial = current+alpha*step
                    ft = solver.residual(trial, problem, force_exact_transport=True)
                    record['local_trials'] += 1
                    if (np.isfinite(ft).all() and np.linalg.norm(ft[sl]/scale[sl]) < local0
                            and (not global_descent or np.linalg.norm(ft/scale) < np.linalg.norm(f/scale))):
                        current, f = trial, ft
                        record['local_updates'] += 1
                        record['accepted_merits'].append(float(np.linalg.norm(f/scale)))
                        improved = True
                        break
                    alpha *= .5
                if not improved:
                    break
        merit1 = float(np.linalg.norm(f/scale))
        record['merit_final'] = merit1
        # Local improvements can harm other blocks. Roll back the entire sweep
        # unless the complete nonlinear residual improves on the fixed scales.
        record['accepted'] = bool(record['local_updates'] and merit1 < merit0)
        return finish(current if record['accepted'] else original,
                      'improved' if record['accepted'] else 'global_rollback')
    except Exception as exc:
        record['exception'] = repr(exc)
        return finish(original, 'exception_rollback')


@contextmanager
def nonlinear_blocks_context(records, global_descent=False):
    used = False
    def first_grid(problem, x, opts, deadline):
        nonlocal used
        if used:
            return x
        used = True
        return sweep(problem, x, opts, records, deadline, global_descent=global_descent)
    source = inspect.getsource(solver._hybrid_newton)
    source = replace_once(source, '    attempt = 0\n',
        '    x = first_grid_sweep(problem, x, opts, deadline)\n    attempt = 0\n')
    ns = dict(vars(solver), first_grid_sweep=first_grid)
    exec(compile(source, '<nonlinear-spatial-blocks>', 'exec'), ns)
    with patch.object(solver, '_hybrid_newton', ns['_hybrid_newton']):
        yield
