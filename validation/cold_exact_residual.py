"""One-state residual handoff experiment; no persistent residual cache."""
from contextlib import contextmanager
import inspect
from unittest.mock import patch
import solver
from cold_strategy_kernels import replace_once


@contextmanager
def exact_residual_context(records):
    source = inspect.getsource(solver._hybrid_newton)
    source = replace_once(source, '        if ok_ss:\n',
                          '        if ok_ss:\n            exact_vector = None\n')
    source = replace_once(source,
        '                exact_finf = _residual_inf(problem, x_ss, force_exact_transport=True)',
        '                exact_vector = residual(x_ss, problem, force_exact_transport=True)\n'
        '                exact_finf = float(np.linalg.norm(exact_vector, ord=np.inf))')
    # Only the first, successful stationary branch changes. No reuse across a
    # callback, state update, transport rejection, grid change or pseudo-time.
    start = source.index('        if ok_ss:\n')
    end = source.index('        if bool(getattr(opts, "accept_residual_converged", False)):', start)
    branch = source[start:end]
    branch = replace_once(branch,
        '            _remember_continuation_linearization(\n',
        '            if exact_vector is None:\n'
        '                exact_vector = residual(x_ss, problem, force_exact_transport=True)\n'
        '            else:\n'
        '                shared_trace.append(dict(kind="exact_residual_shared", nodes=problem.n_points))\n'
        '            _remember_continuation_linearization(\n')
    branch = replace_once(branch,
        '                source_residual=residual(x_ss, problem, force_exact_transport=True),',
        '                source_residual=exact_vector,')
    source = source[:start]+branch+source[end:]
    ns = dict(vars(solver), shared_trace=records)
    exec(compile(source, '<exact-residual-handoff>', 'exec'), ns)
    with patch.object(solver, '_hybrid_newton', ns['_hybrid_newton']):
        yield
