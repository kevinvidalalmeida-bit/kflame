"""Isolated candidates; never imported by production code."""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import inspect
import numpy as np
from numba import njit
import kflame.flame.equations as equations
import kflame.flame.solver as solver
import kflame.chemistry.backend as backend

ORIGINAL_FACTORIZE = equations.factorize
ORIGINAL_HYBRID = solver._hybrid_newton
ORIGINAL_CORE = backend._eval_thermo_kinetics_sparse_numba_core
SERIAL_CORE = njit(cache=False)(ORIGINAL_CORE.py_func)


def backward_euler_hybrid():
    """Cantera-style converged transient subproblem; Jacobian age unchanged."""
    source = inspect.getsource(ORIGINAL_HYBRID)
    old = '            use_ptc = successive_failures == 0'
    if source.count(old) != 1:
        raise RuntimeError('Hybrid source changed; review BE experiment')
    source = source.replace(old, '            use_ptc = False')
    namespace = dict(vars(solver))
    exec(compile(source, '<isolated backward Euler experiment>', 'exec'), namespace)
    return namespace['_hybrid_newton']


def carry_newton(retry):
    """Build the isolated experiment from current Newton; fail on source drift.

    No experiment flags or branches are installed in the production solver.
    A retained residual is valid only at unchanged x, rdt and x_old and when
    rebuilding the Jacobian cannot change frozen multicomponent transport.
    """
    source = inspect.getsource(solver.newton_solve)
    def replace(old, new, count=1):
        nonlocal source
        if source.count(old) != count:
            raise RuntimeError('Newton source changed; review carry experiment')
        source = source.replace(old, new)
    replace('    f_carry: np.ndarray | None = None\n',
            '    f_carry: np.ndarray | None = None\n'
            '    reuse = (bool(getattr(problem, "analytic_spatial", False))\n'
            '             and not bool(getattr(problem, "lag_multicomponent_transport", False)))\n')
    replace('                force_new_jac = False\n                f_carry = None\n',
            '                force_new_jac = False\n'
            '                if not reuse:\n                    f_carry = None\n')
    if retry:
        replace('            force_new_jac = True\n            continue\n',
                '            force_new_jac = True\n'
                '            if reuse:\n                f_carry = f\n            continue\n')
        replace('                force_new_jac = True\n                if verbose:\n',
                '                force_new_jac = True\n'
                '                if reuse:\n                    f_carry = f\n                if verbose:\n', 2)
    namespace = dict(vars(solver))
    exec(compile(source, '<isolated residual carry experiment>', 'exec'), namespace)
    return namespace['newton_solve']


def workspace_factorize(jmat, order='C'):
    if not isinstance(jmat, equations.BlockTridiagJacobian):
        return ORIGINAL_FACTORIZE(jmat)
    problem = getattr(jmat, '_profile_problem', None)
    tic = equations._profile_start(problem)
    n, nv = jmat.n_blocks, jmat.block_size
    factors_array = np.empty((n,nv,nv))
    pivots_array = np.empty((n,nv), dtype=np.int32)
    cprime = np.empty_like(jmat.upper)
    mat = np.empty((nv,nv), order=order)
    product = np.empty((nv,nv))
    lu_blocks = []
    for i in range(n):
        mat[:] = jmat.diag[i]
        if i:
            np.matmul(jmat.lower[i-1], cprime[i-1], out=product)
            mat -= product
        factors,pivots,info = equations._block_getrf(mat, overwrite_a=True)
        if info:
            raise RuntimeError(f'dgetrf failed at block {i}: {info}')
        factors_array[i],pivots_array[i] = factors,pivots
        # Own the factors even if GETRF reused the single Fortran workspace.
        lu_blocks.append((factors_array[i],pivots_array[i]))
        if i<n-1:
            solved,info = equations._block_getrs(factors,pivots,jmat.upper[i])
            if info:
                raise RuntimeError(f'dgetrs failed at block {i}: {info}')
            cprime[i] = solved
    result = dict(method='block_tridiag',solver='block_thomas',lu_blocks=lu_blocks,
        lu_blocks_array=factors_array,pivots_array=pivots_array,
        lower_blocks=np.ascontiguousarray(jmat.lower),cprime=cprime,n_blocks=n,block_size=nv,
        compiled_substitution=bool(getattr(jmat,'use_compiled_substitution',True)))
    equations._profile_record(problem,'linear_factorize',tic)
    return result


@contextmanager
def variant(name):
    with ExitStack() as stack:
        if name in ('defect_corrections', 'scaled_ser', 'defect_step'):
            from global_nonlinear_candidates import build_candidate
            newton, hybrid = build_candidate(name)
            stack.enter_context(patch.object(solver, 'newton_solve', newton))
            stack.enter_context(patch.object(solver, '_hybrid_newton', hybrid))
        if name == 'backward_euler':
            stack.enter_context(patch.object(solver, '_hybrid_newton', backward_euler_hybrid()))
        if name in ('workspace_c','workspace_f','combined'):
            order = 'F' if name == 'workspace_f' else 'C'
            fn = lambda j: workspace_factorize(j,order)
            stack.enter_context(patch.object(equations,'factorize',fn))
            stack.enter_context(patch.object(solver,'factorize',fn))
        if name in ('carry','retry','combined'):
            stack.enter_context(patch.object(solver,'newton_solve',carry_newton(name != 'carry')))
        if name == 'small_serial':
            def core(*args):
                return (SERIAL_CORE if args[0].size <= 16 else ORIGINAL_CORE)(*args)
            stack.enter_context(patch.object(backend,'_eval_thermo_kinetics_sparse_numba_core',core))
        yield
