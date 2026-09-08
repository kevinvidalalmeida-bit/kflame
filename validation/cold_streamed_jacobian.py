"""Streaming FD assembly: consume each thermo batch directly into J blocks.

Unlike the earlier bounded-temporary experiment, this never concatenates all
perturbed thermo outputs back into a full n_species*n_nodes*n_variables tensor.
Only the final tridiagonal Jacobian itself remains global.
"""
import hashlib
import inspect
import textwrap
import equations
from cold_strategy_kernels import replace_once


def build_streamed_jacobian():
    original = equations._block_tridiag_jacobian_local
    source = textwrap.dedent(inspect.getsource(original))
    begin = source.index('    precomputed_thermo = None\n')
    end = source.index('        base = j * nv\n', begin)
    source = source[:begin] + '''    precomputed_thermo = None
    batch_first = 0
    batch_size = max(1, (2*1024*1024)//(8*nv*(n_sp+1)))
    for j in range(n_pts):
        if j % batch_size == 0 and bool(getattr(problem, "precompute_jacobian_thermo", False)):
            batch_first = j
            batch_last = min(n_pts, j+batch_size)
            local_cache = dict(cache, n_pts=batch_last-batch_first)
            # The common state is shared; only this batch's perturbations and
            # outputs are materialized, then consumed directly by J assembly.
            precomputed_thermo = _precompute_block_tridiag_center_thermo(
                x[batch_first*nv:batch_last*nv], problem, local_cache,
                rel_perturb, abs_perturb, eps)
''' + source[end:]
    source = replace_once(source,
        'center_thermo = (rho_b[j], cp_b[j], omega_b[:, j, :], hk_b[:, j, :])',
        'center_thermo = (rho_b[j-batch_first], cp_b[j-batch_first], '
        'omega_b[:, j-batch_first, :], hk_b[:, j-batch_first, :])')
    namespace = dict(vars(equations))
    exec(compile(source, '<streamed-jacobian-audit>', 'exec'), namespace)
    candidate = namespace[original.__name__]
    candidate.audit_sha256 = hashlib.sha256(source.encode()).hexdigest()
    return candidate
