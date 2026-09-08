"""Isolated species-major NASA SIMD experiment, not thermal-value reuse.

Every state is evaluated on every call. No unique-temperature cache, no
fastmath, no reaction reordering. Additional arrays and calls count in timing.
"""
import hashlib
import inspect
import re
import textwrap

from numba import njit
import species_backend_native as backend
from cold_strategy_kernels import replace_once


def build_simd():
    source = textwrap.dedent(inspect.getsource(backend._eval_thermo_kinetics_sparse_numba_core.py_func))
    source = source[source.index('def '):]
    start = source.index('            c = nasa_hi[k]')
    stop = source.index('            cp_mix +=', start)
    nasa = source[start:stop].replace('g_RT[k]', 'g_table[k, m]')
    prep = '''def nasa_species_major(T_arr, nasa_lo, nasa_hi, nasa_tmid, hk_out):
    n_sp, n_pts = nasa_tmid.size, T_arr.size
    cp_table = np.empty((n_sp, n_pts))
    g_table = np.empty((n_sp, n_pts))
    logs = np.empty(n_pts)
    for m in prange(n_pts):
        logs[m] = math.log(T_arr[m])
    for k in prange(n_sp):
        for m in range(n_pts):
            T = T_arr[m]
            logT = logs[m]
''' + nasa + '''            cp_table[k, m] = cp_R
    return cp_table, g_table
'''
    namespace = dict(vars(backend))
    exec(compile(prep, '<NASA-SIMD>', 'exec'), namespace)
    nasa_kernel = njit(parallel=True, fastmath=False)(namespace['nasa_species_major'])
    candidate = replace_once(source, '    for m in prange(n_pts):',
        '    cp_table, g_table = nasa_species_major(T_arr, nasa_lo, nasa_hi, nasa_tmid, hk_out)\n'
        '    for m in prange(n_pts):')
    candidate = replace_once(candidate, source[start:stop],
        '            cp_R = cp_table[k, m]\n            g_RT[k] = g_table[k, m]\n')
    namespace['nasa_species_major'] = nasa_kernel
    exec(compile(candidate, '<chemistry-with-SIMD-NASA>', 'exec'), namespace)
    kernel = njit(parallel=True, fastmath=False)(namespace['_eval_thermo_kinetics_sparse_numba_core'])
    kernel.audit_sha256 = hashlib.sha256((prep + candidate).encode()).hexdigest()
    kernel.nasa_kernel = nasa_kernel
    kernel.audit_source = prep + '\n' + candidate
    return kernel


def vector_evidence(kernel):
    """Packed LLVM operations, not merely a parallel=True claim."""
    ir = '\n'.join(kernel.nasa_kernel.inspect_llvm().values())
    operations = re.findall(r'\b(?:fadd|fmul|fdiv|fsub)\b[^\n]*<\d+ x double>[^\n]*', ir)
    asm = '\n'.join(kernel.nasa_kernel.inspect_asm().values())
    packed_asm = re.findall(r'\b(?:v?addpd|v?mulpd|v?divpd|v?subpd)\b[^\n]*', asm)
    return dict(signatures=[str(s) for s in kernel.nasa_kernel.signatures],
                packed_fp64_operations=len(operations), examples=operations[:8],
                packed_assembly_instructions=len(packed_asm), assembly_examples=packed_asm[:8],
                llvm_sha256=hashlib.sha256(ir.encode()).hexdigest())
