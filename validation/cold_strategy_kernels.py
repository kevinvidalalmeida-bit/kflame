"""Isolated cold-start experiments. No experimental production switches.

The candidate source is derived from the current kernel and hashed. Replacements
fail closed when the reference changes. Every call includes cache construction;
no thermal data survive changes in state, pressure, or mechanism.
"""
import hashlib
import inspect
import textwrap

import numpy as np
from numba import njit
import species_backend_native as backend_module


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Reference kernel changed at: ' + old[:80])
    return source.replace(old, new)


def compile_kernel(source, namespace, parallel=True):
    namespace = dict(namespace)
    exec(compile(source, '<cold-strategy-audit>', 'exec'), namespace)
    name = source.split('def ', 1)[1].split('(', 1)[0]
    kernel = njit(parallel=parallel)(namespace[name])
    kernel.audit_sha256 = hashlib.sha256(source.encode()).hexdigest()
    return kernel


def build_candidates():
    reference = backend_module._eval_thermo_kinetics_sparse_numba_core
    source = textwrap.dedent(inspect.getsource(reference.py_func))
    source = source[source.index('def '):]
    namespace = vars(backend_module)
    names = list(inspect.signature(reference.py_func).parameters)
    positions = {name: i for i, name in enumerate(names)}
    serial = compile_kernel(source.replace('prange(n_pts)', 'range(n_pts)'), namespace, False)

    # Cache the existing exact NASA and Arrhenius expressions, preserving
    # arithmetic order. Concentrations/third bodies/falloff remain live below.
    nasa_begin = source.index('            c = nasa_hi[k]')
    nasa_end = source.index('            cp_mix +=', nasa_begin)
    nasa = source[nasa_begin:nasa_end]
    nasa = nasa.replace('hk_out[k, m]', 'h_table[k, m]')
    nasa += '            cp_table[k, m] = cp_R\n'
    rate_begin = source.index('            kf = A_hi[r]')
    rate_end = source.index('            kr = 0.0', rate_begin)
    rate = source[rate_begin:rate_end]
    rate += '            kf_table[r, m] = kf\n            kc_table[r, m] = Kc\n'
    f_begin = source.index('                    t3 = max(troe_T3[r]')
    f_end = source.index('                    logPr =', f_begin)
    fcent = source[f_begin:f_end]
    prep_source = '''def thermal_factors(T_arr, nasa_lo, nasa_hi, nasa_tmid, W,
    A_hi, b_hi, Ea_hi, A_lo, b_lo, Ea_lo, is_falloff, has_troe,
    troe_A, troe_T3, troe_T1, troe_T2, net_idx, net_nu, net_count, delta_nu):
    n_sp, n_rxn, n_pts = W.size, A_hi.size, T_arr.size
    cp_table = np.empty((n_sp, n_pts))
    h_table = np.empty((n_sp, n_pts))
    kf_table = np.empty((n_rxn, n_pts))
    kc_table = np.empty((n_rxn, n_pts))
    k0_table = np.zeros((n_rxn, n_pts))
    fc_table = np.zeros((n_rxn, n_pts))
    for m in prange(n_pts):
        T = T_arr[m]
        logT = math.log(T)
        inv_RT = 1.0 / (R_UNIV * T)
        g_RT = np.empty(n_sp)
        for k in range(n_sp):
''' + nasa + '''        c_factor = 101325.0 / (R_UNIV * T)
        for r in range(n_rxn):
''' + rate + '''            if is_falloff[r]:
                k0_table[r, m] = A_lo[r] * math.exp(b_lo[r] * logT - Ea_lo[r] * inv_RT)
            if has_troe[r]:
''' + textwrap.indent(textwrap.dedent(fcent), '                ') + '''                fc_table[r, m] = logFcent
    return cp_table, h_table, kf_table, kc_table, k0_table, fc_table
'''
    prep = compile_kernel(prep_source, namespace)
    prep_names = list(inspect.signature(prep.py_func).parameters)[1:]
    candidate = replace_once(source, 'delta_nu, rho_out, cp_out, omega_out, hk_out,',
                             'delta_nu, rho_out, cp_out, omega_out, hk_out,\n'
                             '    thermal_index, cp_table, h_table, kf_table, kc_table, k0_table, fc_table,')
    candidate = replace_once(candidate, '        T = T_arr[m]',
                             '        T = T_arr[m]\n        ti = thermal_index[m]')
    original_nasa = source[nasa_begin:nasa_end]
    candidate = replace_once(candidate, original_nasa,
                             '            cp_R = cp_table[k, ti]\n            hk_out[k, m] = h_table[k, ti]\n')
    candidate = replace_once(candidate, source[rate_begin:rate_end],
                             '            kf = kf_table[r, ti]\n            Kc = kc_table[r, ti]\n')
    candidate = replace_once(candidate, '                k0 = A_lo[r] * math.exp(b_lo[r] * logT - Ea_lo[r] * inv_RT)',
                             '                k0 = k0_table[r, ti]')
    candidate = replace_once(candidate, fcent, '                    logFcent = fc_table[r, ti]\n')
    thermal_core = compile_kernel(candidate, namespace)

    def thermal(*args):
        temperatures, inverse = np.unique(args[0], return_inverse=True)
        factors = prep(temperatures, *(args[positions[key]] for key in prep_names))
        return thermal_core(*args, inverse, *factors)

    thermal.audit_sha256 = hashlib.sha256((prep_source + candidate).encode()).hexdigest()

    def indexed_thermal(temperatures, inverse, *args):
        factors = prep(temperatures, *(args[positions[key]] for key in prep_names))
        return thermal_core(*args, inverse, *factors)

    thermal.indexed = indexed_thermal

    power_source = replace_once(source, 'delta_nu, rho_out, cp_out, omega_out, hk_out,',
                                'delta_nu, rho_out, cp_out, omega_out, hk_out, powers, power_index,')
    power_source = replace_once(power_source, '        c_factor = 101325.0 / (R_UNIV * T)',
                                '        c_factor = 101325.0 / (R_UNIV * T)\n'
                                '        power_values = np.empty(powers.size)\n'
                                '        for di in range(powers.size):\n'
                                '            power_values[di] = c_factor ** powers[di]')
    power_source = replace_once(power_source, '(c_factor ** delta_nu[r])', 'power_values[power_index[r]]')
    power_core = compile_kernel(power_source, namespace)

    def powers(*args):
        # Included in timing; do not key persistent data only by mechanism size.
        unique, inverse = np.unique(args[positions['delta_nu']], return_inverse=True)
        return power_core(*args, unique, inverse)

    powers.audit_sha256 = power_core.audit_sha256
    return dict(baseline=reference, thermal=thermal, powers=powers, serial=serial)
