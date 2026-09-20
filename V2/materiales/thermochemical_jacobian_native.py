"""Exact temperature reuse for native finite-difference Jacobian blocks.

The caller validates the node-major [u, T, Y...] layout. Only the temperature
column changes T, so NASA and thermal rate factors are evaluated twice per
node. Density, concentrations, third bodies and pressure falloff are still
recomputed for every column. No values are cached across nonlinear states.

Keep arithmetic aligned with species_backend_native._eval_thermo_kinetics_sparse_numba_core;
test_native_thermal_reuse.py checks both paths, including signed Newton trials.
"""
import math
import numpy as np
from numba import njit, prange
from mechanism_data import R_UNIV
from kinetics_native import _mass_action_product


@njit(cache=True, parallel=True)
def grouped_core(
    T_arr, Y, P, nasa_lo, nasa_hi, nasa_tmid, W, invW,
    A_hi, b_hi, Ea_hi, A_lo, b_lo, Ea_lo,
    is_three_body, is_falloff, is_reversible, has_troe,
    troe_A, troe_T3, troe_T1, troe_T2,
    r_idx, r_nu, r_count, p_idx, p_nu, p_count, r_plan, p_plan,
    net_idx, net_nu, net_count, eff_idx, eff_delta, eff_count,
    delta_nu, rho_out, cp_out, omega_out, hk_out,
):
    n_sp = Y.shape[0]
    nv = n_sp + 2
    n_nodes = Y.shape[1] // nv
    n_rxn = A_hi.shape[0]
    for node in prange(n_nodes):
        cp_species = np.empty((2, n_sp))
        h_species = np.empty((2, n_sp))
        g_RT = np.empty(n_sp)
        forward = np.empty((2, n_rxn))
        reverse = np.empty((2, n_rxn))
        equilibrium = np.empty((2, n_rxn))
        low_pressure = np.empty((2, n_rxn))
        log_center = np.empty((2, n_rxn))
        for it in range(2):
            T = T_arr[node * nv + it]
            inv_RT = 1.0 / (R_UNIV * T)
            logT = math.log(T)
            for k in range(n_sp):
                c = nasa_hi[k] if T > nasa_tmid[k] else nasa_lo[k]
                cp_R = c[0] + T * (c[1] + T * (c[2] + T * (c[3] + T * c[4])))
                h_RT = (c[0] + T * (c[1] / 2.0 + T * (c[2] / 3.0
                        + T * (c[3] / 4.0 + T * c[4] / 5.0))) + c[5] / T)
                s_R = (c[0] * logT + T * (c[1] + T * (c[2] / 2.0
                       + T * (c[3] / 3.0 + T * c[4] / 4.0))) + c[6])
                cp_species[it, k] = cp_R
                h_species[it, k] = h_RT * R_UNIV * T
                g_RT[k] = h_RT - s_R
            c_factor = 101325.0 / (R_UNIV * T)
            for r in range(n_rxn):
                kf = A_hi[r] * math.exp(b_hi[r] * logT - Ea_hi[r] * inv_RT)
                delta_g = 0.0
                for ii in range(net_count[r]):
                    delta_g += net_nu[r, ii] * g_RT[net_idx[r, ii]]
                if delta_g > 500.0:
                    delta_g = 500.0
                elif delta_g < -500.0:
                    delta_g = -500.0
                Kc = math.exp(-delta_g) * (c_factor ** delta_nu[r])
                kr = kf / max(Kc, 1.0e-300) if is_reversible[r] else 0.0
                forward[it, r], reverse[it, r], equilibrium[it, r] = kf, kr, Kc
                low_pressure[it, r] = 0.0
                log_center[it, r] = 0.0
                if is_falloff[r]:
                    low_pressure[it, r] = A_lo[r] * math.exp(b_lo[r] * logT - Ea_lo[r] * inv_RT)
                    if has_troe[r]:
                        t3 = max(troe_T3[r], 1.0e-300)
                        t1 = max(troe_T1[r], 1.0e-300)
                        Fcent = ((1.0 - troe_A[r]) * math.exp(-T / t3)
                                 + troe_A[r] * math.exp(-T / t1)
                                 + math.exp(-troe_T2[r] / max(T, 1.0e-300)))
                        log_center[it, r] = math.log10(max(Fcent, 1.0e-300))

        # Reuse only the temperature-dependent quantities above. The inner
        # ordering and arithmetic match the production fused kernel.
        logC = np.empty(n_sp)
        concentrations = np.empty(n_sp)
        for column in range(nv):
            m = node * nv + column
            it = 1 if column == 1 else 0
            T = T_arr[m]
            inv_wmix = 0.0
            for k in range(n_sp):
                inv_wmix += Y[k, m] * invW[k]
            if inv_wmix < 1.0e-300:
                inv_wmix = 1.0e-300
            Wmix = 1.0 / inv_wmix
            rho = P * Wmix / (R_UNIV * T)
            rho_out[m] = rho
            cp_mix, c_total = 0.0, 0.0
            has_negative = False
            for k in range(n_sp):
                hk_out[k, m] = h_species[it, k]
                cp_mix += Y[k, m] * cp_species[it, k] * R_UNIV * invW[k]
                conc = rho * Y[k, m] * invW[k]
                concentrations[k] = conc
                c_total += conc
                has_negative = has_negative or conc < 0.0
                conc = abs(conc)
                if conc < 1.0e-300:
                    conc = 1.0e-300
                logC[k] = math.log(conc)
                omega_out[k, m] = 0.0
            cp_out[m] = cp_mix
            for r in range(n_rxn):
                kf, kr, Kc = forward[it, r], reverse[it, r], equilibrium[it, r]
                Rf = _mass_action_product(concentrations, logC, r_idx[r], r_nu[r],
                                          r_count[r], r_plan[r], has_negative) * kf
                Rr = 0.0
                if is_reversible[r]:
                    Rr = _mass_action_product(concentrations, logC, p_idx[r], p_nu[r],
                                              p_count[r], p_plan[r], has_negative) * kr
                M = c_total
                if is_three_body[r] or is_falloff[r]:
                    for ii in range(eff_count[r]):
                        k = eff_idx[r, ii]
                        ck = rho * Y[k, m] * invW[k]
                        M += eff_delta[r, ii] * ck
                if is_three_body[r]:
                    Rf *= M
                    Rr *= M
                if is_falloff[r]:
                    k0 = low_pressure[it, r]
                    Pr = k0 * M / max(kf, 1.0e-300)
                    F_lind = Pr / (1.0 + Pr)
                    F = 1.0
                    if has_troe[r]:
                        logFcent = log_center[it, r]
                        logPr = math.log10(max(Pr, 1.0e-300))
                        c_troe = -0.4 - 0.67 * logFcent
                        n_troe = 0.75 - 1.27 * logFcent
                        d_troe = 0.14
                        f1 = (logPr + c_troe) / (n_troe - d_troe * (logPr + c_troe))
                        F = 10.0 ** (logFcent / (1.0 + f1 * f1))
                    kf_falloff = kf * F_lind * F
                    Rf = kf_falloff * (Rf / max(kf, 1.0e-300))
                    if is_reversible[r]:
                        kr_falloff = kf_falloff / max(Kc, 1.0e-300)
                        Rr = kr_falloff * (Rr / max(kr, 1.0e-300))
                q = Rf - Rr
                for ii in range(net_count[r]):
                    k = net_idx[r, ii]
                    omega_out[k, m] += net_nu[r, ii] * q * W[k]
