"""Native analytic chemical and thermal derivatives at constant pressure.

All Y are independent (no normalization or eliminated species). The chain
rule includes mixture density, third bodies and Lindemann/Troe falloff.
The hybrid integration linearizes species chemistry only. The direct spatial
integration also uses thermal derivatives and retains frozen transport.
"""
import math
import numpy as np
from numba import njit, prange
from kflame.chemistry.mechanism import R_UNIV


@njit(cache=True)
def _product_partials(C, plan, derivative):
    """Polynomial limit at zero; signed continuation matches native kinetics."""
    derivative[:] = 0.0
    degree = plan[0]
    negative = 0
    product = 1.0
    for i in range(degree):
        c = C[plan[i + 1]]
        negative += c < 0.0
        product *= c
    if negative > 1:
        return 0.0
    for i in range(degree):
        value = 1.0
        for j in range(degree):
            if j != i:
                value *= C[plan[j + 1]]
        derivative[plan[i + 1]] += value
    return product


@njit(cache=True, parallel=True)
def species_partials_core(Ts, Ys, P, W, invW, lo, hi, tmid,
                          Ah, bh, Eh, Al, bl, El, three, fall, reversible,
                          troe, ta, t3, t1, t2, rp, pp,
                          ni, nn, nc, ei, ed, ec, dn, temperature_partials=False):
    ns, nodes = Ys.shape
    out = np.empty((ns, nodes, ns))
    out_t = np.zeros((ns, nodes))
    for node in prange(nodes):
        T = Ts[node]
        logT = math.log(T)
        s = 0.0
        for k in range(ns):
            s += Ys[k, node] * invW[k]
        rho = P / (R_UNIV * T * s)
        C = rho * Ys[:, node] * invW
        g = np.empty(ns)
        hrt = np.empty(ns)
        for k in range(ns):
            a = hi[k] if T > tmid[k] else lo[k]
            h = a[0] + T*(a[1]/2 + T*(a[2]/3 + T*(a[3]/4 + T*a[4]/5))) + a[5]/T
            entropy = a[0]*logT + T*(a[1] + T*(a[2]/2 + T*(a[3]/3 + T*a[4]/4))) + a[6]
            g[k] = h - entropy
            hrt[k] = h
        jc = np.zeros((ns, ns))
        jt = np.zeros(ns)
        df, dr = np.empty(ns), np.empty(ns)
        efficiency = np.empty(ns)
        for r in range(Ah.size):
            kf = Ah[r] * math.exp(bh[r]*logT - Eh[r]/(R_UNIV*T))
            dg = 0.0
            dh = 0.0
            for i in range(nc[r]):
                dg += nn[r, i] * g[ni[r, i]]
                dh += nn[r, i] * hrt[ni[r, i]]
            kc = math.exp(-min(500.0, max(-500.0, dg))) * (101325.0/(R_UNIV*T))**dn[r]
            kr = kf/max(kc, 1e-300) if reversible[r] else 0.0
            log_kf_t = bh[r]/T + Eh[r]/(R_UNIV*T*T)
            log_kc_t = ((dh if -500.0 < dg < 500.0 else 0.0)-dn[r])/T
            if kc <= 1e-300:
                log_kc_t = 0.0
            pf = _product_partials(C, rp[r], df)
            pr = _product_partials(C, pp[r], dr) if reversible[r] else 0.0
            if not reversible[r]:
                dr[:] = 0.0
            efficiency[:] = 1.0
            for i in range(ec[r]):
                efficiency[ei[r, i]] += ed[r, i]
            M = np.dot(efficiency, C)
            factor, factor_m = 1.0, 0.0
            factor_t = 0.0
            if three[r]:
                factor, factor_m = M, 1.0
            if fall[r]:
                k0 = Al[r] * math.exp(bl[r]*logT - El[r]/(R_UNIV*T))
                alpha = k0/max(kf, 1e-300)
                reduced = alpha*M
                lind = reduced/(1.0 + reduced)
                F, dF = 1.0, 0.0
                log_alpha_t = bl[r]/T + El[r]/(R_UNIV*T*T)-log_kf_t
                reduced_t = reduced*log_alpha_t
                F_t = 0.0
                if troe[r]:
                    fc = ((1-ta[r])*math.exp(-T/max(t3[r], 1e-300))
                          + ta[r]*math.exp(-T/max(t1[r], 1e-300))
                          + math.exp(-t2[r]/T))
                    L = math.log10(max(fc, 1e-300))
                    v = math.log10(max(reduced, 1e-300)) - .4 - .67*L
                    n = .75 - 1.27*L
                    den = n - .14*v
                    f = v/den
                    F = 10.0**(L/(1+f*f))
                    if reduced > 1e-300:
                        dF = F * L * (-2*f)/(1+f*f)**2 * n/(den*den) / reduced
                    if temperature_partials:
                        fc_t = (-(1-ta[r])*math.exp(-T/max(t3[r], 1e-300))/max(t3[r], 1e-300)
                                -ta[r]*math.exp(-T/max(t1[r], 1e-300))/max(t1[r], 1e-300)
                                +math.exp(-t2[r]/T)*t2[r]/(T*T))
                        L_t = fc_t/(fc*math.log(10.0)) if fc > 1e-300 else 0.0
                        v_t = (log_alpha_t/math.log(10.0) if reduced > 1e-300 else 0.0)-.67*L_t
                        den_t = -1.27*L_t-.14*v_t
                        f_t = (v_t*den-v*den_t)/(den*den)
                        F_t = F*math.log(10.0)*(L_t/(1+f*f)-L*2*f*f_t/(1+f*f)**2)
                factor = lind*F
                factor_m = alpha*(F/(1+reduced)**2 + lind*dF)
                factor_t = reduced_t*F/(1+reduced)**2 + lind*F_t
            q = kf*pf - kr*pr
            if temperature_partials:
                qt = factor*(kf*log_kf_t*pf-kr*(log_kf_t-log_kc_t)*pr)+factor_t*q
                for i in range(nc[r]):
                    jt[ni[r, i]] += nn[r, i]*qt
            for j in range(ns):
                dq = factor*(kf*df[j] - kr*dr[j]) + factor_m*efficiency[j]*q
                if dq != 0.0:
                    for i in range(nc[r]):
                        jc[ni[r, i], j] += nn[r, i]*dq
        for k in range(ns):
            contraction = np.dot(jc[k], C)
            if temperature_partials:
                out_t[k, node] = W[k]*(jt[k]-contraction/T)
            for j in range(ns):
                out[k, node, j] = W[k]*invW[j]*(rho*jc[k, j] - contraction/s)
    return out, out_t


def species_partials(backend, T, Y, *, _temperature=False):
    """Return d(omega_mass[k,node])/dY[j,node], shape (K,N,K)."""
    b, k = backend, backend.kinetics
    if np.any(k._sp_r_plan[:, 0] < 0) or np.any(k._sp_p_plan[:, 0] < 0):
        raise ValueError('Analytic chemistry requires elementary integer orders <= 3')
    Y = np.ascontiguousarray(Y, dtype=float)
    T = np.ascontiguousarray(T, dtype=float)
    if T.ndim != 1 or Y.shape != (b.n_species, T.size):
        raise ValueError('Expected T[N] and Y[K,N] for analytic chemistry')
    if not np.all(np.isfinite(Y)) or np.any(b.invW @ Y <= 1e-300):
        raise ValueError('Analytic chemistry requires finite, unclipped mixture density')
    if np.any(T <= 0) or not np.all(np.isfinite(T)):
        raise ValueError('Analytic chemistry requires positive finite temperatures')
    result = species_partials_core(T, Y, b._P_float, b.W, b.invW,
        b._nb_nasa_lo, b._nb_nasa_hi, b._nb_nasa_tmid,
        k._nb_A_hi, k._nb_b_hi, k._nb_Ea_hi, k._nb_A_lo, k._nb_b_lo, k._nb_Ea_lo,
        k._nb_is_three_body, k._nb_is_falloff, k._nb_is_reversible,
        k._nb_has_troe, k._nb_troe_A, k._nb_troe_T3, k._nb_troe_T1, k._nb_troe_T2,
        k._sp_r_plan, k._sp_p_plan, k._sp_net_idx, k._sp_net_nu, k._sp_net_count,
        k._sp_eff_idx, k._sp_eff_delta, k._sp_eff_count, k._nb_delta_nu, _temperature)
    return result if _temperature else result[0]


def thermochemical_partials(backend, T, Y):
    """Analytic nodal derivatives in [u,T,Y...] for direct block assembly."""
    jy, jt = species_partials(backend, T, Y, _temperature=True)
    ns, nodes = Y.shape
    nv = ns+2
    invW = backend.invW
    s = invW @ Y
    rho = backend._P_float/(R_UNIV*T*s)
    drho = np.zeros((nodes, nv))
    drho[:, 1] = -rho/T
    drho[:, 2:] = -rho[:, None]*invW[None, :]/s[:, None]
    dcp = np.zeros((nodes, nv))
    cp_molar = backend.thermo.partial_molar_cp(T)
    dcp[:, 2:] = (cp_molar*invW[:, None]).T
    a = np.where((T[None, :] > backend._nb_nasa_tmid[:, None])[:, :, None],
                 backend._nb_nasa_hi[:, None, :], backend._nb_nasa_lo[:, None, :])
    cp_t = R_UNIV*(a[:, :, 1]+T*(2*a[:, :, 2]+T*(3*a[:, :, 3]+4*T*a[:, :, 4])))
    dcp[:, 1] = np.sum(cp_t*Y*invW[:, None], axis=0)
    domega = np.zeros((nodes, ns, nv))
    domega[:, :, 1] = jt.T
    domega[:, :, 2:] = jy.transpose(1, 0, 2)
    return drho, dcp, domega, np.ascontiguousarray(cp_molar)


def hybrid_center_properties(backend, x, T_all, Y_all):
    """Linearize chemistry in species columns; retain exact perturbed thermo.

The native residual assembler applies the remaining spatial finite
differences with its usual frozen transport. No species is eliminated.
"""
    nodes, nv = x.shape
    ns = nv - 2
    T, Y = x[:, 1], x[:, 2:].T
    jac = species_partials(backend, T, Y)
    # Only base and temperature-perturbed nodes require reaction evaluation.
    ts = np.column_stack((T, T_all.reshape(nodes, nv)[:, 1])).ravel()
    ys = np.repeat(Y, 2, axis=1)
    om, h = np.empty_like(ys), np.empty_like(ys)
    _, cp2 = backend.eval_grid_thermo_kinetics_into(ts, ys, om, h)
    delta = (Y_all.reshape(ns, nodes, nv)[np.arange(ns), :, np.arange(ns)+2].T
             - x[:, 2:])
    omega = np.repeat(om[:, ::2, None], nv, axis=2)
    omega[:, :, 1] = om[:, 1::2]
    omega[:, :, 2:] += jac * delta[None, :, :]
    # Exact ideal-gas density and NASA cp in the perturbed states are cheap.
    rho = backend._P_float / (R_UNIV * T_all * (backend.invW @ Y_all))
    cp_species = backend.thermo.partial_molar_cp(T) * backend.invW[:, None]
    cp = np.repeat(cp2[::2, None], nv, axis=1)
    cp[:, 1] = cp2[1::2]
    cp[:, 2:] += cp_species.T * delta
    hk = np.repeat(h[:, ::2, None], nv, axis=2)
    hk[:, :, 1] = h[:, 1::2]
    return rho.reshape(nodes, nv), cp, omega, hk
