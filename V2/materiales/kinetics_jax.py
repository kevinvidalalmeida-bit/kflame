from __future__ import annotations
import jax
import jax.numpy as jnp
from mechanism_data import R_UNIV

@jax.jit
def net_production_rates_jax(
    T, C, g_RT, A_hi, b_hi, Ea_hi, A_lo, b_lo, Ea_lo,
    is_three_body, is_falloff, is_reversible, has_troe,
    troe_A, troe_T3, troe_T1, troe_T2,
    nu_r, nu_p, nu_net, eff, delta_nu
):
    """
    Evaluates net production rates for a single grid point using JAX.
    T: scalar
    C: (n_sp,)
    g_RT: (n_sp,)
    """
    # Safe logarithm
    C_safe = jnp.maximum(C, 1e-300)
    logC = jnp.log(C_safe)
    
    T_safe = jnp.maximum(T, 1e-300)
    logT = jnp.log(T_safe)
    c_factor = 101325.0 / (R_UNIV * T_safe)
    inv_RT = 1.0 / (R_UNIV * T_safe)
    
    # Forward rate constants (High pressure)
    kf = A_hi * jnp.exp(b_hi * logT - Ea_hi * inv_RT)
    
    # Equilibrium constants
    delta_g = jnp.dot(g_RT, nu_net) # (n_rxn,)
    delta_g = jnp.clip(delta_g, -500.0, 500.0)
    Kc = jnp.exp(-delta_g) * (c_factor ** delta_nu)
    Kc_safe = jnp.maximum(Kc, 1e-300)
    
    # Reverse rate constants
    kr = jnp.where(is_reversible, kf / Kc_safe, 0.0)
    
    # Reaction rates
    rf_exp = jnp.dot(logC, nu_r) # (n_rxn,)
    rr_exp = jnp.dot(logC, nu_p) # (n_rxn,)
    
    Rf = jnp.exp(rf_exp) * kf
    Rr = jnp.where(is_reversible, jnp.exp(rr_exp) * kr, 0.0)
    
    # Third body efficiencies
    M = jnp.dot(eff, C_safe) # (n_rxn,)
    c_total = jnp.sum(C_safe)
    
    Rf = jnp.where(is_three_body, Rf * M, Rf)
    Rr = jnp.where(is_three_body, Rr * M, Rr)
    
    # Falloff adjustments
    k0 = A_lo * jnp.exp(b_lo * logT - Ea_lo * inv_RT)
    kf_safe = jnp.maximum(kf, 1e-300)
    Pr = k0 * M / kf_safe
    F_lind = Pr / (1.0 + Pr)
    
    # Troe formulation
    troe_T3_safe = jnp.maximum(troe_T3, 1e-300)
    troe_T1_safe = jnp.maximum(troe_T1, 1e-300)
    
    Fcent = (
        (1.0 - troe_A) * jnp.exp(-T_safe / troe_T3_safe)
        + troe_A * jnp.exp(-T_safe / troe_T1_safe)
        + jnp.exp(-troe_T2 / T_safe)
    )
    Fcent = jnp.maximum(Fcent, 1e-300)
    logFcent = jnp.log10(Fcent)
    logPr = jnp.log10(jnp.maximum(Pr, 1e-300))
    
    c_troe = -0.4 - 0.67 * logFcent
    n_troe = 0.75 - 1.27 * logFcent
    d_troe = 0.14
    f1 = (logPr + c_troe) / (n_troe - d_troe * (logPr + c_troe))
    F_troe = 10.0 ** (logFcent / (1.0 + f1 * f1))
    
    F_factor = jnp.where(has_troe, F_troe, 1.0)
    kf_falloff = kf * F_lind * F_factor
    
    kr_safe = jnp.maximum(kr, 1e-300)
    
    Rf_falloff = kf_falloff * (Rf / kf_safe)
    kr_falloff = jnp.where(is_reversible, kf_falloff / Kc_safe, 0.0)
    Rr_falloff = jnp.where(is_reversible, kr_falloff * (Rr / kr_safe), 0.0)
    
    Rf = jnp.where(is_falloff, Rf_falloff, Rf)
    Rr = jnp.where(is_falloff, Rr_falloff, Rr)
    
    q = Rf - Rr # (n_rxn,)
    omega = jnp.dot(nu_net, q) # (n_sp,)
    return omega

# Vectorized versions for batch processing
vmap_net_production_rates_jax = jax.vmap(
    net_production_rates_jax,
    in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None),
    out_axes=0
)
