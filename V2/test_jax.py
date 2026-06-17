import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "materiales"))

from mechanism_data import load_mechanism
from kinetics_jax import net_production_rates_jax
from kinetics_native import _net_production_rates_numba_core
from transport_jax import vmap_eval_faces_poly_jax
from transport_native import _eval_faces_poly_numba_core
mech = load_mechanism(r"C:\Users\lidia\AppData\Local\Programs\Python\Python311\Lib\site-packages\cantera\data\gri30.yaml")
n_sp = mech.n_species
n_rxn = mech.n_reactions

# Dummy state
T_val = 1500.0
C_val = np.ones(n_sp) * 1e-3
g_RT_val = np.random.randn(n_sp)

# We need to extract reaction info
A_hi = np.array([r.A for r in mech.reactions])
b_hi = np.array([r.b for r in mech.reactions])
Ea_hi = np.array([r.Ea for r in mech.reactions])

A_lo = np.array([r.A_low for r in mech.reactions])
b_lo = np.array([r.b_low for r in mech.reactions])
Ea_lo = np.array([r.Ea_low for r in mech.reactions])

is_three_body = np.array([r.rtype == "three-body" for r in mech.reactions])
is_falloff = np.array([r.rtype == "falloff" for r in mech.reactions])
is_reversible = np.array([r.reversible for r in mech.reactions])
has_troe = np.array([r.has_troe for r in mech.reactions])

troe_A = np.array([r.troe_A for r in mech.reactions])
troe_T3 = np.array([r.troe_T3 for r in mech.reactions])
troe_T1 = np.array([r.troe_T1 for r in mech.reactions])
troe_T2 = np.array([r.troe_T2 for r in mech.reactions])

delta_nu = np.sum(mech.nu_net, axis=0)

# 1. Evaluate Numba (which takes 2D arrays)
T_arr = np.array([T_val])
C_arr = C_val.reshape(n_sp, 1)
g_RT_arr = g_RT_val.reshape(n_sp, 1)

out_numba = _net_production_rates_numba_core(
    T_arr, C_arr, g_RT_arr, A_hi, b_hi, Ea_hi, A_lo, b_lo, Ea_lo,
    is_three_body, is_falloff, is_reversible, has_troe,
    troe_A, troe_T3, troe_T1, troe_T2, mech.nu_reactants, mech.nu_products, mech.nu_net, mech.efficiencies,
    delta_nu
)

# 2. Evaluate JAX
out_jax = net_production_rates_jax(
    T_val, C_val, g_RT_val, A_hi, b_hi, Ea_hi, A_lo, b_lo, Ea_lo,
    is_three_body, is_falloff, is_reversible, has_troe,
    troe_A, troe_T3, troe_T1, troe_T2, mech.nu_reactants, mech.nu_products, mech.nu_net, mech.efficiencies,
    delta_nu
)

print("Numba output shape:", out_numba.shape)
print("JAX output shape:", out_jax.shape)
diff_kin = np.max(np.abs(out_numba[:, 0] - out_jax))
print("Kinetics Max absolute difference:", diff_kin)

# --- Test Transport ---
print("\n--- Testing Transport ---")
from transport_native import NativeTransport
trans = NativeTransport(mech, xp=np)
cond_poly = trans._cond_poly
diff_poly = trans._diff_poly

# 1. Evaluate Numba Transport
P = mech.pressure
rho_numba, Dm_numba, lam_numba, Wmix_numba = _eval_faces_poly_numba_core(
    T_arr, C_arr, P, mech.inv_molecular_weights, mech.molecular_weights, cond_poly, diff_poly
)

# 2. Evaluate JAX Transport
rho_jax, Dm_jax, lam_jax, Wmix_jax, sumd_jax = vmap_eval_faces_poly_jax(
    T_arr, C_arr.T, P, mech.inv_molecular_weights, mech.molecular_weights, cond_poly, diff_poly
)

diff_rho = np.max(np.abs(rho_numba - rho_jax))
diff_lam = np.max(np.abs(lam_numba - lam_jax))
diff_Dm = np.max(np.abs(Dm_numba[:, 0] - Dm_jax[0, :]))

print("Transport Max absolute diff rho:", diff_rho)
print("Transport Max absolute diff lam:", diff_lam)
print("Transport Max absolute diff Dm:", diff_Dm)

if diff_Dm > 1e-5:
    idx = np.argmax(np.abs(Dm_numba[:, 0] - Dm_jax[0, :]))
    print(f"Worst Dm at idx {idx}: Numba={Dm_numba[idx, 0]}, JAX={Dm_jax[0, idx]}")
    print(f"sumd_jax[{idx}] = {sumd_jax[0, idx]}")
    
    # Numba values
    inv_wmix = np.sum(C_arr[:, 0] * mech.inv_molecular_weights)
    wm_num = 1.0 / inv_wmix
    print(f"wm_num = {wm_num}, wm_jax = {Wmix_jax[0]}")
