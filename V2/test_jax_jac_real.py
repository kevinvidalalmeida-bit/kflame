import jax
import jax.numpy as jnp
import time
import numpy as np
from jax import config
config.update("jax_enable_x64", True)

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "materiales"))

from mechanism_data import load_mechanism
from kinetics_jax import net_production_rates_jax
from transport_jax import eval_faces_poly_jax

mech = load_mechanism(r"C:\Users\lidia\AppData\Local\Programs\Python\Python311\Lib\site-packages\cantera\data\gri30.yaml")
n_sp = mech.n_species
n_rxn = mech.n_reactions

# Static inputs
A_hi = jnp.array([r.A for r in mech.reactions])
b_hi = jnp.array([r.b for r in mech.reactions])
Ea_hi = jnp.array([r.Ea for r in mech.reactions])

A_lo = jnp.array([r.A_low for r in mech.reactions])
b_lo = jnp.array([r.b_low for r in mech.reactions])
Ea_lo = jnp.array([r.Ea_low for r in mech.reactions])

is_three_body = jnp.array([r.rtype == "three-body" for r in mech.reactions])
is_falloff = jnp.array([r.rtype == "falloff" for r in mech.reactions])
is_reversible = jnp.array([r.reversible for r in mech.reactions])
has_troe = jnp.array([r.has_troe for r in mech.reactions])

troe_A = jnp.array([r.troe_A for r in mech.reactions])
troe_T3 = jnp.array([r.troe_T3 for r in mech.reactions])
troe_T1 = jnp.array([r.troe_T1 for r in mech.reactions])
troe_T2 = jnp.array([r.troe_T2 for r in mech.reactions])

delta_nu = jnp.sum(mech.nu_net, axis=0)
nu_r = jnp.array(mech.nu_reactants)
nu_p = jnp.array(mech.nu_products)
nu_net = jnp.array(mech.nu_net)
eff = jnp.array(mech.efficiencies)

@jax.jit
def single_node_kinetics(Y_T):
    # Y_T is (n_sp + 1,), last is T
    Y = Y_T[:-1]
    T = Y_T[-1]
    
    # fake g_RT
    g_RT = jnp.zeros(n_sp)
    
    omega = net_production_rates_jax(
        T, Y, g_RT, A_hi, b_hi, Ea_hi, A_lo, b_lo, Ea_lo,
        is_three_body, is_falloff, is_reversible, has_troe,
        troe_A, troe_T3, troe_T1, troe_T2,
        nu_r, nu_p, nu_net, eff, delta_nu
    )
    return omega

jac_func = jax.jit(jax.jacfwd(single_node_kinetics))

y_t = jnp.ones(n_sp + 1) * 1e-3
y_t = y_t.at[-1].set(1500.0)

print("Compiling real kinetics jacobian...")
t0 = time.time()
J = jac_func(y_t)
print(f"Compilation took {time.time() - t0:.2f}s")
print(f"J shape: {J.shape}")

t0 = time.time()
for _ in range(100):
    J = jac_func(y_t)
J.block_until_ready()
print(f"100 evaluations took {time.time() - t0:.4f}s")
