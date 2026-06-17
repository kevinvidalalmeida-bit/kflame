import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "materiales"))

from mechanism_data import load_mechanism
from thermo_native import NativeThermo

mech = load_mechanism(r"C:\Users\lidia\AppData\Local\Programs\Python\Python311\Lib\site-packages\cantera\data\gri30.yaml")

thermo = NativeThermo(mech, xp=jnp)

@jax.jit
def get_cp_mass_and_hk(T, Y):
    return thermo.cp_R(T), thermo.cp_mass(T, Y)

T = 1500.0
Y = jnp.ones(mech.n_species) / mech.n_species

cp_R_val, cp_mass_val = get_cp_mass_and_hk(T, Y)
print("cp_mass:", cp_mass_val)
print("JAX thermo compilation successful!")
