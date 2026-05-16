import numpy as np
from config import FlameCase
from problem import FreeFlameProblem
from state import pack_species
from species_backend import SpeciesBackend
from residual_species import residual_species_frozen

case = FlameCase(
    mech="gri30.yaml", fuel="CH4", oxidizer="O2:1.0, N2:3.76",
    phi=1.0, T_in=300.0, P=101325.0, width=0.03,
)
data = np.load("outputs/reference/freeflame_ref.npz")
z_ref = data["z"]
print("Puntos exactos de Cantera:", len(z_ref))

problem = FreeFlameProblem(case, n_points=len(z_ref), locs=z_ref)
problem.set_frozen_profiles_from_reference("outputs/reference/freeflame_ref.npz")
problem.backend = SpeciesBackend(problem)

Y_ref = problem.Y_ref.copy()
xY_ref = pack_species(Y_ref[:-1, :])
F = residual_species_frozen(xY_ref, problem)

Fm = F.reshape((problem.n_ind_species, problem.n_points), order="C")
worst_j = np.argmax(np.max(np.abs(Fm), axis=0))
worst_val = np.max(np.abs(Fm[:, worst_j]))
worst_k = np.argmax(np.abs(Fm[:, worst_j]))
sp_name = problem.species_names[worst_k]

print(f"Norma F en malla exacta: {np.linalg.norm(F, ord=np.inf):.4e}")
print(f"Peor nodo: j={worst_j}, Especie={sp_name}, Valor F={worst_val:.4e}")

# Ahora calcularemos a mano los terminos para worst_j y worst_k
T = problem.T_fixed
Y = problem.Y_ref
z = problem.z
mdot = problem.mdot_fixed
backend = problem.backend

# Eval prop en j, j-1, j+1 para ver derivadas
rj, Dj, oj = backend.eval_node_species_only(T[worst_j], Y[:, worst_j])
print(f"rho={rj:.4e}, D={Dj[worst_k]:.4e}, omega={oj[worst_k]:.4e}")

# Reconstruir terminos
z = problem.z
mdot = problem.mdot_fixed
dz_up = z[worst_j] - z[worst_j-1]
conv = mdot * (Y[worst_k, worst_j] - Y[worst_k, worst_j-1]) / dz_up

# Nodal properties
rm_node, Dm_node, _ = backend.eval_node_species_only(T[worst_j-1], Y[:, worst_j-1])
rp_node, Dp_node, _ = backend.eval_node_species_only(T[worst_j+1], Y[:, worst_j+1])

# Flux j-1/2
rm_face = 0.5 * (rm_node + rj)
Dm_face = 0.5 * (Dm_node + Dj)
dz_m = z[worst_j] - z[worst_j-1]
dYdz_m = (Y[:, worst_j] - Y[:, worst_j-1]) / dz_m
J_star_m = -rm_face * Dm_face * dYdz_m
# En Cantera, el sum_star y el Y se evaluan como? 
# Y_left? o promedio? Cantera usa Y_left que en el caso upwind es Y[j-1]?
# Vamos a usar Y_left = Y[worst_j-1] por ahora (pero en Cantera la conveccion depende del u)
# Wait, el corrección term en Cantera usa 0.5*(Y[j]+Y[j-1]) según su código actual:
# "double Y_face = 0.5 * (Y(k, j) + Y(k, j+1)); flux(k, j) = j_star(k) - Y_face * sum_star;"
Y_face_m = 0.5 * (Y[:, worst_j-1] + Y[:, worst_j])
flux_m = J_star_m[worst_k] - Y_face_m[worst_k] * np.sum(J_star_m)

# Flux j+1/2
rp_face = 0.5 * (rj + rp_node)
Dp_face = 0.5 * (Dj + Dp_node)
dz_p = z[worst_j+1] - z[worst_j]
dYdz_p = (Y[:, worst_j+1] - Y[:, worst_j]) / dz_p
J_star_p = -rp_face * Dp_face * dYdz_p
Y_face_p = 0.5 * (Y[:, worst_j] + Y[:, worst_j+1])
flux_p = J_star_p[worst_k] - Y_face_p[worst_k] * np.sum(J_star_p)

divJ = 2.0 * (flux_p - flux_m) / (z[worst_j+1] - z[worst_j-1])
source = oj[worst_k]

F_calc = (conv + divJ - source) / rj

print(f"conv={conv:.4e}, divJ={divJ:.4e}, source={source:.4e}")
print(f"J_star_m={J_star_m[worst_k]:.4e}, J_star_p={J_star_p[worst_k]:.4e}")
print(f"flux_m={flux_m:.4e}, flux_p={flux_p:.4e}")
print(f"Y_left_m={Y[worst_k, worst_j-1]:.4e}, Y_left_p={Y[worst_k, worst_j]:.4e}")
print(f"dz_m={dz_m:.4e}, dz_p={dz_p:.4e}, mean_dz={(dz_m+dz_p)/2:.4e}")
target_dz = (flux_p - flux_m + mdot * (Y[worst_k, worst_j] - Y[worst_k, worst_j-1])) / source
print(f"target_dz = {target_dz:.4e}")

# Check what dz makes the residual zero
diff_flux = flux_p - flux_m
diff_conv = mdot * (Y[worst_k, worst_j] - Y[worst_k, worst_j-1])
numerator = diff_flux + diff_conv

print("Numerator = ", numerator)
print("Source = ", source)
print("target_dz = numerator / source = ", numerator / source)

print("dz_m = ", dz_m)
print("dz_p = ", dz_p)
# Test central convection
Y_m = Y[worst_k, worst_j-1]
Y_j = Y[worst_k, worst_j]
Y_p = Y[worst_k, worst_j+1]
conv_central = mdot * (Y_p - Y_m) / (dz_m + dz_p)
conv_upwind_dz_m = mdot * (Y_j - Y_m) / dz_m
divJ_dz_m = diff_flux / dz_m
divJ_dz_full = diff_flux / (0.5 * (dz_m + dz_p))

print("conv_central = ", conv_central)
print("conv_upwind / dz_m = ", conv_upwind_dz_m)
print("divJ / dz_m = ", divJ_dz_m)
print("divJ / (dz_full/2) = ", divJ_dz_full)

print("Check: conv_central + divJ / (dz_full/2) = ", conv_central + divJ_dz_full)
print("Check: conv_upwind_dz_m + divJ_dz_m = ", conv_upwind_dz_m + divJ_dz_m)
print("Source = ", source)
