import numpy as np
import cantera as ct

def main():
    # Cargar datos de referencia
    data = np.load("outputs/reference/freeflame_ref.npz")
    z = data["z"]
    T = data["T"]
    Y = data["Y"]
    u = data["u"]
    rho = data["rho"]
    
    gas = ct.Solution('gri30.yaml')
    # Crear un simulador en la misma malla
    f = ct.FreeFlame(gas, grid=z)
    f.energy_enabled = True
    
    # Rellenar estado
    f.set_profile('T', z, T)
    for i, sp in enumerate(gas.species_names):
        f.set_profile(sp, z, Y[i, :])
    f.set_profile('velocity', z, u)
    
    # Asegurarnos de que no hay refinamiento de malla y hacer 1 paso o evaluar el residual
    f.flame.set_steady_tolerances(default=[1.0e-5, 1.0e-9])
    f.flame.set_transient_tolerances(default=[1.0e-5, 1.0e-9])
    
    # We can evaluate the residual using f.eval() if available, but Python API doesn't expose it directly for 1D domains.
    # Actually we can get the source term directly from Cantera!
    # Let's see wdot for worst node j=117
    
    j = 117
    print(f"Node {j}, z={z[j]}")
    gas.TPY = T[j], 101325.0, Y[:, j]
    wdot = gas.net_production_rates
    W = gas.molecular_weights
    
    idx_O2 = gas.species_index('O2')
    print(f"O2 index: {idx_O2}")
    source_O2 = wdot[idx_O2] * W[idx_O2]
    print(f"Cantera source term at j={j} for O2: {source_O2} kg/m^3/s")
    
    # Let's check rho_u
    rho_u_j = rho[j] * u[j]
    rho_u_jm1 = rho[j-1] * u[j-1]
    print(f"rho_u_j = {rho_u_j}, rho_u_jm1 = {rho_u_jm1}")
    
    # Let's solve it with auto=False, max_steps=0 or similar to see if it's already converged.
    f.solve(loglevel=1, auto=False)

if __name__ == "__main__":
    main()
