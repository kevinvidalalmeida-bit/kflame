"""
Diagnostico rapido: verificar que el error de discretizacion baje al
aumentar puntos. Si F(Y_ref) -> 0 con mas puntos, la discretizacion
es correcta.
"""
import numpy as np
from config import FlameCase
from problem import FreeFlameProblem
from state import pack_species
from species_backend import SpeciesBackend
from residual_species import residual_species_frozen

def test_resolution(n_pts):
    case = FlameCase(
        mech="gri30.yaml", fuel="CH4", oxidizer="O2:1.0, N2:3.76",
        phi=1.0, T_in=300.0, P=101325.0, width=0.03,
    )
    problem = FreeFlameProblem(case, n_points=n_pts, locs=(0.0, 0.3, 0.6, 1.0))
    problem.set_frozen_profiles_from_reference("outputs/reference/freeflame_ref.npz")
    problem.backend = SpeciesBackend(problem)

    Y_ref = problem.Y_ref.copy()
    xY_ref = pack_species(Y_ref[:-1, :])
    F = residual_species_frozen(xY_ref, problem)

    Fm = F.reshape((problem.n_ind_species, problem.n_points), order="C")
    # Encontrar el peor punto
    worst_j = np.argmax(np.max(np.abs(Fm), axis=0))
    worst_val = np.max(np.abs(Fm[:, worst_j]))
    worst_k = np.argmax(np.abs(Fm[:, worst_j]))
    sp_name = problem.species_names[worst_k]

    print(f"n_pts={n_pts:4d}  ||F||inf={np.linalg.norm(F, ord=np.inf):.4e}  "
          f"peor: j={worst_j} ({sp_name}), T={problem.T_fixed[worst_j]:.0f}K")

print("Convergencia de discretizacion (F evaluado en solucion de referencia):\n")
for n in [20, 40, 60, 80, 100]:
    test_resolution(n)
