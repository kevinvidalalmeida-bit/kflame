"""
config.py – Parámetros del caso de llama libre premezclada.
"""
from dataclasses import dataclass


@dataclass
class FlameCase:
    mech: str = "gri30.yaml"
    fuel: str = "CH4"
    oxidizer: str = "O2:1.0, N2:3.76"
    phi: float = 1.0
    T_in: float = 300.0
    P: float = 101325.0
    width: float = 0.03
    inlet_mass_fractions: tuple[float, ...] | None = None
    initial_grid: tuple[float, ...] | None = None
    steady_rtol: float = 1e-4
    steady_atol: float = 1e-9

    # Transporte
    transport_model: str = "mixture-averaged"
    flux_gradient_basis: str = "molar"    # "molar" | "mass"
    soret_enabled: bool = False
    outlet_species_bc: str = "zero_gradient"  # "zero_gradient" | "cantera_flux"

    # Discretizacion convectiva:
    # 1.0 = upwind puro (robusto, default actual), 0.0 = central.
    upwind_factor: float = 1.0

    # Malla inicial
    cantera_seed_grid: bool = True
    adaptive_grid: bool = True
    cluster_strength: float = 8.0
    cluster_sigma: float = 0.0           # <= 0 → automático

    # Criterios de refinamiento (Refiner::setCriteria defaults)
    ratio: float = 10.0
    slope: float = 0.8
    curve: float = 0.8
    prune: float = 0.05

    # Qué perfiles se usan para refinar
    refine_with_u: bool = True
    refine_with_T: bool = True
    refine_with_species: bool = True

    loglevel: int = 1
    auto: bool = True
