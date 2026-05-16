from dataclasses import dataclass, asdict


@dataclass
class FlameCase:
    mech: str = "gri30.yaml"
    fuel: str = "CH4"
    oxidizer: str = "O2:1.0, N2:3.76"
    phi: float = 1.0
    T_in: float = 300.0
    P: float = 101325.0
    width: float = 0.03
    transport_model: str = "mixture-averaged"
    adaptive_grid: bool = True
    cluster_strength: float = 8.0
    cluster_sigma: float = 0.0  # <= 0 => automatico desde locs

    # criterios de refinamiento tipo Cantera
    ratio: float = 3.0
    slope: float = 0.07
    curve: float = 0.14
    prune: float = 0.0

    loglevel: int = 1
    auto: bool = True
