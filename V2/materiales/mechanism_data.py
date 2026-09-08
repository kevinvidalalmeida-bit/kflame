"""
mechanism_data.py – Parse Cantera YAML mechanism files and extract all
species, thermodynamic, transport, and kinetic data into plain NumPy arrays.

NO Cantera dependency.  Only uses PyYAML + NumPy.

CPU-oriented: mechanism arrays are stored as float64 NumPy arrays.
"""
from __future__ import annotations
import copy
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required.  Install with: pip install pyyaml")

# ── physical constants (SI) ────────────────────────────────────────────────
_MECHANISM_CACHE: dict[str, "MechanismData"] = {}

R_CGS    = 1.987  # cal/(mol·K)  – for Ea conversion
R_UNIV   = 8314.46261815324  # J/(kmol·K)  – Cantera convention
R_SI     = 8.31446261815324  # J/(mol·K)
AVOGADRO = 6.02214076e23
BOLTZMANN = 1.380649e-23
ONE_ATM  = 101325.0

# ── atomic weights aligned with Cantera defaults ────────────────────────────
_ATOMIC_WEIGHTS: dict[str, float] = {
    "H": 1.008,    "He": 4.002602, "C": 12.011,   "N": 14.007,
    "O": 15.999,   "F": 18.998403, "Ne": 20.1797,  "Ar": 39.950,
    "S": 32.06,    "Cl": 35.45,    "P": 30.973761,
}


# ═══════════════════════════════════════════════════════════════════════════
#  Data-classes for parsed mechanism
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ReactionData:
    """All data for ONE reaction, stored as plain Python / small arrays."""
    index: int
    equation: str
    rtype: str  # "elementary" | "three-body" | "falloff"

    # stoichiometry: reactant_species[k] -> nu_r[k], product_species[k] -> nu_p[k]
    reactant_indices: np.ndarray   # (n_r,)  int
    reactant_stoich: np.ndarray    # (n_r,)  float
    product_indices: np.ndarray    # (n_p,)  int
    product_stoich: np.ndarray     # (n_p,)  float
    reversible: bool

    # Arrhenius parameters (high-P or only)  – SI consistent
    A: float       # pre-exp  [SI]
    b: float       # temperature exponent
    Ea: float      # activation energy [J/kmol]

    # Three-body
    efficiencies: np.ndarray | None  # (n_sp,) – None if not 3-body/falloff

    # Falloff (low-pressure Arrhenius)
    A_low: float   = 0.0
    b_low: float   = 0.0
    Ea_low: float  = 0.0

    # Troe parameters (may be None)
    troe_A: float  = 0.0
    troe_T3: float = 0.0
    troe_T1: float = 0.0
    troe_T2: float = 1e30   # very large → exp(-T/T2)≈0
    has_troe: bool = False

    # Net stoichiometric coefficients  (n_sp,) — built after init
    net_stoich: np.ndarray | None = None


@dataclass
class MechanismData:
    """Complete parsed mechanism in plain arrays."""
    # species info
    species_names: list[str]
    n_species: int
    molecular_weights: np.ndarray         # (n_sp,)  [kg/kmol]
    inv_molecular_weights: np.ndarray     # (n_sp,)  [kmol/kg]

    # thermodynamics – NASA-7
    nasa_low: np.ndarray                  # (n_sp, 7)
    nasa_high: np.ndarray                 # (n_sp, 7)
    nasa_Tmid: np.ndarray                 # (n_sp,)

    # transport
    geometry: np.ndarray                  # (n_sp,)  0=atom 1=linear 2=nonlinear
    well_depth: np.ndarray                # (n_sp,)  ε/kB  [K]
    diameter: np.ndarray                  # (n_sp,)  σ     [m]  (Å→m)
    dipole: np.ndarray                    # (n_sp,)  μ     [C·m]
    polarizability: np.ndarray            # (n_sp,)  α     [m³]
    rot_relax: np.ndarray                 # (n_sp,)  Z_rot [-]

    # kinetics
    reactions: list[ReactionData]
    n_reactions: int

    # dense stoichiometry matrices  (n_sp × n_rxn)
    nu_reactants: np.ndarray              # (n_sp, n_rxn) float
    nu_products: np.ndarray               # (n_sp, n_rxn) float
    nu_net: np.ndarray                    # (n_sp, n_rxn) float  (products - reactants)

    # pressure, reference
    pressure: float = ONE_ATM
    ref_pressure: float = ONE_ATM
    min_temperature: float = 300.0
    max_temperature: float = 3000.0

# ═══════════════════════════════════════════════════════════════════════════
#  Unit conversions used by gri30.yaml
# ═══════════════════════════════════════════════════════════════════════════

def _ea_to_SI(ea_val: float, ea_units: str) -> float:
    """Convert activation energy to J/kmol."""
    u = ea_units.lower().replace("/", "").replace(" ", "")
    if u in ("calmol", "cal/mol"):
        return ea_val * 4.184 * 1000.0  # cal/mol → J/kmol
    if u in ("jmol", "j/mol"):
        return ea_val * 1000.0  # J/mol → J/kmol
    if u in ("jkmol", "j/kmol"):
        return ea_val
    if u in ("kjmol", "kj/mol"):
        return ea_val * 1e6  # kJ/mol → J/kmol
    if u in ("kcalmol", "kcal/mol"):
        return ea_val * 4.184e6
    if u in ("kelvin", "k"):
        return ea_val * R_UNIV
    raise ValueError(f"Unknown Ea unit: {ea_units}")


def _A_to_SI(A_val: float, length_units: str, quantity_units: str,
             n_reactants_total: float) -> float:
    """
    Convert pre-exponential factor A to SI (kmol, m, s).

    For a reaction with total order n,
    A has units [conc]^(1-n) / s  →  [mol/length³]^(1-n) / s

    Cantera YAML specifies length and quantity units in the header.
    """
    # length conversion
    if length_units == "cm":
        len_factor = 1e-2  # cm → m
    elif length_units == "m":
        len_factor = 1.0
    else:
        raise ValueError(f"Unknown length unit: {length_units}")

    # quantity conversion
    if quantity_units == "mol":
        qty_factor = 1e-3  # mol → kmol  (concentration: mol/cm³ → kmol/m³)
    elif quantity_units == "kmol":
        qty_factor = 1.0
    elif quantity_units == "molecule":
        qty_factor = 1.0 / AVOGADRO / 1000.0
    else:
        raise ValueError(f"Unknown quantity unit: {quantity_units}")

    # concentration factor: [quantity / length³]
    conc_factor = qty_factor / (len_factor ** 3)
    order = n_reactants_total
    # A * [old_conc]^(1-n) = A_SI * [SI_conc]^(1-n)
    return A_val * conc_factor ** (1.0 - order)


# ═══════════════════════════════════════════════════════════════════════════
#  Equation parser
# ═══════════════════════════════════════════════════════════════════════════

_COEFF_SPECIES = re.compile(r"^(\d*\.?\d*)\s*(.+)$")

def _parse_species_list(text: str, sp_index: dict[str, int]):
    """Parse '2 H + O2 + M' → (indices, stoich, has_M)."""
    text = text.strip()
    has_M = False

    # Remove "(+M)" BEFORE splitting by "+" to avoid breaking species names
    if "(+M)" in text:
        has_M = True
        text = text.replace("(+M)", "").strip()

    parts = [s.strip() for s in text.split("+")]
    indices: list[int] = []
    stoich: list[float] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p == "M":
            has_M = True
            continue
        m = _COEFF_SPECIES.match(p)
        if not m:
            raise ValueError(f"Cannot parse species token: '{p}'")
        coeff_str, name = m.group(1), m.group(2).strip()
        coeff = float(coeff_str) if coeff_str else 1.0
        if name not in sp_index:
            raise KeyError(f"Species '{name}' not found in mechanism")
        indices.append(sp_index[name])
        stoich.append(coeff)
    return np.array(indices, dtype=int), np.array(stoich, dtype=float), has_M


def _parse_equation(eq: str, sp_index: dict[str, int]):
    """Parse full equation string → reactants, products, reversible."""
    # strip comments
    eq = re.sub(r"#.*$", "", eq).strip()
    reversible = "<=>" in eq
    if "<=>" in eq:
        left, right = eq.split("<=>")
    elif "=>" in eq:
        left, right = eq.split("=>")
    else:
        raise ValueError(f"Cannot determine direction in equation: {eq}")

    # Remove (+M) markers from sides for parsing
    ri, rs, rM = _parse_species_list(left, sp_index)
    pi, ps, pM = _parse_species_list(right, sp_index)
    return ri, rs, pi, ps, reversible, (rM or pM)


# ═══════════════════════════════════════════════════════════════════════════
#  Main parser
# ═══════════════════════════════════════════════════════════════════════════

def load_mechanism(filepath: str | Path) -> MechanismData:
    """Parse a Cantera YAML mechanism file into MechanismData."""
    filepath = Path(filepath)
    
    # If the file doesn't exist, try to find it in common locations
    if not filepath.exists():
        candidates = [
            filepath,
            Path(__file__).parent.parent / "no_esencial" / "cantera-main" / "data" / filepath.name,
            Path(__file__).parent / "data" / filepath.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                filepath = candidate
                break
        else:
            # If still not found, raise the original error
            raise FileNotFoundError(f"[Errno 2] No such file or directory: '{filepath}'")

    cache_key = str(filepath.resolve())
    cached = _MECHANISM_CACHE.get(cache_key)
    if cached is not None:
        return copy.deepcopy(cached)
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # ── units ──────────────────────────────────────────────────────────
    units = data.get("units", {})
    length_u = units.get("length", "cm")
    # time_u   = units.get("time", "s")
    quant_u  = units.get("quantity", "mol")
    ea_u     = units.get("activation-energy", "cal/mol")

    # ── species ────────────────────────────────────────────────────────
    sp_list = data["species"]
    n_sp = len(sp_list)

    # YAML interprets 'NO' and 'ON' as booleans (False/True).
    # Force every name to string.
    _YAML_BOOL_MAP = {False: "NO", True: "ON"}

    def _fix_name(v):
        if isinstance(v, bool):
            return _YAML_BOOL_MAP[v]
        return str(v)

    species_names = [_fix_name(sp["name"]) for sp in sp_list]
    sp_index = {name: i for i, name in enumerate(species_names)}

    molecular_weights = np.zeros(n_sp)
    nasa_low  = np.zeros((n_sp, 7))
    nasa_high = np.zeros((n_sp, 7))
    nasa_Tmid = np.zeros(n_sp)
    temperature_minima = np.zeros(n_sp)
    temperature_maxima = np.zeros(n_sp)

    geometry      = np.zeros(n_sp)
    well_depth    = np.zeros(n_sp)
    diameter_arr  = np.zeros(n_sp)
    dipole_arr    = np.zeros(n_sp)
    polar_arr     = np.zeros(n_sp)
    rot_relax_arr = np.zeros(n_sp)

    for i, sp in enumerate(sp_list):
        # molecular weight
        comp = sp["composition"]
        mw = sum(count * _ATOMIC_WEIGHTS[elem] for elem, count in comp.items())
        molecular_weights[i] = mw

        # thermo
        thermo = sp["thermo"]
        assert thermo["model"] == "NASA7", f"Only NASA7 supported, got {thermo['model']}"
        tranges = thermo["temperature-ranges"]
        temperature_minima[i] = tranges[0]
        temperature_maxima[i] = tranges[-1]
        nasa_Tmid[i] = tranges[1]
        coefs = thermo["data"]
        nasa_low[i, :] = coefs[0]
        nasa_high[i, :] = coefs[1]

        # transport
        tr = sp.get("transport", {})
        geom_str = tr.get("geometry", "nonlinear")
        geometry[i] = {"atom": 0, "linear": 1, "nonlinear": 2}.get(geom_str, 2)
        well_depth[i]    = tr.get("well-depth", 0.0)          # K
        diameter_arr[i]  = tr.get("diameter", 0.0) * 1e-10     # Å → m
        dipole_arr[i]    = tr.get("dipole", 0.0) * 3.33564e-30  # Debye → C·m
        polar_arr[i]     = tr.get("polarizability", 0.0) * 1e-30  # Å³ → m³
        rot_relax_arr[i] = tr.get("rotational-relaxation", 0.0)

    inv_mw = 1.0 / molecular_weights

    # ── reactions ──────────────────────────────────────────────────────
    raw_rxns = data.get("reactions", [])
    reactions: list[ReactionData] = []

    for idx, rxn in enumerate(raw_rxns):
        eq_str = rxn["equation"]
        rtype = rxn.get("type", "elementary")

        ri, rs, pi, ps, reversible, has_M = _parse_equation(eq_str, sp_index)

        # total reactant stoichiometric order (for unit conversion)
        total_order = float(rs.sum())

        # Arrhenius parameters
        if rtype == "falloff":
            rc_hi = rxn["high-P-rate-constant"]
            A_hi  = _A_to_SI(rc_hi["A"], length_u, quant_u, total_order)
            b_hi  = rc_hi["b"]
            Ea_hi = _ea_to_SI(rc_hi["Ea"], ea_u)

            rc_lo = rxn["low-P-rate-constant"]
            # low-P has one extra order (+1 for [M])
            A_lo  = _A_to_SI(rc_lo["A"], length_u, quant_u, total_order + 1.0)
            b_lo  = rc_lo["b"]
            Ea_lo = _ea_to_SI(rc_lo["Ea"], ea_u)
        elif rtype == "three-body":
            rc = rxn["rate-constant"]
            # three-body: extra order from [M]
            A_hi  = _A_to_SI(rc["A"], length_u, quant_u, total_order + 1.0)
            b_hi  = rc["b"]
            Ea_hi = _ea_to_SI(rc["Ea"], ea_u)
            A_lo, b_lo, Ea_lo = 0.0, 0.0, 0.0
        else:
            rc = rxn["rate-constant"]
            A_hi  = _A_to_SI(rc["A"], length_u, quant_u, total_order)
            b_hi  = rc["b"]
            Ea_hi = _ea_to_SI(rc["Ea"], ea_u)
            A_lo, b_lo, Ea_lo = 0.0, 0.0, 0.0

        # efficiencies
        eff = None
        if rtype in ("three-body", "falloff"):
            eff = np.ones(n_sp)
            for sp_name, val in rxn.get("efficiencies", {}).items():
                sp_name_fixed = _fix_name(sp_name)
                if sp_name_fixed in sp_index:
                    eff[sp_index[sp_name_fixed]] = val

        # Troe
        has_troe = False
        troe_A, troe_T3, troe_T1, troe_T2 = 0.0, 0.0, 0.0, 1e30
        if "Troe" in rxn:
            has_troe = True
            troe = rxn["Troe"]
            troe_A  = troe["A"]
            troe_T3 = troe["T3"]
            troe_T1 = troe["T1"]
            troe_T2 = troe.get("T2", 1e30)

        # build net stoich for this reaction
        net = np.zeros(n_sp)
        for k, nu in zip(ri, rs):
            net[k] -= nu
        for k, nu in zip(pi, ps):
            net[k] += nu

        rd = ReactionData(
            index=idx, equation=eq_str, rtype=rtype,
            reactant_indices=ri, reactant_stoich=rs,
            product_indices=pi, product_stoich=ps,
            reversible=reversible,
            A=A_hi, b=b_hi, Ea=Ea_hi,
            efficiencies=eff,
            A_low=A_lo, b_low=b_lo, Ea_low=Ea_lo,
            troe_A=troe_A, troe_T3=troe_T3, troe_T1=troe_T1, troe_T2=troe_T2,
            has_troe=has_troe,
            net_stoich=net,
        )
        reactions.append(rd)

    n_rxn = len(reactions)

    # Dense stoichiometry matrices
    nu_r = np.zeros((n_sp, n_rxn))
    nu_p = np.zeros((n_sp, n_rxn))
    for j, rxn in enumerate(reactions):
        for k, nu in zip(rxn.reactant_indices, rxn.reactant_stoich):
            # Accumulate in case a species appears multiple times
            # (e.g. "CH2 + CH2" instead of "2 CH2").
            nu_r[k, j] += nu
        for k, nu in zip(rxn.product_indices, rxn.product_stoich):
            nu_p[k, j] += nu
    nu_net = nu_p - nu_r

    mech = MechanismData(
        species_names=species_names,
        n_species=n_sp,
        molecular_weights=molecular_weights,
        inv_molecular_weights=inv_mw,
        nasa_low=nasa_low,
        nasa_high=nasa_high,
        nasa_Tmid=nasa_Tmid,
        geometry=geometry,
        well_depth=well_depth,
        diameter=diameter_arr,
        dipole=dipole_arr,
        polarizability=polar_arr,
        rot_relax=rot_relax_arr,
        reactions=reactions,
        n_reactions=n_rxn,
        nu_reactants=nu_r,
        nu_products=nu_p,
        nu_net=nu_net,
        min_temperature=float(np.max(temperature_minima)),
        max_temperature=float(np.min(temperature_maxima)),
    )
    _MECHANISM_CACHE[cache_key] = copy.deepcopy(mech)
    return mech
