"""
generate_fgm_tables_cantera.py  (v2 – Z-C space)

Genera tablas FGM usando solo Cantera.
Ahora almacena la fracción de mezcla de Bilger Z para cada flamelet,
de modo que la tabla queda parametrizada en (Z, c) en lugar de (phi, c).

Para llamas premezcladas Z es constante a lo largo del flamelet y depende
únicamente de phi mediante la definición de Bilger.

Uso:
  python generate_fgm_tables_cantera.py --phi-min 0.7 --phi-max 1.4 --n-phi 13 \
         --save-raw-profiles
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cantera as ct
import numpy as np

from fgm_common import (
    build_adaptive_c_grid,
    build_global_indicator,
    compute_bilger_Z,
    compute_progress_variable,
    invert_bilger_Z_to_phi,
    monotonicize_on_c,
    parse_phi_values,
    parse_progress_weights,
    parse_species_list,
    parse_z_values,
    validate_fgm_table,
)

CANTERA_REFERENCE_WORKFLOW = "direct_cantera_freeflame"
CANTERA_REFERENCE_SOLVER = "ct.FreeFlame"
EXTERNAL_FLAMELET_TABLE_TOOL_USED = False
EXTERNAL_FLAMELET_TABLE_TOOL = None
TABLE_BUILDER = "repo_fgm_common"


# ---------------------------------------------------------------------------
# Dataclass de registro por flamelet
# ---------------------------------------------------------------------------

@dataclass
class FlameRecord:
    phi: float
    Z: float
    solve_time_s: float
    n_points: int
    width_m: float
    Su_m_per_s: float
    z: np.ndarray
    u: np.ndarray
    T: np.ndarray
    rho: np.ndarray
    cp_mass: np.ndarray
    conductivity: np.ndarray
    qdot: np.ndarray
    omega_c: np.ndarray
    Y: np.ndarray
    c: np.ndarray
    beta: np.ndarray


# ---------------------------------------------------------------------------
# Resolución de flamelet con Cantera
# ---------------------------------------------------------------------------

def solve_flame_cantera(
    phi: float,
    args: argparse.Namespace,
    prev_solution: Any | None,
    progress_weights: dict[str, float],
) -> tuple[FlameRecord, Any]:
    gas = ct.Solution(args.mech)
    gas.TP = float(args.T_in), float(args.P)
    gas.set_equivalence_ratio(float(phi), args.fuel, args.oxidizer)

    # Bilger mixture fraction of the unburned inlet composition.
    Z = float(gas.mixture_fraction(args.fuel, args.oxidizer, basis="mass"))

    flame = ct.FreeFlame(gas, width=float(args.width))
    flame.transport_model = args.transport_model
    if hasattr(flame, "flux_gradient_basis"):
        flame.flux_gradient_basis = args.flux_gradient_basis
    if hasattr(flame, "soret_enabled"):
        flame.soret_enabled = bool(args.soret_enabled)

    flame.set_refine_criteria(
        ratio=float(args.ratio),
        slope=float(args.slope),
        curve=float(args.curve),
        prune=float(args.prune),
    )
    flame.set_max_grid_points(flame.flame, int(args.max_grid_points))
    if args.grid_min > 0.0:
        flame.set_grid_min(float(args.grid_min))

    if prev_solution is not None:
        try:
            flame.set_initial_guess(data=prev_solution)
        except Exception:
            pass

    t0 = time.perf_counter()
    flame.solve(loglevel=int(args.loglevel), auto=True, refine_grid=True)

    if int(args.tight_refine_passes) > 0:
        flame.set_refine_criteria(
            ratio=float(args.tight_ratio),
            slope=float(args.tight_slope),
            curve=float(args.tight_curve),
            prune=float(args.tight_prune),
        )
        for _ in range(int(args.tight_refine_passes)):
            flame.solve(loglevel=int(args.loglevel), auto=False, refine_grid=True)

    dt = float(time.perf_counter() - t0)

    z = np.asarray(flame.grid, dtype=float)
    u = np.asarray(flame.velocity, dtype=float)
    T = np.asarray(flame.T, dtype=float)
    rho = np.asarray(flame.density, dtype=float)
    cp_mass = np.asarray(flame.cp_mass, dtype=float)
    conductivity = np.asarray(flame.thermal_conductivity, dtype=float)
    qdot = np.asarray(flame.heat_release_rate, dtype=float)
    Y = np.asarray(flame.Y, dtype=float)

    c, beta, _ = compute_progress_variable(
        species_names=list(flame.gas.species_names),
        Y=Y,
        T=T,
        progress_weights=progress_weights,
    )
    beta_span = float(beta[-1] - beta[0])
    if abs(beta_span) <= 1.0e-14:
        raise RuntimeError(
            f"La variable de progreso no define una fuente normalizada en phi={phi:g}."
        )
    molecular_weights = np.asarray(flame.gas.molecular_weights, dtype=float)
    net_rates_mass = (
        np.asarray(flame.net_production_rates, dtype=float)
        * molecular_weights[:, None]
    )
    omega_beta = np.zeros_like(T)
    species_index = {name: k for k, name in enumerate(flame.gas.species_names)}
    for species, weight in progress_weights.items():
        if species in species_index:
            omega_beta += float(weight) * net_rates_mass[species_index[species]]
    omega_c = omega_beta / beta_span

    rec = FlameRecord(
        phi=float(phi),
        Z=Z,
        solve_time_s=dt,
        n_points=int(z.size),
        width_m=float(z[-1] - z[0]),
        Su_m_per_s=float(u[0]),
        z=z, u=u, T=T, rho=rho, cp_mass=cp_mass,
        conductivity=conductivity, qdot=qdot, omega_c=omega_c,
        Y=Y, c=c, beta=beta,
    )
    return rec, flame.to_array(normalize=True)


# ---------------------------------------------------------------------------
# Escritura de resultados
# ---------------------------------------------------------------------------

def write_summary_csv(path: Path, records: list[FlameRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["phi", "Z", "Su_m_per_s", "n_points", "width_m", "solve_time_s"])
        for rec in records:
            wr.writerow([rec.phi, rec.Z, rec.Su_m_per_s, rec.n_points,
                         rec.width_m, rec.solve_time_s])


def write_raw_profiles(out_dir: Path, records: list[FlameRecord], species_names: list[str]) -> None:
    raw_dir = out_dir / "raw_profiles"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        tag = f"phi_{rec.phi:.6f}".replace(".", "p")
        np.savez_compressed(
            raw_dir / f"{tag}.npz",
            phi=np.array([rec.phi], dtype=float),
            Z=np.array([rec.Z], dtype=float),
            z=rec.z, u=rec.u, T=rec.T, rho=rec.rho,
            cp_mass=rec.cp_mass, conductivity=rec.conductivity,
            qdot=rec.qdot, omega_c=rec.omega_c,
            Y=rec.Y, c=rec.c, beta=rec.beta,
            species_names=np.array(species_names, dtype=object),
        )


# ---------------------------------------------------------------------------
# Construcción de tablas
# ---------------------------------------------------------------------------

def build_tables(
    records: list[FlameRecord],
    species_names: list[str],
    c_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    n_phi = len(records)
    n_c = int(c_grid.size)
    n_sp = len(species_names)

    phi_grid = np.array([r.phi for r in records], dtype=float)
    Z_grid = np.array([r.Z for r in records], dtype=float)
    Su       = np.array([r.Su_m_per_s for r in records], dtype=float)
    n_points = np.array([r.n_points   for r in records], dtype=int)
    width    = np.array([r.width_m    for r in records], dtype=float)
    solve_time = np.array([r.solve_time_s for r in records], dtype=float)

    T_tab    = np.zeros((n_phi, n_c), dtype=float)
    u_tab    = np.zeros((n_phi, n_c), dtype=float)
    rho_tab  = np.zeros((n_phi, n_c), dtype=float)
    cp_tab   = np.zeros((n_phi, n_c), dtype=float)
    conductivity_tab = np.zeros((n_phi, n_c), dtype=float)
    qdot_tab = np.zeros((n_phi, n_c), dtype=float)
    omega_c_tab = np.zeros((n_phi, n_c), dtype=float)
    beta_tab = np.zeros((n_phi, n_c), dtype=float)
    Y_tab    = np.zeros((n_phi, n_sp, n_c), dtype=float)

    for i, rec in enumerate(records):
        c_u, base = monotonicize_on_c(
            rec.c,
            {"T": rec.T, "u": rec.u, "rho": rec.rho,
             "cp": rec.cp_mass, "conductivity": rec.conductivity,
             "qdot": rec.qdot, "omega_c": rec.omega_c, "beta": rec.beta},
        )
        T_tab[i]    = np.interp(c_grid, c_u, base["T"])
        u_tab[i]    = np.interp(c_grid, c_u, base["u"])
        rho_tab[i]  = np.interp(c_grid, c_u, base["rho"])
        cp_tab[i]   = np.interp(c_grid, c_u, base["cp"])
        conductivity_tab[i] = np.interp(c_grid, c_u, base["conductivity"])
        qdot_tab[i] = np.interp(c_grid, c_u, base["qdot"])
        omega_c_tab[i] = np.interp(c_grid, c_u, base["omega_c"])
        beta_tab[i] = np.interp(c_grid, c_u, base["beta"])
        for k in range(n_sp):
            c_k, out_k = monotonicize_on_c(rec.c, {"Y": rec.Y[k]})
            Y_tab[i, k] = np.interp(c_grid, c_k, out_k["Y"])

    return {
        "phi_grid": phi_grid,
        "Z_grid": Z_grid,
        "c_grid":   c_grid,
        "Su": Su, "n_points": n_points, "width": width, "solve_time": solve_time,
        "T": T_tab, "u": u_tab, "rho": rho_tab, "cp_mass": cp_tab,
        "conductivity": conductivity_tab, "qdot": qdot_tab,
        "omega_c": omega_c_tab, "beta": beta_tab, "Y": Y_tab,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Genera tablas FGM en espacio (Z, c) con Cantera."
    )
    p.add_argument("--mech",             type=str,   default="gri30.yaml")
    p.add_argument("--fuel",             type=str,   default="CH4")
    p.add_argument("--oxidizer",         type=str,   default="O2:1.0, N2:3.76")
    p.add_argument("--transport-model",  type=str,   default="mixture-averaged")
    p.add_argument("--flux-gradient-basis", type=str, default="molar")
    p.add_argument("--soret-enabled",    action="store_true")
    p.add_argument("--T-in",             type=float, default=300.0)
    p.add_argument("--P",                type=float, default=101325.0)
    p.add_argument("--width",            type=float, default=0.03)

    p.add_argument("--phi-min",    type=float, default=0.7)
    p.add_argument("--phi-max",    type=float, default=1.4)
    p.add_argument("--n-phi",      type=int,   default=13)
    p.add_argument("--phi-values", type=str,   default="",
                   help="Lista manual separada por comas. Ignora phi-min/max/n-phi.")
    p.add_argument("--use-z-grid", action="store_true",
                   help="Barrer por Z objetivo en lugar de phi.")
    p.add_argument("--z-min",      type=float, default=0.03)
    p.add_argument("--z-max",      type=float, default=0.08)
    p.add_argument("--n-z",        type=int,   default=13)
    p.add_argument("--z-values",   type=str,   default="",
                   help="Lista manual de Z separada por comas. Ignora z-min/max/n-z.")
    p.add_argument("--phi-bracket-min", type=float, default=1e-4,
                   help="Límite inferior de phi para invertir Z->phi.")
    p.add_argument("--phi-bracket-max", type=float, default=1e3,
                   help="Límite superior de phi para invertir Z->phi.")

    # Refinamiento Cantera
    p.add_argument("--ratio",             type=float, default=2.5)
    p.add_argument("--slope",             type=float, default=0.04)
    p.add_argument("--curve",             type=float, default=0.08)
    p.add_argument("--prune",             type=float, default=0.003)
    p.add_argument("--max-grid-points",   type=int,   default=1600)
    p.add_argument("--grid-min",          type=float, default=0.0)
    p.add_argument("--tight-refine-passes", type=int, default=0)
    p.add_argument("--tight-ratio",       type=float, default=2.5)
    p.add_argument("--tight-slope",       type=float, default=0.04)
    p.add_argument("--tight-curve",       type=float, default=0.08)
    p.add_argument("--tight-prune",       type=float, default=0.003)
    p.add_argument("--loglevel",          type=int,   default=0)

    # Tabla FGM
    p.add_argument("--n-c",            type=int,   default=241)
    p.add_argument("--c-fine",         type=int,   default=2001)
    p.add_argument("--refine-bias",    type=float, default=5.0)
    p.add_argument("--progress-species", type=str,
                   default="CO2:1.0,H2O:1.0,CO:1.0,H2:0.5")
    p.add_argument("--indicator-species", type=str,
                   default="CH4,O2,CO2,H2O,CO,H2,OH")
    p.add_argument("--indicator-weight-grad", type=float, default=1.0)
    p.add_argument("--indicator-weight-conc", type=float, default=0.6)
    p.add_argument("--indicator-weight-temp", type=float, default=0.8)
    p.add_argument("--indicator-weight-qdot", type=float, default=0.4)

    p.add_argument("--output-root",    type=str,   default="fgm_runs")
    p.add_argument("--run-name",       type=str,   default="")
    p.add_argument("--save-raw-profiles", action="store_true")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_argparser().parse_args()
    t_global0 = time.perf_counter()
    bilger_gas = ct.Solution(args.mech)

    z_mode = bool(args.use_z_grid or args.z_values.strip())
    if z_mode:
        z_targets = parse_z_values(args)
        phi_vals = np.array(
            [
                invert_bilger_Z_to_phi(
                    Z_target=float(z),
                    args=args,
                    phi_lo=float(args.phi_bracket_min),
                    phi_hi=float(args.phi_bracket_max),
                    gas=bilger_gas,
                )
                for z in z_targets
            ],
            dtype=float,
        )
    else:
        phi_vals = parse_phi_values(args)
        z_targets = np.array(
            [compute_bilger_Z(phi, args, bilger_gas) for phi in phi_vals],
            dtype=float,
        )

    Z_preview = (
        np.array(
            [compute_bilger_Z(phi, args, bilger_gas) for phi in phi_vals],
            dtype=float,
        )
        if z_mode
        else z_targets
    )
    Z_st = compute_bilger_Z(1.0, args, bilger_gas)

    progress_weights = parse_progress_weights(args.progress_species)
    indicator_species = parse_species_list(args.indicator_species)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() or f"run_{timestamp}_fgm_cantera"
    out_dir = Path(args.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Previsualizar mapa phi → Z
    print("=" * 70)
    print("FGM TABLE GENERATOR (Cantera) — Z-C space")
    print("=" * 70)
    print(f"Cantera version : {ct.__version__}")
    print(f"Reference flow  : {CANTERA_REFERENCE_WORKFLOW}")
    print("External tool   : none (Streamline Flamelet Table Tool not invoked)")
    print(f"Output          : {out_dir.resolve()}")
    print(f"mode            : {'Z-grid' if z_mode else 'phi-grid'}")
    print(f"phis            : {phi_vals}")
    if z_mode:
        print(f"Z target        : {[f'{z:.4f}' for z in z_targets]}")
    print(f"Z (Bilger)      : {[f'{z:.4f}' for z in Z_preview]}")
    print(f"Z_st (phi=1)    : {Z_st:.6f}")

    records: list[FlameRecord] = []
    prev_solution = None
    species_names: list[str] | None = None
    used_progress_species: list[str] = []

    for i, phi in enumerate(phi_vals, start=1):
        z_tag = z_targets[i - 1] if z_mode else Z_preview[i - 1]
        print(
            f"\n[{i}/{len(phi_vals)}] Solving phi={phi:.4f}  "
            f"Z_target={z_tag:.4f} ..."
        )
        rec, prev_solution = solve_flame_cantera(
            phi=float(phi), args=args,
            prev_solution=prev_solution,
            progress_weights=progress_weights,
        )
        records.append(rec)
        print(f"  Su={rec.Su_m_per_s:.4f} m/s | Z={rec.Z:.4f} | "
              f"n={rec.n_points} | t={rec.solve_time_s:.2f}s")

        if species_names is None:
            species_names = list(ct.Solution(args.mech).species_names)
        c_tmp, _, used = compute_progress_variable(
            species_names=species_names, Y=rec.Y, T=rec.T,
            progress_weights=progress_weights,
        )
        rec.c = c_tmp
        if used:
            used_progress_species = used

    assert species_names is not None

    c_fine = np.linspace(0.0, 1.0, int(max(101, args.c_fine)))
    indicator = build_global_indicator(
        records=records, species_names=species_names,
        indicator_species=indicator_species, c_fine=c_fine,
        w_grad=float(args.indicator_weight_grad),
        w_conc=float(args.indicator_weight_conc),
        w_temp=float(args.indicator_weight_temp),
        w_qdot=float(args.indicator_weight_qdot),
    )
    c_grid = build_adaptive_c_grid(
        c_fine=c_fine, indicator=indicator,
        n_c=int(args.n_c), bias=float(args.refine_bias),
    )

    tables = build_tables(records=records, species_names=species_names, c_grid=c_grid)
    table_validation = validate_fgm_table(tables)

    np.savez_compressed(
        out_dir / "fgm_table.npz",
        species_names=np.array(species_names, dtype=object),
        used_progress_species=np.array(used_progress_species, dtype=object),
        indicator_species=np.array(indicator_species, dtype=object),
        indicator_fine_c=c_fine,
        indicator_fine_value=indicator,
        **tables,
    )
    write_summary_csv(out_dir / "summary_phi.csv", records)
    if args.save_raw_profiles:
        write_raw_profiles(out_dir=out_dir, records=records, species_names=species_names)

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "solver_backend": "cantera",
        "reference_workflow": CANTERA_REFERENCE_WORKFLOW,
        "reference_solver": CANTERA_REFERENCE_SOLVER,
        "cantera_solve_call": "flame.solve(auto=True, refine_grid=True)",
        "tight_refine_solve_call": "flame.solve(auto=False, refine_grid=True)",
        "external_flamelet_table_tool_used": EXTERNAL_FLAMELET_TABLE_TOOL_USED,
        "external_flamelet_table_tool": EXTERNAL_FLAMELET_TABLE_TOOL,
        "table_builder": TABLE_BUILDER,
        "timing_scope": (
            "direct Cantera ct.FreeFlame solve plus optional tight refine passes; "
            "excludes any external Streamline Flamelet Table Tool execution"
        ),
        "continuation_seed_policy": (
            "previous accepted ct.FreeFlame solution array is reused sequentially "
            "with flame.set_initial_guess(data=...) when available"
        ),
        "cantera_version": ct.__version__,
        "args": vars(args),
        "mode": "Z-grid" if z_mode else "phi-grid",
        "z_targets": [float(z) for z in z_targets],
        "n_species": len(species_names),
        "species_names": species_names,
        "used_progress_species": used_progress_species,
        "table_field_units": {
            "T": "K",
            "u": "m/s",
            "rho": "kg/m^3",
            "cp_mass": "J/(kg K)",
            "conductivity": "W/(m K)",
            "qdot": "W/m^3",
            "omega_c": "kg/(m^3 s)",
            "Y": "1",
        },
        "omega_c_definition": (
            "sum_k(a_k W_k omega_k_molar)/(beta_b-beta_u), "
            "with the progress weights stored in args.progress_species"
        ),
        "n_phi": len(phi_vals),
        "n_c": int(c_grid.size),
        "table_validation": table_validation,
        "Z_range": [float(tables["Z_grid"].min()), float(tables["Z_grid"].max())],
        "Z_st": Z_st,
        "runtime_s": float(time.perf_counter() - t_global0),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "-" * 70)
    print("FGM listo — tabla en espacio (Z, c).")
    print(f"Z_grid : {tables['Z_grid']}")
    print(f"c_grid : {c_grid.size} puntos adaptativos")
    print(f"Tabla  : {(out_dir / 'fgm_table.npz').resolve()}")
    print(f"Runtime: {meta['runtime_s']:.1f} s")
    print("-" * 70)


if __name__ == "__main__":
    main()
