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


# ---------------------------------------------------------------------------
# Dataclass de registro por flamelet
# ---------------------------------------------------------------------------

@dataclass
class FlameRecord:
    phi: float
    Z: float            # <<< NUEVO: fracción de mezcla de Bilger
    solve_time_s: float
    n_points: int
    width_m: float
    Su_m_per_s: float
    z: np.ndarray
    u: np.ndarray
    T: np.ndarray
    rho: np.ndarray
    cp_mass: np.ndarray
    qdot: np.ndarray
    Y: np.ndarray
    c: np.ndarray
    beta: np.ndarray


# ---------------------------------------------------------------------------
# Parsers de argumentos
# ---------------------------------------------------------------------------

def parse_progress_weights(text: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    if not text.strip():
        return weights
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Formato inválido en progress-species: '{token}'. Usa especie:peso.")
        sp, w = token.split(":", 1)
        weights[sp.strip()] = float(w.strip())
    return weights


def parse_species_list(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def parse_phi_values(args: argparse.Namespace) -> np.ndarray:
    if args.phi_values.strip():
        vals = [float(v.strip()) for v in args.phi_values.split(",") if v.strip()]
        return np.array(sorted(vals), dtype=float)
    if args.n_phi < 2:
        return np.array([float(args.phi_min)], dtype=float)
    return np.linspace(float(args.phi_min), float(args.phi_max), int(args.n_phi))


# ---------------------------------------------------------------------------
# Fracción de mezcla de Bilger  <<<  NUEVA FUNCIÓN
# ---------------------------------------------------------------------------

def compute_bilger_Z(phi: float, args: argparse.Namespace) -> float:
    """
    Calcula la fracción de mezcla de Bilger para la mezcla premezclada
    con equivalence ratio `phi`.  Para una llama premezclada Z es uniforme
    a lo largo del flamelet.

    Cantera define Z=1 para combustible puro y Z=0 para oxidante puro,
    usando la fórmula de Bilger basada en átomos de C, H, O.
    """
    gas = ct.Solution(args.mech)
    gas.TP = float(args.T_in), float(args.P)
    gas.set_equivalence_ratio(float(phi), args.fuel, args.oxidizer)
    # basis='mass' → fracción de mezcla másica (estándar en FGM)
    Z = float(gas.mixture_fraction(args.fuel, args.oxidizer, basis="mass"))
    return Z


# ---------------------------------------------------------------------------
# Variable de progreso
# ---------------------------------------------------------------------------

def compute_progress_variable(
    species_names: list[str],
    Y: np.ndarray,
    T: np.ndarray,
    progress_weights: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    sp_to_idx = {sp: i for i, sp in enumerate(species_names)}
    used: list[str] = []
    beta = np.zeros(Y.shape[1], dtype=float)
    for sp, w in progress_weights.items():
        if sp not in sp_to_idx:
            continue
        beta += float(w) * Y[sp_to_idx[sp]]
        used.append(sp)

    if not used:
        dT = float(T[-1] - T[0])
        c = np.linspace(0.0, 1.0, T.size) if abs(dT) < 1e-14 else (T - T[0]) / dT
        return np.clip(c, 0.0, 1.0), c.copy(), used

    beta_u, beta_b = float(beta[0]), float(beta[-1])
    den = beta_b - beta_u
    if abs(den) < 1e-14:
        dT = float(T[-1] - T[0])
        c = np.linspace(0.0, 1.0, T.size) if abs(dT) < 1e-14 else (T - T[0]) / dT
    else:
        c = (beta - beta_u) / den
    return np.clip(c, 0.0, 1.0), beta, used


# ---------------------------------------------------------------------------
# Monotonicización en c
# ---------------------------------------------------------------------------

def monotonicize_on_c(
    c: np.ndarray,
    fields: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    mask = np.isfinite(c)
    for arr in fields.values():
        mask &= np.isfinite(arr)
    if int(np.count_nonzero(mask)) < 2:
        raise RuntimeError("No hay puntos finitos suficientes para construir c monótona.")

    c0 = np.asarray(c[mask], dtype=float)
    order = np.argsort(c0)
    c_sorted = c0[order]
    uniq_c, inv = np.unique(c_sorted, return_inverse=True)

    out: dict[str, np.ndarray] = {}
    for key, arr in fields.items():
        a_sorted = np.asarray(arr[mask], dtype=float)[order]
        acc = np.zeros(uniq_c.size, dtype=float)
        cnt = np.zeros(uniq_c.size, dtype=float)
        np.add.at(acc, inv, a_sorted)
        np.add.at(cnt, inv, 1.0)
        out[key] = acc / np.maximum(cnt, 1.0)

    c_u = uniq_c.copy()
    c_u[0] = max(0.0, c_u[0])
    c_u[-1] = min(1.0, c_u[-1])

    if c_u[0] > 0.0:
        c_u = np.concatenate(([0.0], c_u))
        for k in out:
            out[k] = np.concatenate(([out[k][0]], out[k]))
    if c_u[-1] < 1.0:
        c_u = np.concatenate((c_u, [1.0]))
        for k in out:
            out[k] = np.concatenate((out[k], [out[k][-1]]))

    return c_u, out


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

    # <<< Fracción de mezcla de Bilger ANTES de resolver (composición de entrada)
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
    qdot = np.asarray(flame.heat_release_rate, dtype=float)
    Y = np.asarray(flame.Y, dtype=float)

    c, beta, _ = compute_progress_variable(
        species_names=list(flame.gas.species_names),
        Y=Y,
        T=T,
        progress_weights=progress_weights,
    )

    rec = FlameRecord(
        phi=float(phi),
        Z=Z,                    # <<< NUEVO
        solve_time_s=dt,
        n_points=int(z.size),
        width_m=float(z[-1] - z[0]),
        Su_m_per_s=float(u[0]),
        z=z, u=u, T=T, rho=rho, cp_mass=cp_mass,
        qdot=qdot, Y=Y, c=c, beta=beta,
    )
    return rec, flame.to_array(normalize=True)


# ---------------------------------------------------------------------------
# Indicador global para malla adaptativa en c
# ---------------------------------------------------------------------------

def build_global_indicator(
    records: list[FlameRecord],
    species_names: list[str],
    indicator_species: list[str],
    c_fine: np.ndarray,
    w_grad: float, w_conc: float, w_temp: float, w_qdot: float,
) -> np.ndarray:
    sp_to_idx = {sp: i for i, sp in enumerate(species_names)}
    idx_sel = [sp_to_idx[sp] for sp in indicator_species if sp in sp_to_idx]
    eps = 1e-30
    acc = np.zeros_like(c_fine)
    n_ok = 0

    for rec in records:
        c_u, out = monotonicize_on_c(rec.c, {"T": rec.T, "qdot": rec.qdot})
        score = np.zeros_like(c_u)
        dTdc = np.abs(np.gradient(out["T"], c_u, edge_order=1))
        score += float(w_temp) * (dTdc / (np.max(dTdc) + eps))
        qn = np.abs(out["qdot"])
        score += float(w_qdot) * (qn / (np.max(qn) + eps))
        for k in idx_sel:
            c_k, y_k_map = monotonicize_on_c(rec.c, {"Y": rec.Y[k]})
            yk = y_k_map["Y"]
            dykdc = np.abs(np.gradient(yk, c_k, edge_order=1))
            loc = float(w_grad) * (dykdc / (np.max(dykdc) + eps)) + \
                  float(w_conc) * (yk / (np.max(yk) + eps))
            score += np.interp(c_u, c_k, loc, left=loc[0], right=loc[-1])
        if np.max(score) > 0:
            score /= np.max(score)
        acc += np.interp(c_fine, c_u, score, left=score[0], right=score[-1])
        n_ok += 1

    return acc / float(n_ok) if n_ok > 0 else np.ones_like(c_fine)


def build_adaptive_c_grid(
    c_fine: np.ndarray, indicator: np.ndarray, n_c: int, bias: float
) -> np.ndarray:
    n_c = int(max(8, n_c))
    w = 1.0 + float(bias) * np.maximum(indicator, 0.0)
    dc = np.diff(c_fine)
    w_mid = 0.5 * (w[:-1] + w[1:])
    cdf = np.concatenate(([0.0], np.cumsum(w_mid * dc)))
    total = float(cdf[-1])
    if total <= 0.0:
        return np.linspace(0.0, 1.0, n_c)
    cdf /= total
    c_adapt = np.interp(np.linspace(0.0, 1.0, n_c), cdf, c_fine)
    c_uni = np.linspace(0.0, 1.0, max(10, n_c // 6))
    c_mix = np.unique(np.concatenate(([0.0], c_adapt, c_uni, [1.0])))
    if c_mix.size != n_c:
        c_mix = np.interp(
            np.linspace(0.0, 1.0, n_c),
            np.linspace(0.0, 1.0, c_mix.size),
            c_mix,
        )
    c_mix[0] = 0.0
    c_mix[-1] = 1.0
    return c_mix


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
            Z=np.array([rec.Z], dtype=float),   # <<< NUEVO
            z=rec.z, u=rec.u, T=rec.T, rho=rec.rho,
            cp_mass=rec.cp_mass, qdot=rec.qdot,
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
    Z_grid   = np.array([r.Z   for r in records], dtype=float)   # <<< NUEVO
    Su       = np.array([r.Su_m_per_s for r in records], dtype=float)
    n_points = np.array([r.n_points   for r in records], dtype=int)
    width    = np.array([r.width_m    for r in records], dtype=float)
    solve_time = np.array([r.solve_time_s for r in records], dtype=float)

    T_tab    = np.zeros((n_phi, n_c), dtype=float)
    u_tab    = np.zeros((n_phi, n_c), dtype=float)
    rho_tab  = np.zeros((n_phi, n_c), dtype=float)
    cp_tab   = np.zeros((n_phi, n_c), dtype=float)
    qdot_tab = np.zeros((n_phi, n_c), dtype=float)
    beta_tab = np.zeros((n_phi, n_c), dtype=float)
    Y_tab    = np.zeros((n_phi, n_sp, n_c), dtype=float)

    for i, rec in enumerate(records):
        c_u, base = monotonicize_on_c(
            rec.c,
            {"T": rec.T, "u": rec.u, "rho": rec.rho,
             "cp": rec.cp_mass, "qdot": rec.qdot, "beta": rec.beta},
        )
        T_tab[i]    = np.interp(c_grid, c_u, base["T"])
        u_tab[i]    = np.interp(c_grid, c_u, base["u"])
        rho_tab[i]  = np.interp(c_grid, c_u, base["rho"])
        cp_tab[i]   = np.interp(c_grid, c_u, base["cp"])
        qdot_tab[i] = np.interp(c_grid, c_u, base["qdot"])
        beta_tab[i] = np.interp(c_grid, c_u, base["beta"])
        for k in range(n_sp):
            c_k, out_k = monotonicize_on_c(rec.c, {"Y": rec.Y[k]})
            Y_tab[i, k] = np.interp(c_grid, c_k, out_k["Y"])

    return {
        "phi_grid": phi_grid,
        "Z_grid":   Z_grid,      # <<< NUEVO: eje primario de la tabla
        "c_grid":   c_grid,
        "Su": Su, "n_points": n_points, "width": width, "solve_time": solve_time,
        "T": T_tab, "u": u_tab, "rho": rho_tab, "cp_mass": cp_tab,
        "qdot": qdot_tab, "beta": beta_tab, "Y": Y_tab,
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

    # Refinamiento Cantera
    p.add_argument("--ratio",             type=float, default=3.0)
    p.add_argument("--slope",             type=float, default=0.08)
    p.add_argument("--curve",             type=float, default=0.12)
    p.add_argument("--prune",             type=float, default=0.01)
    p.add_argument("--max-grid-points",   type=int,   default=1600)
    p.add_argument("--grid-min",          type=float, default=0.0)
    p.add_argument("--tight-refine-passes", type=int, default=1)
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

    phi_vals = parse_phi_values(args)
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
    print(f"Output          : {out_dir.resolve()}")
    print(f"phis            : {phi_vals}")
    Z_preview = [compute_bilger_Z(phi, args) for phi in phi_vals]
    print(f"Z (Bilger)      : {[f'{z:.4f}' for z in Z_preview]}")
    print(f"Z_st (phi=1)    : {compute_bilger_Z(1.0, args):.6f}")

    records: list[FlameRecord] = []
    prev_solution = None
    species_names: list[str] | None = None
    used_progress_species: list[str] = []

    for i, phi in enumerate(phi_vals, start=1):
        print(f"\n[{i}/{len(phi_vals)}] Solving phi={phi:.4f}  Z={Z_preview[i-1]:.4f} ...")
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
        "cantera_version": ct.__version__,
        "args": vars(args),
        "n_species": len(species_names),
        "species_names": species_names,
        "used_progress_species": used_progress_species,
        "n_phi": len(phi_vals),
        "n_c": int(c_grid.size),
        "Z_range": [float(tables["Z_grid"].min()), float(tables["Z_grid"].max())],
        "Z_st": compute_bilger_Z(1.0, args),
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