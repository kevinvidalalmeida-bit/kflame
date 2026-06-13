"""
generate_stageA_controlled_report.py

Run the controlled fixed-mesh experiment (same final Cantera mesh) and
export a compact report with figures for Stage A comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np

from config import FlameCase
from controlled_cantera_mesh_experiment import cantera_reference, run_hybrid_on_fixed_mesh
from problem import FreeFlameProblem
from solver import SolveOptions
from species_backend import SpeciesBackend as CanteraSpeciesBackend
from state import pack_state


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Genera reporte Stage A en malla fija de Cantera."
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="resultados_estado/stageA_controlled",
        help="Carpeta de salida para summary y figuras.",
    )
    p.add_argument("--loglevel", type=int, default=0)
    p.add_argument(
        "--backend",
        type=str,
        default="cantera",
        choices=["cantera", "native"],
        help="Backend de propiedades para resolver Stage A.",
    )
    p.add_argument(
        "--use-gpu",
        action="store_true",
        help="Solo para backend=native: intentar evaluar propiedades con CuPy/GPU.",
    )
    p.add_argument(
        "--include-linear",
        action="store_true",
        help="Incluir también arranque lineal en resumen y gráficas.",
    )
    p.add_argument(
        "--steady-max-iter",
        type=int,
        default=20,
        help="Máximo de iteraciones Newton por intento steady.",
    )
    p.add_argument(
        "--max-time-step-count",
        type=int,
        default=500,
        help="Máximo de pasos transitorios acumulados del híbrido.",
    )
    return p


def _default_case() -> FlameCase:
    return FlameCase(
        mech="gri30.yaml",
        fuel="CH4",
        oxidizer="O2:1.0, N2:3.76",
        phi=1.0,
        T_in=300.0,
        P=101325.0,
        width=0.03,
        transport_model="mixture-averaged",
        flux_gradient_basis="molar",
        soret_enabled=False,
        ratio=10.0,
        slope=0.8,
        curve=0.8,
        prune=-0.1,
    )


def _resolve_mech_path(mech: str) -> str:
    p = Path(mech)
    if p.exists():
        return str(p.resolve())
    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / "no_esencial" / "cantera-main" / "data" / mech
    if candidate.exists():
        return str(candidate.resolve())
    return mech


def _make_backend(problem: FreeFlameProblem, backend_name: str, use_gpu: bool = False):
    if backend_name == "cantera":
        return CanteraSpeciesBackend(problem)
    if backend_name == "native":
        materials_dir = Path(__file__).resolve().parent / "materiales"
        mat_path = str(materials_dir)
        if mat_path not in sys.path:
            sys.path.insert(0, mat_path)
        from species_backend_native import NativeSpeciesBackend  # type: ignore

        return NativeSpeciesBackend(problem, use_gpu=use_gpu)
    raise ValueError(f"Backend no soportado: {backend_name}")


def _build_problem(case: FlameCase, z_ct: np.ndarray, backend_name: str, use_gpu: bool = False) -> FreeFlameProblem:
    problem = FreeFlameProblem(case, n_points=int(z_ct.size))
    problem.z = z_ct.copy()
    problem.n_points = int(z_ct.size)
    problem.width = float(z_ct[-1] - z_ct[0])
    problem.backend = _make_backend(problem, backend_name, use_gpu=use_gpu)
    return problem


def _plot_scalar(
    x: np.ndarray,
    y_ct: np.ndarray,
    y_a: np.ndarray,
    ylabel: str,
    title: str,
    out_path: Path,
    y_b: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    ax.plot(1e3 * x, y_ct, color="#1f77b4", label="Cantera")
    ax.plot(1e3 * x, y_a, "--", color="#2ca02c", label="Nuestro (Stage A)")
    if y_b is not None:
        ax.plot(1e3 * x, y_b, ":", color="#d62728", label="Nuestro (arranque lineal)")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_species(
    x: np.ndarray,
    species_names: list[str],
    Y_ct: np.ndarray,
    Y_a: np.ndarray,
    out_path: Path,
    Y_b: np.ndarray | None = None,
) -> None:
    pick = [sp for sp in ["CH4", "O2", "H2O", "CO2", "OH"] if sp in species_names]
    idx = [species_names.index(sp) for sp in pick]
    if not idx:
        return

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    cmap = plt.cm.tab10(np.linspace(0, 1, len(idx)))
    for c, k, sp in zip(cmap, idx, pick):
        ax.plot(1e3 * x, Y_ct[k], color=c, lw=2.0, label=f"{sp} Cantera")
        ax.plot(1e3 * x, Y_a[k], "--", color=c, lw=1.5, label=f"{sp} Stage A")
        if Y_b is not None:
            ax.plot(1e3 * x, Y_b[k], ":", color=c, lw=1.3, label=f"{sp} Lineal")

    ax.set_xlabel("z [mm]")
    ax.set_ylabel("Y_k [-]")
    ax.set_title("Perfiles de especies en misma malla (Cantera)")
    ax.grid(True, alpha=0.35)
    ax.legend(ncol=2, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_deltas(
    x: np.ndarray,
    T_ct: np.ndarray,
    u_ct: np.ndarray,
    T_a: np.ndarray,
    u_a: np.ndarray,
    out_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 6.2), sharex=True)
    ax1.plot(1e3 * x, T_a - T_ct, color="#2ca02c")
    ax1.axhline(0.0, color="black", lw=1.0, alpha=0.6)
    ax1.set_ylabel("Delta T [K]")
    ax1.set_title("Stage A - Cantera (misma malla)")
    ax1.grid(True, alpha=0.35)

    ax2.plot(1e3 * x, u_a - u_ct, color="#9467bd")
    ax2.axhline(0.0, color="black", lw=1.0, alpha=0.6)
    ax2.set_ylabel("Delta u [m/s]")
    ax2.set_xlabel("z [mm]")
    ax2.grid(True, alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case = _default_case()
    case.mech = _resolve_mech_path(case.mech)
    z_ct, u_ct, T_ct, Y_ct, t_ct = cantera_reference(case, loglevel=args.loglevel)
    Su_ct = float(u_ct[0])

    case_fix = FlameCase(
        mech=case.mech,
        fuel=case.fuel,
        oxidizer=case.oxidizer,
        phi=case.phi,
        T_in=case.T_in,
        P=case.P,
        width=float(z_ct[-1] - z_ct[0]),
        transport_model=case.transport_model,
        flux_gradient_basis=case.flux_gradient_basis,
        soret_enabled=case.soret_enabled,
        ratio=case.ratio,
        slope=case.slope,
        curve=case.curve,
        prune=case.prune,
    )

    problem = _build_problem(case_fix, z_ct, args.backend, use_gpu=bool(args.use_gpu))
    opts = SolveOptions(
        verbose=False,
        steady_max_iter=int(args.steady_max_iter),
        max_time_step_count=int(args.max_time_step_count),
    )

    x_ct = pack_state(u_ct, T_ct, Y_ct)
    _, res_a = run_hybrid_on_fixed_mesh(problem, opts, x_ct)

    res_b = None
    if args.include_linear:
        x_guess = problem.make_initial_guess(u_left_guess=opts.u_left_guess)
        _, res_b = run_hybrid_on_fixed_mesh(problem, opts, x_guess)

    res_a["dSu_rel"] = float(abs(res_a["Su"] - Su_ct) / max(abs(Su_ct), 1e-30))
    res_a["T_linf_vs_ct"] = float(np.max(np.abs(res_a["T"] - T_ct)))
    res_a["u_linf_vs_ct"] = float(np.max(np.abs(res_a["u"] - u_ct)))
    res_a["Y_linf_vs_ct"] = float(np.max(np.abs(res_a["Y"] - Y_ct)))
    if res_b is not None:
        res_b["dSu_rel"] = float(abs(res_b["Su"] - Su_ct) / max(abs(Su_ct), 1e-30))
        res_b["T_linf_vs_ct"] = float(np.max(np.abs(res_b["T"] - T_ct)))
        res_b["u_linf_vs_ct"] = float(np.max(np.abs(res_b["u"] - u_ct)))
        res_b["Y_linf_vs_ct"] = float(np.max(np.abs(res_b["Y"] - Y_ct)))

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cantera_version": ct.__version__,
        "case": {
            "mech": case.mech,
            "phi": case.phi,
            "T_in": case.T_in,
            "P": case.P,
            "transport_model": case.transport_model,
            "backend": args.backend,
            "use_gpu": bool(args.use_gpu),
            "n_points_cantera_mesh": int(z_ct.size),
            "width_m": float(z_ct[-1] - z_ct[0]),
        },
        "cantera_reference": {
            "Su_m_per_s": Su_ct,
            "solve_time_s": float(t_ct),
        },
        "stageA_start_cantera_profile": {
            "ok": bool(res_a["ok"]),
            "time_s": float(res_a["time_s"]),
            "steps": int(res_a["steps"]),
            "Su_m_per_s": float(res_a["Su"]),
            "dSu_rel": float(res_a["dSu_rel"]),
            "Finf": float(res_a["Finf"]),
            "T_linf_vs_ct": float(res_a["T_linf_vs_ct"]),
            "u_linf_vs_ct": float(res_a["u_linf_vs_ct"]),
            "Y_linf_vs_ct": float(res_a["Y_linf_vs_ct"]),
        },
    }
    if res_b is not None:
        summary["start_linear_guess_same_mesh"] = {
            "ok": bool(res_b["ok"]),
            "time_s": float(res_b["time_s"]),
            "steps": int(res_b["steps"]),
            "Su_m_per_s": float(res_b["Su"]),
            "dSu_rel": float(res_b["dSu_rel"]),
            "Finf": float(res_b["Finf"]),
            "T_linf_vs_ct": float(res_b["T_linf_vs_ct"]),
            "u_linf_vs_ct": float(res_b["u_linf_vs_ct"]),
            "Y_linf_vs_ct": float(res_b["Y_linf_vs_ct"]),
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    species_names = list(problem.species_names)
    _plot_scalar(
        z_ct, T_ct, res_a["T"],
        ylabel="T [K]",
        title="Temperatura: misma malla de Cantera",
        out_path=out_dir / "stageA_temperature_vs_z.png",
        y_b=(res_b["T"] if res_b is not None else None),
    )
    _plot_scalar(
        z_ct, u_ct, res_a["u"],
        ylabel="u [m/s]",
        title="Velocidad axial: misma malla de Cantera",
        out_path=out_dir / "stageA_velocity_vs_z.png",
        y_b=(res_b["u"] if res_b is not None else None),
    )
    _plot_species(
        z_ct, species_names, Y_ct, res_a["Y"],
        out_path=out_dir / "stageA_species_vs_z.png",
        Y_b=(res_b["Y"] if res_b is not None else None),
    )
    _plot_deltas(
        z_ct, T_ct, u_ct, res_a["T"], res_a["u"],
        out_path=out_dir / "stageA_delta_profiles.png",
    )

    print(f"Reporte generado en: {out_dir.resolve()}")
    print(f"  Stage A dSu_rel = {res_a['dSu_rel']:.3e}")
    print(f"  Stage A Finf    = {res_a['Finf']:.3e}")


if __name__ == "__main__":
    main()
