"""Run Cantera vs V2 and save comparison data plus plots."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Must be set before NumPy/SciPy are imported. The V2 block solver uses many
# small dense factorizations, where OpenBLAS thread management is overhead.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# Keep the standalone comparison on the same small-grid Numba default as V2.
# An explicit NUMBA_NUM_THREADS value still takes precedence.
os.environ.setdefault("NUMBA_NUM_THREADS", "4")

import cantera as ct
import numpy as np

from config import FlameCase
from plot_saved_comparison import generate_clean_plots
from problem import FreeFlameProblem
from solver import SolveOptions, solve_free_flame
from state import unpack_state

CANTERA_REFERENCE_WORKFLOW = "direct_cantera_freeflame"
CANTERA_REFERENCE_SOLVER = "ct.FreeFlame"
V2_REFERENCE_WORKFLOW = "v2_native_freeflame"
EXTERNAL_FLAMELET_TABLE_TOOL_USED = False


def _resolve_mechanism(mech: str) -> str:
    path = Path(mech)
    if path.exists():
        return str(path.resolve())
    for folder in ct.get_data_directories():
        candidate = Path(folder) / mech
        if candidate.exists():
            return str(candidate.resolve())
    return mech


def _default_case() -> FlameCase:
    return FlameCase(
        mech=_resolve_mechanism("gri30.yaml"),
        fuel="CH4",
        oxidizer="O2:1.0, N2:3.76",
        phi=1.0,
        T_in=300.0,
        P=101325.0,
        width=0.03,
        transport_model="mixture-averaged",
        flux_gradient_basis="molar",
        soret_enabled=False,
        ratio=2.5,
        slope=0.04,
        curve=0.08,
        prune=0.003,
    )


def _cantera_solve(case: FlameCase, loglevel: int = 0, prev_ct_data: dict | None = None) -> dict:
    gas = ct.Solution(case.mech)
    gas.TP = case.T_in, case.P
    gas.set_equivalence_ratio(case.phi, case.fuel, case.oxidizer)

    if prev_ct_data is not None:
        z_old = prev_ct_data["z"]
        old_width = float(z_old[-1] - z_old[0])
        flame = ct.FreeFlame(gas, width=old_width)
    else:
        flame = ct.FreeFlame(gas, width=case.width)

    flame.transport_model = case.transport_model
    if hasattr(flame, "flux_gradient_basis"):
        flame.flux_gradient_basis = str(case.flux_gradient_basis)
    if hasattr(flame, "soret_enabled"):
        flame.soret_enabled = bool(case.soret_enabled)
    flame.set_refine_criteria(
        ratio=case.ratio,
        slope=case.slope,
        curve=case.curve,
        prune=case.prune,
    )
    flame.set_max_grid_points(flame.flame, 1600)

    if prev_ct_data is not None:
        z_old = prev_ct_data["z"]
        # Normalize locs for Cantera set_profile (0.0 to 1.0)
        z_rel = (z_old - z_old[0]) / (z_old[-1] - z_old[0])
        flame.set_profile("T", z_rel, prev_ct_data["T"])
        flame.set_profile("velocity", z_rel, prev_ct_data["u"])
        for k, sp in enumerate(gas.species_names):
            flame.set_profile(sp, z_rel, prev_ct_data["Y"][k])

    t0 = time.perf_counter()
    flame.solve(loglevel=int(loglevel), auto=True)
    elapsed = time.perf_counter() - t0

    return {
        "time_s": float(elapsed),
        "z": np.asarray(flame.grid, dtype=float),
        "u": np.asarray(flame.velocity, dtype=float),
        "T": np.asarray(flame.T, dtype=float),
        "Y": np.asarray(flame.Y, dtype=float),
        "Su": float(flame.velocity[0]),
        "width": float(flame.grid[-1] - flame.grid[0]),
        "n_points": int(flame.flame.n_points),
    }


def _v2_solve(
    case: FlameCase,
    profile: bool = True,
    verbose: bool = False,
    prev_v2_data: dict | None = None,
    compiled_block_substitution: bool = True,
    acceptance_criterion: str = "cantera",
    residual_guard_inf: float = 1.0e4,
    final_residual_inf: float | None = None,
    seed_mesh_points: int | None = None,
    solve_option_overrides: dict[str, Any] | None = None,
) -> dict:
    materials_dir = Path(__file__).resolve().parent / "materiales"
    mat_path = str(materials_dir)
    if mat_path not in sys.path:
        sys.path.insert(0, mat_path)
    from mechanism_data import load_mechanism  # type: ignore
    from species_backend_native import NativeSpeciesBackend  # type: ignore

    problem = FreeFlameProblem(case, n_points=8)
    problem.assume_finite_y = True
    mech_data = load_mechanism(case.mech)
    problem.backend_factory = lambda prob: NativeSpeciesBackend(prob, mech_data=mech_data)
    problem.backend = problem.backend_factory(problem)
    opts = SolveOptions(
        verbose=bool(verbose),
        profile=bool(profile),
        jacobian_mode="block_tridiag",
        compiled_block_substitution=bool(compiled_block_substitution),
        refine_ratio=case.ratio,
        refine_slope=case.slope,
        refine_curve=case.curve,
        refine_prune=case.prune,
        refine_max_points=1600,
        max_refine_passes=6,
        require_grid_convergence=True,
        acceptance_criterion=str(acceptance_criterion),
        weighted_step_norm_limit=1.0,
        residual_guard_inf=float(residual_guard_inf),
        max_total_time_s=300.0,
    )
    if final_residual_inf is not None:
        opts.refine_Finf_limit = float(final_residual_inf)
        opts.final_Finf_limit = float(final_residual_inf)
    if solve_option_overrides:
        for name, value in solve_option_overrides.items():
            if not hasattr(opts, name):
                raise ValueError(f"Unknown SolveOptions field: {name}")
            setattr(opts, name, value)

    if prev_v2_data is not None:
        transferred_seed = (
            _resample_v2_seed(prev_v2_data, seed_mesh_points)
            if seed_mesh_points is not None
            else prev_v2_data
        )
        x0 = transferred_seed["x"]
        z0 = transferred_seed["z"]
        problem.z = z0.copy()
        problem.n_points = len(z0)
        problem.width = float(z0[-1] - z0[0])
    else:
        x0 = None

    t0 = time.perf_counter()
    x_sol, ok, report = solve_free_flame(problem, options=opts, x0=x0)
    elapsed = time.perf_counter() - t0
    u, T, Y = unpack_state(x_sol, problem.n_points, problem.n_species)

    return {
        "ok": bool(ok),
        "time_s": float(elapsed),
        "report": report,
        "species_names": list(problem.species_names),
        "z": np.asarray(problem.z, dtype=float),
        "u": np.asarray(u, dtype=float),
        "T": np.asarray(T, dtype=float),
        "Y": np.asarray(Y, dtype=float),
        "Su": float(u[0]),
        "n_points": int(problem.n_points),
        "width": float(problem.width),
        "x": x_sol,
    }


def _interp_species(z_src: np.ndarray, Y_src: np.ndarray, z_dst: np.ndarray) -> np.ndarray:
    out = np.empty((Y_src.shape[0], z_dst.size), dtype=float)
    for k in range(Y_src.shape[0]):
        out[k] = np.interp(z_dst, z_src, Y_src[k])
    return out


def _resample_v2_seed(seed: dict, n_points: int) -> dict:
    """Transfer a certified profile without inheriting its adaptive mesh.

    Continuation needs the thermochemical state, not necessarily the mesh that
    happened to be optimal at the source parameter.  In particular, copying a
    pressure-neighbour mesh can force a thin target flame to start from an
    oversized discretization.  The target solver still performs and certifies
    its own adaptive refinement after this bounded transfer.
    """
    z_old = np.asarray(seed["z"], dtype=float)
    u_old = np.asarray(seed["u"], dtype=float)
    T_old = np.asarray(seed["T"], dtype=float)
    Y_old = np.asarray(seed["Y"], dtype=float)
    n_target = max(2, min(int(n_points), int(z_old.size)))
    # Decimate in node index rather than imposing a uniform grid.  The source
    # adaptive mesh already locates the thin reaction zone; retaining that
    # geometry avoids erasing the very feature that makes a continuation seed
    # useful, while still bounding its size.
    selected = np.rint(np.linspace(0, z_old.size - 1, n_target)).astype(int)
    selected[0], selected[-1] = 0, z_old.size - 1
    z_new = z_old[selected]
    u_new = np.interp(z_new, z_old, u_old)
    T_new = np.interp(z_new, z_old, T_old)
    Y_new = _interp_species(z_old, Y_old, z_new)
    Y_new /= np.maximum(np.sum(Y_new, axis=0, keepdims=True), 1.0e-300)
    return {
        **seed,
        "z": z_new,
        "u": u_new,
        "T": T_new,
        "Y": Y_new,
        "x": np.column_stack((u_new, T_new, Y_new.T)).ravel(),
        "n_points": int(n_target),
        "width": float(z_new[-1] - z_new[0]),
        "seed_mesh_transfer": "bounded_adaptive_subsampling",
    }


def _write_profiles(path: Path, z: np.ndarray, u: np.ndarray, T: np.ndarray) -> None:
    data = np.column_stack((z, u, T))
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="z_m,u_m_per_s,T_K",
        comments="",
    )


def run(
    output_root: Path,
    case: FlameCase | None = None,
    loglevel: int = 0,
    max_products: int = 6,
    verbose_ours: bool = False,
    prev_ct_data: dict | None = None,
    prev_v2_data: dict | None = None,
    compiled_block_substitution: bool = True,
) -> tuple[Path, dict, dict]:
    if case is None:
        case = _default_case()
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    cantera = _cantera_solve(case, loglevel=loglevel, prev_ct_data=prev_ct_data)
    ours = _v2_solve(
        case,
        profile=True,
        verbose=verbose_ours,
        prev_v2_data=prev_v2_data,
        compiled_block_substitution=compiled_block_substitution,
    )

    z_ct = cantera["z"]
    z_ours = ours["z"]
    u_ours_on_ct = np.interp(z_ct, z_ours, ours["u"])
    T_ours_on_ct = np.interp(z_ct, z_ours, ours["T"])
    Y_ours_on_ct = _interp_species(z_ours, ours["Y"], z_ct)

    summary = {
        "comparison_scope": "direct_cantera_freeflame_vs_v2_native_freeflame",
        "external_flamelet_table_tool_used": EXTERNAL_FLAMELET_TABLE_TOOL_USED,
        "case": {
            "mechanism": case.mech,
            "fuel": case.fuel,
            "oxidizer": case.oxidizer,
            "phi": case.phi,
            "T_in": case.T_in,
            "P": case.P,
            "transport_model": case.transport_model,
            "upwind_factor_v2": float(getattr(case, "upwind_factor", 1.0)),
            "refine": {
                "ratio": case.ratio,
                "slope": case.slope,
                "curve": case.curve,
                "prune": case.prune,
            },
        },
        "cantera": {
            "solver_backend": "cantera",
            "reference_workflow": CANTERA_REFERENCE_WORKFLOW,
            "reference_solver": CANTERA_REFERENCE_SOLVER,
            "solve_call": "flame.solve(auto=True)",
            "external_flamelet_table_tool_used": EXTERNAL_FLAMELET_TABLE_TOOL_USED,
            "time_s": cantera["time_s"],
            "n_points": cantera["n_points"],
            "width": cantera["width"],
            "Su": cantera["Su"],
        },
        "v2": {
            "solver_backend": "v2_native",
            "reference_workflow": V2_REFERENCE_WORKFLOW,
            "external_flamelet_table_tool_used": EXTERNAL_FLAMELET_TABLE_TOOL_USED,
            "ok": ours["ok"],
            "time_s": ours["time_s"],
            "n_points": ours["n_points"],
            "width": ours["width"],
            "Su": ours["Su"],
            "Finf_final": ours["report"].get("Finf_final"),
            "compiled_block_substitution": bool(compiled_block_substitution),
        },
        "metrics": {
            "dSu": ours["Su"] - cantera["Su"],
            "dSu_rel": abs(ours["Su"] - cantera["Su"]) / max(abs(cantera["Su"]), 1e-300),
            "speedup": cantera["time_s"] / ours["time_s"] if ours["time_s"] > 0.0 else None,
            "max_abs_dT_on_cantera_grid": float(np.max(np.abs(T_ours_on_ct - cantera["T"]))),
            "max_abs_du_on_cantera_grid": float(np.max(np.abs(u_ours_on_ct - cantera["u"]))),
        },
        "v2_report": ours["report"],
    }

    np.savez_compressed(
        run_dir / "comparison_data.npz",
        species_names=np.asarray(ours["species_names"], dtype="U"),
        has_ours=np.asarray([1], dtype=np.int8),
        z_cantera=cantera["z"],
        u_cantera=cantera["u"],
        T_cantera=cantera["T"],
        Y_cantera=cantera["Y"],
        Su_cantera=np.asarray([cantera["Su"]], dtype=float),
        z_ours=ours["z"],
        u_ours=ours["u"],
        T_ours=ours["T"],
        Y_ours=ours["Y"],
        Su_ours=np.asarray([ours["Su"]], dtype=float),
        u_ours_on_cantera=u_ours_on_ct,
        T_ours_on_cantera=T_ours_on_ct,
        Y_ours_on_cantera=Y_ours_on_ct,
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _write_profiles(run_dir / "profiles_cantera.csv", cantera["z"], cantera["u"], cantera["T"])
    _write_profiles(run_dir / "profiles_ours.csv", ours["z"], ours["u"], ours["T"])

    generate_clean_plots(run_dir, max_products=max_products, align_domains=False)
    return run_dir, cantera, ours


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and plot Cantera vs V2 comparison.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "comparison_runs",
        help="Folder where run_* comparison folders are written.",
    )
    parser.add_argument("--loglevel", type=int, default=0, help="Cantera solve loglevel.")
    parser.add_argument("--max-products", type=int, default=6)
    parser.add_argument(
        "--verbose-ours",
        action="store_true",
        help="Print detailed progress from the V2 solver.",
    )
    parser.add_argument(
        "--compiled-block-substitution",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the fused compiled substitution for the block-tridiagonal LU.",
    )
    args = parser.parse_args()

    case = _default_case()
    run_dir, _, _ = run(
        args.output_root,
        case=case,
        loglevel=args.loglevel,
        max_products=args.max_products,
        verbose_ours=args.verbose_ours,
        compiled_block_substitution=args.compiled_block_substitution,
    )
    print(run_dir.resolve())


if __name__ == "__main__":
    main()
