"""
generate_fgm_tables_native.py  (v2 - Z-C space)

Genera tablas FGM usando V2 con backend nativo CPU.
Ahora almacena la fracción de mezcla de Bilger Z para cada flamelet,
de modo que la tabla queda parametrizada en (Z, c) en lugar de (phi, c).

Para llamas premezcladas Z es constante a lo largo del flamelet y depende
únicamente de phi mediante la definición de Bilger.

Uso:
  python generate_fgm_tables_native.py --phi-min 0.7 --phi-max 1.4 --n-phi 5 \
         --save-raw-profiles
"""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The 1-D block solver factors many small (55 x 55 for GRI30) dense blocks.
# OpenBLAS thread-management overhead dominates those calls on this machine;
# one BLAS thread is faster while Numba retains its independent chemistry pool.
# Respect an explicit user setting for other hardware or workloads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

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


_CONTINUATION_MAX_PHI_RATIO = 1.15
V2_REFERENCE_WORKFLOW = "v2_native_freeflame"
V2_REFERENCE_SOLVER = "solve_free_flame"
EXTERNAL_FLAMELET_TABLE_TOOL_USED = False
EXTERNAL_FLAMELET_TABLE_TOOL = None
TABLE_BUILDER = "repo_fgm_common"


def _is_local_phi_step(
    phi_from: float,
    phi_to: float,
    max_ratio: float = _CONTINUATION_MAX_PHI_RATIO,
) -> bool:
    """Return whether a multiplicative phi step is safe for continuation."""
    if phi_from <= 0.0 or phi_to <= 0.0:
        return False
    if not np.isfinite(max_ratio) or max_ratio <= 0.0:
        return True
    max_ratio = max(float(max_ratio), 1.0)
    ratio = float(phi_to) / float(phi_from)
    return (1.0 / max_ratio) <= ratio <= max_ratio


v2_path = Path(__file__).resolve().parent.parent.parent / "V2"
v2_dir = str(v2_path)
if v2_dir not in sys.path:
    sys.path.insert(0, v2_dir)
materials_dir = str(v2_path / "materiales")
if materials_dir not in sys.path:
    sys.path.insert(0, materials_dir)

from run_saved_comparison import FlameCase, _resolve_mechanism
from solver import solve_free_flame, SolveOptions
from problem import FreeFlameProblem
from state import interpolate_state, pack_state, unpack_state
from mechanism_data import load_mechanism
from species_backend_native import NativeSpeciesBackend


def configure_numba_kinetics_threads(count: int) -> int:
    """Set the Numba pool size used by the batched native chemistry kernel."""
    try:
        import numba
        available = int(numba.config.NUMBA_NUM_THREADS)
        resolved = max(1, min(int(count), available))
        numba.set_num_threads(resolved)
        return resolved
    except Exception:
        return 1


# ---------------------------------------------------------------------------
# Dataclass de registro por flamelet
# ---------------------------------------------------------------------------

@dataclass
class FlameRecord:
    phi: float
    Z: float
    solve_ok: bool
    residual_inf: float
    weighted_step_norm: float
    final_accepted: bool
    acceptance_criterion: str
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


def load_seed_profile_npz(path: Path, n_species: int, source: str = "auto") -> dict[str, np.ndarray]:
    data = np.load(path)
    source_l = str(source).strip().lower()

    candidates: list[tuple[str, tuple[str, str, str, str]]] = []
    if source_l in ("auto", "raw"):
        candidates.append(("raw", ("z", "u", "T", "Y")))
    if source_l in ("auto", "ours", "v2"):
        candidates.append(("ours", ("z_ours", "u_ours", "T_ours", "Y_ours")))
    if source_l in ("auto", "cantera", "ct"):
        candidates.append(("cantera", ("z_cantera", "u_cantera", "T_cantera", "Y_cantera")))

    for _name, keys in candidates:
        if all(k in data.files for k in keys):
            z = np.asarray(data[keys[0]], dtype=float)
            u = np.asarray(data[keys[1]], dtype=float)
            T = np.asarray(data[keys[2]], dtype=float)
            Y = np.asarray(data[keys[3]], dtype=float)
            if Y.shape != (n_species, z.size):
                raise ValueError(
                    f"Semilla {path} incompatible: Y.shape={Y.shape}, "
                    f"esperado=({n_species}, {z.size})."
                )
            if u.shape != (z.size,) or T.shape != (z.size,):
                raise ValueError(
                    f"Semilla {path} incompatible: z/u/T no tienen la misma longitud."
                )
            out = {"z": z.copy(), "x": pack_state(u, T, Y)}
            if "phi" in data.files:
                out["phi"] = np.asarray(data["phi"], dtype=float).copy()
            return out

    available = ", ".join(data.files)
    raise ValueError(
        f"No pude leer semilla '{source}' desde {path}. "
        f"Claves disponibles: {available}"
    )


def build_continuation_seed(
    problem: FreeFlameProblem,
    phi: float,
    prev_solution: dict[str, np.ndarray] | None,
    prev_prev_solution: dict[str, np.ndarray] | None,
    use_predictor: bool,
    predictor_damping: float,
    trust_ratio: float = _CONTINUATION_MAX_PHI_RATIO,
) -> dict[str, np.ndarray] | None:
    if prev_solution is None or "z" not in prev_solution or "x" not in prev_solution:
        return None

    # Parameter continuation is a local method. A distant flame profile can
    # send the nonlinear solve through a much denser adaptive mesh than a cold
    # start. Keep the seed only inside the configured multiplicative trust
    # region; a non-positive value is reserved for the unbounded ablation.
    if "phi" in prev_solution:
        phi_prev = float(np.asarray(prev_solution["phi"]).ravel()[0])
        if not _is_local_phi_step(phi_prev, float(phi), trust_ratio):
            return None

    z_prev = np.asarray(prev_solution.get("z"), dtype=float)
    x_prev = np.asarray(prev_solution.get("x"), dtype=float)
    n_sp = int(problem.n_species)
    expected = int(z_prev.size) * (2 + n_sp)
    if z_prev.ndim != 1 or z_prev.size < 2 or x_prev.size != expected:
        return None

    x_seed = x_prev.copy()
    if (
        use_predictor
        and prev_prev_solution is not None
        and "z" in prev_prev_solution
        and "x" in prev_prev_solution
        and "phi" in prev_prev_solution
        and "phi" in prev_solution
    ):
        phi0 = float(np.asarray(prev_prev_solution.get("phi")).ravel()[0])
        phi1 = float(np.asarray(prev_solution.get("phi")).ravel()[0])
        dphi = phi1 - phi0
        if abs(dphi) > 1.0e-14 and _is_local_phi_step(phi0, phi1, trust_ratio):
            z0 = np.asarray(prev_prev_solution.get("z"), dtype=float)
            x0 = np.asarray(prev_prev_solution.get("x"), dtype=float)
            expected0 = int(z0.size) * (2 + n_sp)
            if z0.ndim == 1 and z0.size >= 2 and x0.size == expected0:
                x0_on_prev = interpolate_state(x0, z0, z_prev, n_sp)
                factor = (float(phi) - phi1) / dphi
                factor *= float(np.clip(predictor_damping, 0.0, 1.0))
                factor = float(np.clip(factor, -0.5, 1.0))
                x_seed = x_prev + factor * (x_prev - x0_on_prev)

    u, T, Y = unpack_state(x_seed, int(z_prev.size), n_sp)

    T = np.clip(T, float(problem.T_lower_bound), float(problem.T_upper_bound))
    Y = np.clip(Y, 0.0, None)
    sums = Y.sum(axis=0, keepdims=True)
    sums = np.where(sums > 0.0, sums, 1.0)
    Y = Y / sums

    # Current inlet composition is a hard boundary condition.
    u[0] = max(float(u[0]), 1.0e-8)
    T[0] = float(problem.T_in)
    Y[:, 0] = np.asarray(problem.Y_in, dtype=float)

    return {
        "phi": np.array([float(phi)], dtype=float),
        "z": z_prev.copy(),
        "x": pack_state(u, T, Y),
    }


def seed_cache_directory(args: argparse.Namespace) -> Path:
    cache_dir = str(getattr(args, "seed_cache_dir", "")).strip()
    if cache_dir:
        return Path(cache_dir).expanduser()
    return Path(args.output_root).expanduser() / "_v2_seed_cache"


def seed_cache_stats(cache_dir: Path) -> tuple[int, float]:
    """Return the number and size in MiB of cached seed profiles."""
    if not cache_dir.exists():
        return 0, 0.0
    files = [p for p in cache_dir.glob("*.npz") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in files)
    return len(files), float(total_bytes) / (1024.0 * 1024.0)


def seed_cache_key(args: argparse.Namespace, resolved_mech: str) -> str:
    payload = {
        "mech": str(Path(resolved_mech).resolve()) if Path(resolved_mech).exists() else str(resolved_mech),
        "fuel": str(args.fuel),
        "oxidizer": str(args.oxidizer),
        "transport_model": str(args.transport_model),
        "flux_gradient_basis": str(args.flux_gradient_basis),
        "soret_enabled": bool(args.soret_enabled),
        "outlet_species_bc": str(args.outlet_species_bc),
        "upwind_factor": float(args.upwind_factor),
        "T_in": float(args.T_in),
        "P": float(args.P),
        "width": float(args.width),
        "ratio": float(args.ratio),
        "slope": float(args.slope),
        "curve": float(args.curve),
        "prune": float(args.prune),
        "max_grid_points": int(args.max_grid_points),
        # A seed is valid as an accelerator only when its mesh-certification
        # policy is compatible with the requested calculation.
        "max_refine_passes": int(args.max_refine_passes),
        "require_grid_convergence": bool(args.require_grid_convergence),
        "tight_refine_passes": int(args.tight_refine_passes),
        "tight_ratio": float(args.tight_ratio),
        "tight_slope": float(args.tight_slope),
        "tight_curve": float(args.tight_curve),
        "tight_prune": float(args.tight_prune),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def seed_cache_path(args: argparse.Namespace, resolved_mech: str, phi: float) -> Path:
    tag = f"phi_{float(phi):.6f}".replace(".", "p").replace("-", "m")
    return seed_cache_directory(args) / f"{seed_cache_key(args, resolved_mech)}_{tag}.npz"


def maybe_load_cached_seed(
    args: argparse.Namespace,
    resolved_mech: str,
    phi: float,
    n_species: int,
) -> tuple[dict[str, np.ndarray] | None, Path | None]:
    if bool(getattr(args, "disable_seed_cache", False)):
        return None, None
    path = seed_cache_path(args, resolved_mech, phi)
    if not path.exists():
        return None, None
    try:
        return load_seed_profile_npz(path, n_species=n_species, source="raw"), path
    except Exception as exc:
        print(f"seed cache skip : {path} ({exc})")
        return None, None


def write_cached_seed(
    args: argparse.Namespace,
    resolved_mech: str,
    rec: FlameRecord,
    species_names: list[str],
) -> Path | None:
    if bool(getattr(args, "disable_seed_cache", False)):
        return None
    path = seed_cache_path(args, resolved_mech, rec.phi)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        phi=np.array([rec.phi], dtype=float),
        Z=np.array([rec.Z], dtype=float),
        z=rec.z,
        u=rec.u,
        T=rec.T,
        Y=rec.Y,
        species_names=np.array(species_names, dtype=object),
        Su_m_per_s=np.array([rec.Su_m_per_s], dtype=float),
        residual_inf=np.array([rec.residual_inf], dtype=float),
        weighted_step_norm=np.array([rec.weighted_step_norm], dtype=float),
    )
    return path


def make_solve_options(args: argparse.Namespace) -> SolveOptions:
    opts = SolveOptions(
        verbose=(args.loglevel > 0),
        profile=False,
        max_total_time_s=float(args.max_flame_time_s),
    )
    opts.refine_Finf_limit = float(args.max_residual_inf)
    opts.final_Finf_limit = float(args.max_residual_inf)
    opts.acceptance_criterion = str(args.acceptance_criterion).strip().lower()
    opts.weighted_step_norm_limit = float(args.max_weighted_step_norm)
    opts.residual_guard_inf = float(args.residual_guard_inf)
    opts.accept_residual_converged = (opts.acceptance_criterion == "residual")
    opts.max_jac_age = int(args.max_jac_age)
    opts.damp_factor = float(args.damp_factor)
    opts.jac_threshold = float(args.jac_threshold)
    opts.jacobian_mode = str(args.jacobian_mode)
    opts.precompute_jacobian_thermo = bool(args.precompute_jacobian_thermo)
    opts.compiled_block_substitution = bool(args.compiled_block_substitution)
    opts.max_refine_passes = int(args.max_refine_passes)
    opts.require_grid_convergence = bool(args.require_grid_convergence)
    opts.auto_bootstrap_grids = bool(args.auto_bootstrap_grids)
    opts.restart_insert_anchor = bool(args.restart_insert_anchor)
    return opts


# ---------------------------------------------------------------------------
# Resolución de flamelet con V2 / backend nativo CPU
# ---------------------------------------------------------------------------

def solve_flame_native(
    phi: float,
    args: argparse.Namespace,
    mech_data: Any,
    opts: SolveOptions,
    progress_weights: dict[str, float],
    prev_solution: dict[str, np.ndarray] | None = None,
    prev_prev_solution: dict[str, np.ndarray] | None = None,
) -> tuple[FlameRecord, Any]:
    gas = ct.Solution(args.mech)
    gas.TP = float(args.T_in), float(args.P)
    gas.set_equivalence_ratio(float(phi), args.fuel, args.oxidizer)
    Z = float(gas.mixture_fraction(args.fuel, args.oxidizer, basis="mass"))

    flame_case = FlameCase(
        mech=_resolve_mechanism(args.mech),
        fuel=args.fuel,
        oxidizer=args.oxidizer,
        phi=float(phi),
        T_in=args.T_in,
        P=args.P,
        width=args.width,
        transport_model=args.transport_model,
        flux_gradient_basis=args.flux_gradient_basis,
        soret_enabled=args.soret_enabled,
        outlet_species_bc=args.outlet_species_bc,
        upwind_factor=float(args.upwind_factor),
        ratio=args.ratio,
        slope=args.slope,
        curve=args.curve,
        prune=args.prune,
    )
    opts.refine_ratio = args.ratio
    opts.refine_slope = args.slope
    opts.refine_curve = args.curve
    opts.refine_prune = args.prune
    opts.refine_max_points = int(args.max_grid_points)
    opts.max_refine_passes = int(args.max_refine_passes)
    opts.require_grid_convergence = bool(args.require_grid_convergence)
    opts.refine_Finf_limit = float(args.max_residual_inf)
    opts.final_Finf_limit = float(args.max_residual_inf)
    opts.acceptance_criterion = str(args.acceptance_criterion).strip().lower()
    opts.weighted_step_norm_limit = float(args.max_weighted_step_norm)
    opts.residual_guard_inf = float(args.residual_guard_inf)
    opts.accept_residual_converged = (opts.acceptance_criterion == "residual")
    opts.restart_insert_anchor = bool(args.restart_insert_anchor)
    if float(args.grid_min) > 0.0:
        opts.refine_grid_min = float(args.grid_min)

    def run_once(jacobian_mode: str):
        problem = FreeFlameProblem(
            flame_case, n_points=max(2, int(args.initial_grid_points))
        )
        problem.assume_finite_y = True
        problem.backend_factory = lambda prob: NativeSpeciesBackend(prob, mech_data=mech_data)

        # Keep cold-start Jacobian aging conservative.  A converged profile
        # is much closer to the next phi state, where a longer reuse window
        # avoids costly rebuilds without degrading the certified result.
        run_opts = copy.copy(opts)
        seed_solution = build_continuation_seed(
            problem=problem,
            phi=float(phi),
            prev_solution=prev_solution,
            prev_prev_solution=prev_prev_solution,
            use_predictor=not bool(args.disable_continuation_predictor),
            predictor_damping=float(args.continuation_predictor_damping),
            trust_ratio=float(args.continuation_trust_ratio),
        )

        x0 = None
        if seed_solution is not None and "z" in seed_solution and "x" in seed_solution:
            z_prev = np.asarray(seed_solution.get("z"), dtype=float)
            x_prev = np.asarray(seed_solution.get("x"), dtype=float)
            expected = int(z_prev.size) * (2 + problem.n_species)
            if z_prev.ndim == 1 and z_prev.size >= 2 and x_prev.size == expected:
                problem.z = z_prev.copy()
                problem.n_points = int(z_prev.size)
                problem.width = float(z_prev[-1] - z_prev[0])
                x0 = x_prev.copy()

        if x0 is not None:
            continuation_jac_age = int(args.continuation_max_jac_age)
            if continuation_jac_age > 0:
                run_opts.max_jac_age = continuation_jac_age

        problem.backend = problem.backend_factory(problem)
        run_opts.jacobian_mode = str(jacobian_mode)
        t0 = time.perf_counter()
        x_sol, solve_ok, report = solve_free_flame(problem, options=run_opts, x0=x0)
        report["max_jac_age_used"] = int(run_opts.max_jac_age)
        base_report = report

        # First obtain a robust continuation state with the requested criteria.
        # When requested, a second stage enforces the tighter mesh and refuses
        # to label a grid capped by max_refine_passes as certified.
        tight_passes = int(getattr(args, "tight_refine_passes", 0))
        if bool(solve_ok) and tight_passes > 0:
            tight_opts = copy.copy(run_opts)
            tight_opts.refine_ratio = float(args.tight_ratio)
            tight_opts.refine_slope = float(args.tight_slope)
            tight_opts.refine_curve = float(args.tight_curve)
            tight_opts.refine_prune = float(args.tight_prune)
            # A tight stage is a new adaptive problem, not a single cosmetic
            # pass.  Give it at least the normal refinement budget; otherwise
            # ``--tight-refine-passes 1 --require-grid-convergence`` rejects a
            # valid flame merely because the first strict grid still has marks.
            tight_opts.max_refine_passes = max(
                tight_passes, int(args.max_refine_passes)
            )
            tight_opts.require_grid_convergence = bool(
                args.require_grid_convergence
            )
            x_sol, solve_ok, tight_report = solve_free_flame(
                problem, options=tight_opts, x0=x_sol
            )
            report = tight_report
            report["tight_refinement"] = {
                "requested": True,
                "criteria": {
                    "ratio": tight_opts.refine_ratio,
                    "slope": tight_opts.refine_slope,
                    "curve": tight_opts.refine_curve,
                    "prune": tight_opts.refine_prune,
                    "max_passes": tight_opts.max_refine_passes,
                },
                "base_solved": bool(base_report.get("solved", False)),
                "base_n_points": int(base_report.get("n_points_final", 0)),
            }
        dt = float(time.perf_counter() - t0)
        report["jacobian_mode"] = str(jacobian_mode)
        return problem, x_sol, bool(solve_ok), report, dt

    requested_mode = str(args.jacobian_mode).strip().lower()
    modes = [requested_mode]
    attempts: list[dict[str, Any]] = []
    t_total0 = time.perf_counter()
    problem = None
    x_sol = None
    solve_ok = False
    report: dict[str, Any] = {}
    residual_inf = float("nan")
    weighted_step_norm = float("nan")
    final_accepted = False
    acceptance_criterion = str(args.acceptance_criterion).strip().lower()

    for attempt_i, mode in enumerate(modes):
        problem, x_sol, solve_ok, report, dt_attempt = run_once(mode)
        residual_inf = float(report.get("Finf_final", np.nan))
        weighted_step_norm = float(report.get("weighted_step_norm_final", np.nan))
        final_accepted = bool(report.get("final_accepted", solve_ok))
        acceptance_criterion = str(report.get("acceptance_criterion", acceptance_criterion))
        accepted = bool(solve_ok) and final_accepted
        attempts.append({
            "jacobian_mode": mode,
            "solve_ok": bool(solve_ok),
            "final_accepted": bool(final_accepted),
            "acceptance_criterion": acceptance_criterion,
            "residual_inf": residual_inf,
            "weighted_step_norm": weighted_step_norm,
            "solve_time_s": dt_attempt,
            "accepted": accepted,
        })
        if accepted or attempt_i == len(modes) - 1:
            break

    if report is not None:
        report["jacobian_attempts"] = attempts
    dt = float(time.perf_counter() - t_total0)
    assert problem is not None and x_sol is not None

    # Extraer variables de solucion V2 (organizadas como [U, T, Y0...Yk])
    nv = 2 + problem.n_species
    x_reshaped = x_sol.reshape(-1, nv)
    u = np.asarray(x_reshaped[:, 0], dtype=float)
    T = np.asarray(x_reshaped[:, 1], dtype=float)
    Y = np.asarray(x_reshaped[:, 2:], dtype=float).T # Shape: (n_species, n_points)
    z = np.asarray(problem.z, dtype=float)

    # Posprocesado común de los campos tabulados. Cantera no interviene en la
    # convergencia V2; se usa aquí para evaluar todos los campos con la misma
    # convención que la referencia.
    rho = np.zeros_like(T)
    cp_mass = np.zeros_like(T)
    conductivity = np.zeros_like(T)
    qdot = np.zeros_like(T)
    omega_beta = np.zeros_like(T)
    species_index = {name: k for k, name in enumerate(gas.species_names)}

    for j in range(z.size):
        gas.TPY = T[j], args.P, Y[:, j]
        rho[j] = gas.density
        cp_mass[j] = gas.cp_mass
        conductivity[j] = gas.thermal_conductivity
        qdot[j] = gas.heat_release_rate
        net_rates_mass = gas.net_production_rates * gas.molecular_weights
        for species, weight in progress_weights.items():
            if species in species_index:
                omega_beta[j] += float(weight) * net_rates_mass[species_index[species]]

    c, beta, _ = compute_progress_variable(
        species_names=list(gas.species_names),
        Y=Y,
        T=T,
        progress_weights=progress_weights,
    )
    beta_span = float(beta[-1] - beta[0])
    if abs(beta_span) <= 1.0e-14:
        raise RuntimeError(
            f"La variable de progreso no define una fuente normalizada en phi={phi:g}."
        )
    omega_c = omega_beta / beta_span

    rec = FlameRecord(
        phi=float(phi),
        Z=Z,
        solve_ok=bool(solve_ok),
        residual_inf=residual_inf,
        weighted_step_norm=weighted_step_norm,
        final_accepted=bool(final_accepted),
        acceptance_criterion=acceptance_criterion,
        solve_time_s=dt,
        n_points=int(z.size),
        width_m=float(z[-1] - z[0]),
        Su_m_per_s=float(u[0]),
        z=z, u=u, T=T, rho=rho, cp_mass=cp_mass,
        conductivity=conductivity, qdot=qdot, omega_c=omega_c,
        Y=Y, c=c, beta=beta,
    )
    next_solution = {
        "phi": np.array([float(phi)], dtype=float),
        "z": z.copy(),
        "x": np.asarray(x_sol, dtype=float).copy(),
    }
    return rec, next_solution


def solve_flame_from_seed_cache_worker(payload: tuple[int, float, str, argparse.Namespace]):
    idx, phi, seed_path_text, args = payload
    configure_numba_kinetics_threads(
        int(getattr(args, "numba_kinetics_threads_resolved", 1))
    )
    resolved_mech = _resolve_mechanism(args.mech)
    mech_data = load_mechanism(resolved_mech)
    opts = make_solve_options(args)
    progress_weights = parse_progress_weights(args.progress_species)
    n_species = ct.Solution(resolved_mech).n_species
    seed_solution = load_seed_profile_npz(
        Path(seed_path_text), n_species=n_species, source="raw"
    )
    rec, _ = solve_flame_native(
        phi=float(phi),
        args=args,
        mech_data=mech_data,
        opts=opts,
        progress_weights=progress_weights,
        prev_solution=seed_solution,
        prev_prev_solution=None,
    )
    return idx, rec


# ---------------------------------------------------------------------------
# Escritura de resultados
# ---------------------------------------------------------------------------

def write_summary_csv(path: Path, records: list[FlameRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow([
            "phi", "Z", "solve_ok", "final_accepted", "acceptance_criterion",
            "residual_inf", "weighted_step_norm", "Su_m_per_s",
            "n_points", "width_m", "solve_time_s",
        ])
        for rec in records:
            wr.writerow([
                rec.phi, rec.Z, rec.solve_ok, rec.final_accepted,
                rec.acceptance_criterion, rec.residual_inf, rec.weighted_step_norm,
                rec.Su_m_per_s, rec.n_points, rec.width_m, rec.solve_time_s,
            ])


def write_raw_profiles(out_dir: Path, records: list[FlameRecord], species_names: list[str]) -> None:
    raw_dir = out_dir / "raw_profiles"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        tag = f"phi_{rec.phi:.6f}".replace(".", "p")
        np.savez_compressed(
            raw_dir / f"{tag}.npz",
            phi=np.array([rec.phi], dtype=float),
            Z=np.array([rec.Z], dtype=float),
            solve_ok=np.array([rec.solve_ok], dtype=bool),
            final_accepted=np.array([rec.final_accepted], dtype=bool),
            acceptance_criterion=np.array([rec.acceptance_criterion], dtype=object),
            residual_inf=np.array([rec.residual_inf], dtype=float),
            weighted_step_norm=np.array([rec.weighted_step_norm], dtype=float),
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
    solve_ok = np.array([r.solve_ok for r in records], dtype=bool)
    residual_inf = np.array([r.residual_inf for r in records], dtype=float)
    weighted_step_norm = np.array([r.weighted_step_norm for r in records], dtype=float)
    final_accepted = np.array([r.final_accepted for r in records], dtype=bool)
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
        "Su": Su, "solve_ok": solve_ok, "final_accepted": final_accepted,
        "residual_inf": residual_inf, "weighted_step_norm": weighted_step_norm,
        "n_points": n_points, "width": width, "solve_time": solve_time,
        "T": T_tab, "u": u_tab, "rho": rho_tab, "cp_mass": cp_tab,
        "conductivity": conductivity_tab, "qdot": qdot_tab,
        "omega_c": omega_c_tab, "beta": beta_tab, "Y": Y_tab,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Genera tablas FGM en espacio (Z, c) con V2 nativo CPU."
    )
    p.add_argument("--mech",             type=str,   default="gri30.yaml")
    p.add_argument("--fuel",             type=str,   default="CH4")
    p.add_argument("--oxidizer",         type=str,   default="O2:1.0, N2:3.76")
    p.add_argument("--transport-model",  type=str,   default="mixture-averaged")
    p.add_argument("--flux-gradient-basis", type=str, default="molar")
    p.add_argument("--soret-enabled",    action="store_true")
    p.add_argument("--outlet-species-bc", type=str, default="zero_gradient",
                   choices=("zero_gradient", "cantera_flux"),
                   help="Salida de especies: zero_gradient reproduce Outlet de Cantera FreeFlame; cantera_flux usa la condicion cruda de Flow1D.")
    p.add_argument("--upwind-factor", type=float, default=1.0,
                   help="Peso convectivo: 1.0=upwind puro, 0.0=central; experimental para ensayar esquemas tipo Lapointe.")
    p.add_argument("--T-in",             type=float, default=300.0)
    p.add_argument("--P",                type=float, default=101325.0)
    p.add_argument("--width",            type=float, default=0.03)
    p.add_argument("--initial-grid-points", type=int, default=8,
                   help="Nodos del arranque frío; 12 es un modo rápido validado en phi=0.9–1.1."
                   )

    p.add_argument("--phi-min",    type=float, default=0.7)
    p.add_argument("--phi-max",    type=float, default=1.4)
    p.add_argument("--n-phi",      type=int,   default=5)
    p.add_argument("--phi-values", type=str,   default="",
                   help="Lista manual separada por comas. Ignora phi-min/max/n-phi.")
    p.add_argument("--use-z-grid", action="store_true",
                   help="Barrer por Z objetivo en lugar de phi.")
    p.add_argument("--z-min",      type=float, default=0.03)
    p.add_argument("--z-max",      type=float, default=0.08)
    p.add_argument("--n-z",        type=int,   default=5)
    p.add_argument("--z-values",   type=str,   default="",
                   help="Lista manual de Z separada por comas. Ignora z-min/max/n-z.")
    p.add_argument("--phi-bracket-min", type=float, default=1e-4,
                   help="Límite inferior de phi para invertir Z->phi.")
    p.add_argument("--phi-bracket-max", type=float, default=1e3,
                   help="Límite superior de phi para invertir Z->phi.")

    # Refinamiento tipo Cantera
    p.add_argument("--ratio",             type=float, default=2.5)
    p.add_argument("--slope",             type=float, default=0.04)
    p.add_argument("--curve",             type=float, default=0.08)
    p.add_argument("--prune",             type=float, default=0.003)
    p.add_argument("--max-grid-points",   type=int,   default=1600)
    p.add_argument("--max-refine-passes", type=int,   default=6,
                   help="Máximo de ciclos solve/refine por flamelet.")
    p.add_argument(
        "--require-grid-convergence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Rechaza el flamelet si llega al máximo de refinamientos sin "
            "converger la malla (activo por defecto)."
        ),
    )
    p.add_argument("--grid-min",          type=float, default=0.0)
    p.add_argument(
        "--tight-refine-passes",
        type=int,
        default=0,
        help=(
            "Activa una segunda etapa con los criterios tight; su presupuesto "
            "es al menos --max-refine-passes. Cero la desactiva."
        ),
    )
    p.add_argument("--tight-ratio",       type=float, default=2.5)
    p.add_argument("--tight-slope",       type=float, default=0.04)
    p.add_argument("--tight-curve",       type=float, default=0.08)
    p.add_argument("--tight-prune",       type=float, default=0.003)
    p.add_argument("--loglevel",          type=int,   default=0)
    p.add_argument("--max-residual-inf",  type=float, default=10.0,
                   help="Maximo ||F||inf para criterio residual estricto/diagnostico.")
    p.add_argument("--acceptance-criterion", type=str, default="cantera",
                   choices=("cantera", "residual", "combined"),
                   help="Criterio final: cantera usa norma ponderada del paso Newton.")
    p.add_argument("--max-weighted-step-norm", type=float, default=1.0,
                   help="Limite de norma ponderada del paso Newton para criterio cantera.")
    p.add_argument("--residual-guard-inf", type=float, default=1.0e4,
                   help="Guardia de ||F||inf bruto usada con criterio cantera.")
    p.add_argument("--max-flame-time-s", type=float, default=300.0,
                   help="Tiempo maximo por flamelet en el solver V2.")
    p.add_argument("--max-jac-age", type=int, default=20,
                   help="Numero maximo de pasos Newton reutilizando el Jacobiano.")
    p.add_argument("--damp-factor", type=float, default=float(np.sqrt(2.0)),
                   help="Factor de backtracking Newton; sqrt(2) reproduce el valor de Cantera.")
    p.add_argument("--continuation-max-jac-age", type=int, default=20,
                   help="Edad del Jacobiano con una semilla convergida; 20 es el valor validado en el barrido FGM frio; 0 conserva --max-jac-age.")
    p.add_argument("--jac-threshold", type=float, default=0.0,
                   help="Umbral para descartar entradas pequenas del Jacobiano.")
    p.add_argument("--jacobian-mode", type=str, default="block_tridiag",
                   choices=("numba_local", "banded_lapack", "block_tridiag", "cantera_local"),
                   help="Backend del Jacobiano usado por el solver V2.")
    p.add_argument("--precompute-jacobian-thermo", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Precalcula termoquimica perturbada de todo el Jacobiano block_tridiag.")
    p.add_argument(
        "--compiled-block-substitution",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fusiona en Numba las sustituciones de la LU block-tridiagonal.",
    )
    p.add_argument("--allow-failed-flamelets", action="store_true",
                   help="Permite guardar la tabla aunque algun flamelet no converja.")
    p.add_argument("--disable-continuation", action="store_true",
                   help="Resuelve cada flamelet desde la conjetura inicial base.")
    p.add_argument("--disable-continuation-predictor", action="store_true",
                   help="Desactiva predictor secante entre flamelets.")
    p.add_argument("--continuation-predictor-damping", type=float, default=0.7,
                   help="Amortiguamiento del predictor secante en phi.")
    p.add_argument(
        "--continuation-trust-ratio",
        type=float,
        default=_CONTINUATION_MAX_PHI_RATIO,
        help=(
            "Razon multiplicativa maxima para reutilizar una semilla en phi; "
            "0 desactiva solo esta cota para la ablacion no acotada."
        ),
    )
    p.add_argument("--restart-insert-anchor", action="store_true",
                   help="Inserta un punto exacto de ancla de T tambien en reinicios.")
    p.add_argument("--auto-bootstrap-grids", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Activa los grids fijos intermedios 12/24/48 antes del refinamiento adaptativo; es una ruta diagnostica mas lenta en FGM frio.")
    p.add_argument("--seed-profile-npz", type=str, default="",
                   help="Perfil .npz para sembrar el primer flamelet (raw profile o comparison_data).")
    p.add_argument("--seed-profile-source", type=str, default="auto",
                   choices=("auto", "raw", "ours", "v2", "cantera", "ct"),
                   help="Fuente dentro del .npz usado como semilla inicial.")
    p.add_argument("--disable-seed-cache", action="store_true",
                   help="No lee ni escribe cache automatica de perfiles semilla.")
    p.add_argument("--seed-cache-dir", type=str, default="",
                   help="Directorio de cache de perfiles semilla; default: output-root/_v2_seed_cache.")
    p.add_argument("--parallel-workers", type=int, default=0,
                   help="Procesos paralelos desde seed cache; 0 usa auto conservador.")
    p.add_argument("--numba-kinetics-threads", type=int, default=0,
                   help="Hilos Numba por proceso; 0 divide los CPU entre workers.")

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
    if not 0.0 <= float(args.upwind_factor) <= 1.0:
        raise SystemExit("--upwind-factor debe estar entre 0.0 y 1.0")
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
    run_name = args.run_name.strip() or f"run_{timestamp}_fgm_native"
    out_dir = Path(args.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Previsualizar mapa phi → Z
    print("=" * 70)
    print("FGM TABLE GENERATOR (V2 native CPU) - Z-C space")
    print("=" * 70)
    print(f"Cantera version : {ct.__version__}")
    print(f"Reference flow  : {V2_REFERENCE_WORKFLOW}")
    print("External tool   : none (Streamline Flamelet Table Tool not invoked)")
    print(f"Output          : {out_dir.resolve()}")
    print(
        "refine          : "
        f"ratio={args.ratio:g}, slope={args.slope:g}, "
        f"curve={args.curve:g}, prune={args.prune:g}, "
        f"max_points={args.max_grid_points}"
    )
    print(
        "solver          : "
        f"initial_grid={args.initial_grid_points}, "
        "transient=PTC-SER/BE-fallback, "
        f"damp=step_norm/{args.damp_factor:g}, "
        f"upwind_factor={args.upwind_factor:g}, "
        f"jacobian={args.jacobian_mode}, "
        "transient_linear=direct"
    )
    print(f"linear BLAS     : {os.environ.get('OPENBLAS_NUM_THREADS', 'auto')} thread(s)")
    print(
        "table           : "
        f"n_phi={len(phi_vals)}, n_c={args.n_c}, "
        f"c_fine={args.c_fine}, bias={args.refine_bias:g}"
    )
    print(f"mode            : {'Z-grid' if z_mode else 'phi-grid'}")
    print(f"phis            : {phi_vals}")
    if z_mode:
        print(f"Z target        : {[f'{z:.4f}' for z in z_targets]}")
    print(f"Z (Bilger)      : {[f'{z:.4f}' for z in Z_preview]}")
    print(f"Z_st (phi=1)    : {Z_st:.6f}")

    records: list[FlameRecord] = []
    species_names: list[str] | None = None
    used_progress_species: list[str] = []

    # === V2 NATIVE BACKEND SETUP ===
    print("Inicializando backend nativo para el barrido...")
    resolved_mech = _resolve_mechanism(args.mech)
    mech_data = load_mechanism(resolved_mech)
    opts = make_solve_options(args)
    prev_solution: dict[str, np.ndarray] | None = None
    prev_prev_solution: dict[str, np.ndarray] | None = None
    cache_dir = seed_cache_directory(args)
    cache_files0, cache_mb0 = seed_cache_stats(cache_dir)
    seed_cache_used = ""
    seed_cache_used_paths: list[str] = []
    seed_cache_written: list[str] = []
    requested_parallel_workers = int(args.parallel_workers)
    if requested_parallel_workers <= 0:
        parallel_workers = min(4, max(1, (os.cpu_count() or 2) - 1))
    else:
        parallel_workers = max(1, requested_parallel_workers)
    parallel_used = False
    parallel_seed_paths: list[Path] = []
    if (
        parallel_workers > 1
        and len(phi_vals) > 1
        and not bool(args.disable_seed_cache)
        and not args.seed_profile_npz.strip()
    ):
        candidate_paths = [
            seed_cache_path(args, resolved_mech=resolved_mech, phi=float(phi))
            for phi in phi_vals
        ]
        missing = [p for p in candidate_paths if not p.exists()]
        if not missing:
            parallel_used = True
            parallel_seed_paths = candidate_paths
            seed_cache_used_paths = [str(p.resolve()) for p in parallel_seed_paths]
            seed_cache_used = ";".join(seed_cache_used_paths)
            print(f"parallel seeds  : {len(parallel_seed_paths)} cached profiles")
        else:
            print(f"parallel skip   : faltan {len(missing)} seed cache files")

    if (not parallel_used) and args.seed_profile_npz.strip():
        seed_path = Path(args.seed_profile_npz).expanduser()
        prev_solution = load_seed_profile_npz(
            seed_path,
            n_species=ct.Solution(resolved_mech).n_species,
            source=str(args.seed_profile_source),
        )
        print(f"seed profile    : {seed_path.resolve()} [{args.seed_profile_source}]")
    elif (not parallel_used) and len(phi_vals) > 0:
        cached_seed, cached_path = maybe_load_cached_seed(
            args,
            resolved_mech=resolved_mech,
            phi=float(phi_vals[0]),
            n_species=ct.Solution(resolved_mech).n_species,
        )
        if cached_seed is not None and cached_path is not None:
            prev_solution = cached_seed
            seed_cache_used = str(cached_path.resolve())
            seed_cache_used_paths.append(seed_cache_used)
            print(f"seed cache      : {seed_cache_used}")

    active_workers = min(parallel_workers, len(phi_vals)) if parallel_used else 1
    cpu_count = max(1, int(os.cpu_count() or 1))
    requested_numba_threads = int(args.numba_kinetics_threads)
    if requested_numba_threads <= 0:
        if parallel_used:
            # En procesos separados repartimos los hilos para no saturar la CPU.
            requested_numba_threads = max(1, cpu_count // max(1, active_workers))
        else:
            # Para una llama individual, GRI30 y los lotes pequeños de FGM,
            # demasiados hilos introducen más sincronización que trabajo útil.
            # El usuario puede sobreescribirlo con --numba-kinetics-threads.
            requested_numba_threads = min(4, cpu_count)
    args.numba_kinetics_threads_resolved = configure_numba_kinetics_threads(
        requested_numba_threads
    )
    print(
        f"numba chemistry : {args.numba_kinetics_threads_resolved} threads/process "
        f"({active_workers} process(es))"
    )

    if parallel_used:
        max_workers = min(parallel_workers, len(phi_vals))
        print(f"parallel solve  : {max_workers} workers")
        payloads = [
            (i, float(phi), str(parallel_seed_paths[i]), args)
            for i, phi in enumerate(phi_vals)
        ]
        records_parallel: list[FlameRecord | None] = [None] * len(phi_vals)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(solve_flame_from_seed_cache_worker, payload): payload[0]
                for payload in payloads
            }
            for fut in as_completed(futures):
                idx, rec = fut.result()
                records_parallel[idx] = rec
                status = "OK" if rec.solve_ok else "FAIL"
                print(
                    f"  [{idx + 1}/{len(phi_vals)}] {status} "
                    f"phi={rec.phi:.4f} | Su={rec.Su_m_per_s:.4f} m/s | "
                    f"n={rec.n_points} | ||F||inf={rec.residual_inf:.3e} | "
                    f"wstep={rec.weighted_step_norm:.3e} | t={rec.solve_time_s:.2f}s"
                )
                if (not (rec.solve_ok and rec.final_accepted)) and not bool(args.allow_failed_flamelets):
                    raise RuntimeError(
                        f"Flamelet phi={rec.phi:.6g} no aceptado: "
                        f"solve_ok={rec.solve_ok}, criterio={rec.acceptance_criterion}, "
                        f"||F||inf={rec.residual_inf:.3e}, "
                        f"wstep={rec.weighted_step_norm:.3e}."
                    )
        records = [rec for rec in records_parallel if rec is not None]
        species_names = list(ct.Solution(args.mech).species_names)
        for rec in records:
            c_tmp, _, used = compute_progress_variable(
                species_names=species_names, Y=rec.Y, T=rec.T,
                progress_weights=progress_weights,
            )
            rec.c = c_tmp
            if used:
                used_progress_species = used
            if rec.solve_ok and rec.final_accepted:
                cached_path = write_cached_seed(
                    args, resolved_mech=resolved_mech, rec=rec, species_names=species_names
                )
                if cached_path is not None:
                    seed_cache_written.append(str(cached_path.resolve()))
    else:
        for i, phi in enumerate(phi_vals, start=1):
            z_tag = z_targets[i - 1] if z_mode else Z_preview[i - 1]
            seed_for_phi = None
            seed_path_for_phi = None
            if (
                not bool(args.disable_seed_cache)
                and not args.seed_profile_npz.strip()
            ):
                seed_for_phi, seed_path_for_phi = maybe_load_cached_seed(
                    args,
                    resolved_mech=resolved_mech,
                    phi=float(phi),
                    n_species=ct.Solution(resolved_mech).n_species,
                )
                if seed_for_phi is not None and seed_path_for_phi is not None:
                    seed_path_text = str(seed_path_for_phi.resolve())
                    if seed_path_text not in seed_cache_used_paths:
                        seed_cache_used_paths.append(seed_path_text)
                    seed_cache_used = ";".join(seed_cache_used_paths)
            print(
                f"\n[{i}/{len(phi_vals)}] Solving phi={phi:.4f}  "
                f"Z_target={z_tag:.4f} ..."
            )
            if seed_for_phi is not None and seed_path_for_phi is not None:
                print(f"  seed cache phi : {seed_path_for_phi.resolve()}")
            rec, next_solution = solve_flame_native(
                phi=float(phi), args=args,
                mech_data=mech_data, opts=opts,
                progress_weights=progress_weights,
                prev_solution=(
                    None if args.disable_continuation
                    else (seed_for_phi if seed_for_phi is not None else prev_solution)
                ),
                prev_prev_solution=(
                    None if args.disable_continuation or seed_for_phi is not None
                    else prev_prev_solution
                ),
            )
            records.append(rec)
            status = "OK" if rec.solve_ok else "FAIL"
            print(f"  {status} | Su={rec.Su_m_per_s:.4f} m/s | Z={rec.Z:.4f} | "
                  f"n={rec.n_points} | ||F||inf={rec.residual_inf:.3e} | "
                  f"wstep={rec.weighted_step_norm:.3e} | "
                  f"t={rec.solve_time_s:.2f}s")
            if rec.solve_ok and rec.final_accepted:
                prev_prev_solution = prev_solution
                prev_solution = next_solution
            elif not bool(args.allow_failed_flamelets):
                raise RuntimeError(
                    f"Flamelet phi={phi:.6g} no aceptado: "
                    f"solve_ok={rec.solve_ok}, criterio={rec.acceptance_criterion}, "
                    f"||F||inf={rec.residual_inf:.3e}, "
                    f"wstep={rec.weighted_step_norm:.3e}."
                )

            if species_names is None:
                species_names = list(ct.Solution(args.mech).species_names)
            c_tmp, _, used = compute_progress_variable(
                species_names=species_names, Y=rec.Y, T=rec.T,
                progress_weights=progress_weights,
            )
            rec.c = c_tmp
            if used:
                used_progress_species = used
            if rec.solve_ok and rec.final_accepted:
                cached_path = write_cached_seed(
                    args, resolved_mech=resolved_mech, rec=rec, species_names=species_names
                )
                if cached_path is not None:
                    seed_cache_written.append(str(cached_path.resolve()))

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
        "solver_backend": "v2_native",
        "reference_workflow": V2_REFERENCE_WORKFLOW,
        "reference_solver": V2_REFERENCE_SOLVER,
        "external_flamelet_table_tool_used": EXTERNAL_FLAMELET_TABLE_TOOL_USED,
        "external_flamelet_table_tool": EXTERNAL_FLAMELET_TABLE_TOOL,
        "table_builder": TABLE_BUILDER,
        "timing_scope": (
            "V2 native solve plus configured continuation/cache/parallel policy; "
            "excludes any external Streamline Flamelet Table Tool execution"
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
        "all_solve_ok": bool(np.all(tables["solve_ok"])),
        "max_residual_inf": float(np.nanmax(tables["residual_inf"])),
        "max_weighted_step_norm": float(np.nanmax(tables["weighted_step_norm"])),
        "all_final_accepted": bool(np.all(tables["final_accepted"])),
        "acceptance_criterion": str(args.acceptance_criterion),
        "residual_guard_inf": float(args.residual_guard_inf),
        "upwind_factor": float(args.upwind_factor),
        "precompute_jacobian_thermo": bool(args.precompute_jacobian_thermo),
        "compiled_block_substitution": bool(args.compiled_block_substitution),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "auto"),
        "continuation_predictor_enabled": not bool(args.disable_continuation_predictor),
        "continuation_predictor_damping": float(args.continuation_predictor_damping),
        "continuation_trust_ratio": float(args.continuation_trust_ratio),
        "restart_insert_anchor": bool(args.restart_insert_anchor),
        "seed_cache_dir": str(seed_cache_directory(args).resolve()),
        "seed_cache_used": seed_cache_used,
        "seed_cache_used_paths": seed_cache_used_paths,
        "seed_cache_written": seed_cache_written,
        "seed_cache_files_before": int(cache_files0),
        "seed_cache_size_mib_before": float(cache_mb0),
        "parallel_workers_requested": int(args.parallel_workers),
        "parallel_workers_resolved": int(parallel_workers),
        "parallel_used": bool(parallel_used),
        "numba_kinetics_threads_requested": int(args.numba_kinetics_threads),
        "numba_kinetics_threads_resolved": int(
            getattr(args, "numba_kinetics_threads_resolved", 1)
        ),
        "Z_range": [float(tables["Z_grid"].min()), float(tables["Z_grid"].max())],
        "Z_st": Z_st,
        "runtime_s": float(time.perf_counter() - t_global0),
    }
    cache_files1, cache_mb1 = seed_cache_stats(cache_dir)
    meta["seed_cache_files_after"] = int(cache_files1)
    meta["seed_cache_size_mib_after"] = float(cache_mb1)
    meta["seed_cache_files_delta"] = int(cache_files1 - cache_files0)
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "-" * 70)
    print("FGM listo — tabla en espacio (Z, c).")
    print(f"Z_grid : {tables['Z_grid']}")
    print(f"c_grid : {c_grid.size} puntos adaptativos")
    print(f"Tabla  : {(out_dir / 'fgm_table.npz').resolve()}")
    print(
        f"Cache  : {cache_files1} files | {cache_mb1:.1f} MiB "
        f"(delta {cache_files1 - cache_files0:+d})"
    )
    print(f"Runtime: {meta['runtime_s']:.1f} s")
    print("-" * 70)


if __name__ == "__main__":
    main()
