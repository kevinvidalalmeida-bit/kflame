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
from fgm_continuation import AdaptiveContinuationController, ContinuationConfig


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
from equations import residual, solve_linear
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
    requested: bool = True
    bridge: bool = False
    predictor_kind: str = "cold"
    prediction_defect: float = float("nan")


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


def _jacobian_tangent_state(
    problem: FreeFlameProblem,
    target_phi: float,
    previous: dict[str, Any],
    damping: float,
) -> tuple[np.ndarray, dict[str, float]] | None:
    """Form a first-order continuation predictor from the prior stationary LU.

    At an accepted state :math:`x_i` and ``lambda = log(phi)``, the discrete
    tangent satisfies ``J_i dx/dlambda = -dF/dlambda``.  Rather than forming a
    second finite-difference Jacobian, the residual of the *target physical
    problem* at ``x_i`` and the retained source residual provide the local
    parameter chord:

    ``dx ~= -J_i^{-1} [F(x_i, lambda_target) - F(x_i, lambda_source)]``.

    The subtraction is required whenever the source flame is accepted by the
    operational ``weighted step + residual guard`` certificate instead of an
    exact zero residual.  Using the target residual alone would incorrectly
    interpret the remaining source solve error as a change in ``phi``.

    The LU is the exact stationary factorization retained in memory from the
    preceding certified flame.  This function therefore performs one RHS and
    one triangular solve; it does not run a Newton iteration on the target.
    Any failure returns ``None`` so the normal copy/secant continuation remains
    available, and every returned state is still corrected and certified by
    ``solve_free_flame``.
    """
    try:
        phi_source = float(np.asarray(previous["phi"], dtype=float).ravel()[0])
        z = np.asarray(previous["z"], dtype=float)
        x_previous = np.asarray(previous["x"], dtype=float)
        linearization = previous["continuation_linearization"]
        lu = linearization["lu"]
        anchor_index = int(linearization["anchor_index"])
        linearization_state = np.asarray(linearization["state"], dtype=float)
        source_residual = np.asarray(linearization["residual"], dtype=float)
    except (KeyError, TypeError, ValueError, IndexError):
        return None

    if phi_source <= 0.0 or target_phi <= 0.0 or z.ndim != 1 or z.size < 3:
        return None
    n_points = int(z.size)
    n_species = int(problem.n_species)
    expected = n_points * (n_species + 2)
    if x_previous.size != expected:
        return None
    if (
        not isinstance(linearization, dict)
        or int(linearization.get("n_points", -1)) != n_points
        or int(linearization.get("n_species", -1)) != n_species
        or not 0 < anchor_index < n_points - 1
        or not isinstance(lu, dict)
        or lu.get("method") != "block_tridiag"
    ):
        return None
    # A finite-difference Jacobian is local to both a mesh and a state. Do
    # not silently apply a stale hand-off after any external seed mutation.
    if (
        linearization_state.shape != x_previous.shape
        or source_residual.shape != x_previous.shape
        or not np.allclose(
            linearization_state,
            x_previous,
            rtol=1.0e-11,
            atol=1.0e-13,
        )
        or not np.all(np.isfinite(source_residual))
    ):
        return None

    lambda_source = float(np.log(phi_source))
    lambda_target = float(np.log(target_phi))
    delta_lambda = lambda_target - lambda_source
    if not np.isfinite(delta_lambda) or abs(delta_lambda) <= 1.0e-14:
        return None

    # Evaluate the target residual on precisely the source mesh. The actual
    # seeded solve below starts from this same mesh, so the tangent and
    # corrector refer to the same discrete variables and anchor convention.
    problem.z = z.copy()
    problem.n_points = n_points
    problem.width = float(z[-1] - z[0])
    problem.solve_energy = True
    try:
        # The source LU contains a phase row at ``anchor_index``. Keep that
        # row fixed while taking the parameter chord; only its prescribed
        # target temperature changes with the new equilibrium state. Letting
        # the target select a different node would pair F(lambda_target) with
        # a Jacobian of another discrete system.
        problem.j_fixed = anchor_index
        problem.T_fixed_point = float(problem.anchor_T)
        problem.backend = problem.backend_factory(problem)
        target_residual = np.asarray(residual(x_previous, problem), dtype=float)
        if target_residual.shape != x_previous.shape or not np.all(np.isfinite(target_residual)):
            return None
        # Written in sensitivity form to retain a measurable dF/dlambda. The
        # source subtraction preserves the distinction between a parameter
        # chord and any residual left by the accepted source corrector. The
        # multiplication by delta_lambda recovers the chord correction and
        # avoids a second target-Jacobian assembly.
        parameter_chord = target_residual - source_residual
        dF_dlambda = parameter_chord / delta_lambda
        sensitivity = np.asarray(solve_linear(lu, -dF_dlambda), dtype=float)
        correction = float(np.clip(damping, 0.0, 1.0)) * delta_lambda * sensitivity
    except Exception:
        return None

    x_prediction = x_previous + correction
    if x_prediction.shape != x_previous.shape or not np.all(np.isfinite(x_prediction)):
        return None
    metadata: dict[str, float] = {
        "lambda_source": lambda_source,
        "lambda_target": lambda_target,
        "delta_lambda": float(delta_lambda),
        "source_jacobian_age": float(linearization.get("jacobian_age", np.nan)),
        "source_jacobian_evaluations": float(
            linearization.get("jacobian_evaluations", np.nan)
        ),
        "source_anchor_index": float(anchor_index),
        "source_residual_inf": float(np.linalg.norm(source_residual, ord=np.inf)),
        "target_residual_inf_before_tangent": float(
            np.linalg.norm(target_residual, ord=np.inf)
        ),
        "parameter_chord_inf": float(np.linalg.norm(parameter_chord, ord=np.inf)),
        "tangent_sensitivity_weighted_norm": float(
            np.linalg.norm(sensitivity) / max(np.sqrt(sensitivity.size), 1.0)
        ),
    }
    return x_prediction, metadata


def build_continuation_seed(
    problem: FreeFlameProblem,
    phi: float,
    prev_solution: dict[str, np.ndarray] | None,
    prev_prev_solution: dict[str, np.ndarray] | None,
    use_predictor: bool,
    predictor_damping: float,
    trust_ratio: float = _CONTINUATION_MAX_PHI_RATIO,
    predictor_model: str = "secant",
) -> dict[str, Any] | None:
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

    predictor_model = str(predictor_model).strip().lower()
    if predictor_model not in ("secant", "tangent"):
        raise ValueError("predictor_model must be 'secant' or 'tangent'")

    x_seed = x_prev.copy()
    predictor_kind = "copy"
    predictor_metadata: dict[str, float] = {}
    if use_predictor and predictor_model == "tangent":
        tangent_prediction = _jacobian_tangent_state(
            problem=problem,
            target_phi=float(phi),
            previous=prev_solution,
            damping=float(predictor_damping),
        )
        if tangent_prediction is not None:
            x_seed, predictor_metadata = tangent_prediction
            predictor_kind = "tangent_jacobian"
        else:
            # No valid stationary source LU is a normal occurrence after a
            # transient-only solve. Fall through to the existing predictor;
            # the trace distinguishes this from a genuine tangent prediction.
            predictor_metadata = {"tangent_fallback": 1.0}
    if (
        use_predictor
        and predictor_kind == "copy"
        and prev_prev_solution is not None
        and "z" in prev_prev_solution
        and "x" in prev_prev_solution
        and "phi" in prev_prev_solution
        and "phi" in prev_solution
    ):
        phi0 = float(np.asarray(prev_prev_solution.get("phi")).ravel()[0])
        phi1 = float(np.asarray(prev_solution.get("phi")).ravel()[0])
        lambda0 = float(np.log(phi0)) if phi0 > 0.0 else float("nan")
        lambda1 = float(np.log(phi1)) if phi1 > 0.0 else float("nan")
        lambda_target = float(np.log(phi)) if phi > 0.0 else float("nan")
        dlambda = lambda1 - lambda0
        if (
            np.isfinite(lambda_target)
            and abs(dlambda) > 1.0e-14
            and _is_local_phi_step(phi0, phi1, trust_ratio)
        ):
            z0 = np.asarray(prev_prev_solution.get("z"), dtype=float)
            x0 = np.asarray(prev_prev_solution.get("x"), dtype=float)
            expected0 = int(z0.size) * (2 + n_sp)
            if z0.ndim == 1 and z0.size >= 2 and x0.size == expected0:
                x0_on_prev = interpolate_state(x0, z0, z_prev, n_sp)
                # The continuation coordinate is lambda=log(phi), rather
                # than phi itself, so rich and lean changes are symmetric.
                factor = (lambda_target - lambda1) / dlambda
                factor *= float(np.clip(predictor_damping, 0.0, 1.0))
                factor = float(np.clip(factor, -0.5, 1.0))
                x_seed = x_prev + factor * (x_prev - x0_on_prev)
                predictor_kind = "secant"

    u, T, Y = unpack_state(x_seed, int(z_prev.size), n_sp)

    T = np.clip(T, float(problem.T_lower_bound), float(problem.T_upper_bound))
    Y = np.clip(Y, 0.0, None)
    sums = Y.sum(axis=0, keepdims=True)
    sums = np.where(sums > 0.0, sums, 1.0)
    Y = Y / sums

    # A copy or secant predictor has no direct knowledge of the changed inlet
    # chemistry, so project the target fresh mixture through the preheat zone.
    # The Jacobian tangent already contains that target-residual response and
    # must not receive the same composition change a second time.
    source_inlet = Y[:, 0].copy()
    target_inlet = np.asarray(problem.Y_in, dtype=float)
    thermal_span = max(abs(float(T[-1] - float(problem.T_in))), 1.0e-12)
    thermal_progress = np.clip((T - float(problem.T_in)) / thermal_span, 0.0, 1.0)
    fresh_weight = 1.0 - thermal_progress
    inlet_delta = target_inlet - source_inlet
    if (
        not predictor_kind.startswith("tangent_jacobian")
        and np.max(np.abs(inlet_delta)) > 1.0e-14
    ):
        Y += fresh_weight[None, :] * inlet_delta[:, None]
        Y = np.clip(Y, 0.0, None)
        Y /= np.maximum(Y.sum(axis=0, keepdims=True), 1.0e-300)
        predictor_kind += "_fresh_projected"

    # Current inlet composition is a hard boundary condition.
    u[0] = max(float(u[0]), 1.0e-8)
    T[0] = float(problem.T_in)
    Y[:, 0] = target_inlet

    result = {
        "phi": np.array([float(phi)], dtype=float),
        "z": z_prev.copy(),
        "x": pack_state(u, T, Y),
        "predictor_kind": predictor_kind,
        "predictor_frame": "z",
        "predictor_model": predictor_model,
    }
    if predictor_metadata:
        result["predictor_metadata"] = predictor_metadata
    return result


def bound_continuation_seed_mesh(
    seed: dict[str, Any],
    n_species: int,
    max_points: int,
) -> dict[str, Any]:
    """Transfer a predicted state without inheriting every source mesh node.

    A continuation profile and the mesh on which it was certified are distinct
    objects. Retaining an excessively dense source mesh can force a nearby
    flame through an unnecessarily difficult nonlinear trajectory. This
    bounded transfer preserves both physical endpoints and samples the source
    adaptive geometry by node index; the target still refines and certifies its
    own mesh from scratch. ``max_points <= 0`` is an explicit no-transfer
    baseline.
    """
    if max_points <= 0:
        return seed
    z_old = np.asarray(seed.get("z"), dtype=float)
    x_old = np.asarray(seed.get("x"), dtype=float)
    expected = int(z_old.size) * (2 + int(n_species))
    if (
        z_old.ndim != 1
        or z_old.size < 2
        or x_old.size != expected
        or not np.all(np.isfinite(z_old))
        or np.any(np.diff(z_old) <= 0.0)
    ):
        return seed
    n_target = max(2, min(int(max_points), int(z_old.size)))
    if n_target == z_old.size:
        return seed
    selected = np.rint(np.linspace(0, z_old.size - 1, n_target)).astype(int)
    selected[0], selected[-1] = 0, z_old.size - 1
    selected = np.unique(selected)
    z_new = z_old[selected]
    x_new = interpolate_state(x_old, z_old, z_new, int(n_species))
    result = dict(seed)
    result["z"] = z_new
    result["x"] = x_new
    result["seed_mesh_transfer"] = "bounded_adaptive_subsampling"
    result["seed_mesh_source_points"] = int(z_old.size)
    result["seed_mesh_target_points"] = int(z_new.size)
    return result


def weighted_prediction_defect(
    problem: FreeFlameProblem,
    corrected_x: np.ndarray,
    prediction: dict[str, np.ndarray] | None,
) -> float:
    """Return the weighted state defect after a full nonlinear correction.

    The predictor is resampled on the final adaptive mesh.  Component scales
    match the solver's weighted-step convention, which avoids letting
    temperature or trace species dominate purely by their units.
    """
    if prediction is None or "x" not in prediction or "z" not in prediction:
        return float("nan")
    z_prediction = np.asarray(prediction["z"], dtype=float)
    x_prediction = np.asarray(prediction["x"], dtype=float)
    n_sp = int(problem.n_species)
    n_vars = 2 + n_sp
    expected = int(z_prediction.size) * n_vars
    if z_prediction.ndim != 1 or z_prediction.size < 2 or x_prediction.size != expected:
        return float("nan")
    try:
        predicted_on_final = interpolate_state(
            x_prediction, z_prediction, np.asarray(problem.z, dtype=float), n_sp
        )
    except Exception:
        return float("nan")
    corrected = np.asarray(corrected_x, dtype=float).reshape(-1, n_vars)
    predicted = np.asarray(predicted_on_final, dtype=float).reshape(-1, n_vars)
    if corrected.shape != predicted.shape or not np.all(np.isfinite(predicted)):
        return float("nan")
    scales = (
        float(getattr(problem, "steady_atol", 1.0e-9))
        + float(getattr(problem, "steady_rtol", 1.0e-4))
        * np.mean(np.abs(corrected), axis=0)
    )
    scales = np.maximum(scales, 1.0e-300)
    return float(np.sqrt(np.mean(((corrected - predicted) / scales) ** 2)))


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
        profile=bool(args.profile_solver),
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
    opts.local_jacobian_refresh = bool(args.local_jacobian_refresh)
    opts.local_jacobian_refresh_defect_threshold = float(
        args.local_jacobian_refresh_defect_threshold
    )
    opts.local_jacobian_refresh_max_fraction = float(
        args.local_jacobian_refresh_max_fraction
    )
    opts.local_jacobian_refresh_min_blocks = int(
        args.local_jacobian_refresh_min_blocks
    )
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
    continuation_trust_ratio: float | None = None,
) -> tuple[FlameRecord, dict[str, np.ndarray], dict[str, Any]]:
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

    selected_prediction: dict[str, Any] | None = None

    def run_once(jacobian_mode: str):
        nonlocal selected_prediction
        problem = FreeFlameProblem(
            flame_case, n_points=max(2, int(args.initial_grid_points))
        )
        problem.assume_finite_y = True
        problem.backend_factory = lambda prob: NativeSpeciesBackend(prob, mech_data=mech_data)

        # Keep cold-start Jacobian aging conservative.  A converged profile
        # is much closer to the next phi state, where a longer reuse window
        # avoids costly rebuilds without degrading the certified result.
        run_opts = copy.copy(opts)
        predictor_damping = (
            float(args.continuation_tangent_damping)
            if str(args.continuation_predictor_model).strip().lower() == "tangent"
            else float(args.continuation_predictor_damping)
        )
        t_predictor = time.perf_counter()
        seed_solution = build_continuation_seed(
            problem=problem,
            phi=float(phi),
            prev_solution=prev_solution,
            prev_prev_solution=prev_prev_solution,
            use_predictor=not bool(args.disable_continuation_predictor),
            predictor_damping=predictor_damping,
            trust_ratio=(
                float(args.continuation_trust_ratio)
                if continuation_trust_ratio is None
                else float(continuation_trust_ratio)
            ),
            predictor_model=str(args.continuation_predictor_model),
        )
        predictor_build_time_s = float(time.perf_counter() - t_predictor)
        if seed_solution is not None:
            seed_solution["predictor_build_time_s"] = predictor_build_time_s
            seed_solution = bound_continuation_seed_mesh(
                seed_solution,
                n_species=int(problem.n_species),
                max_points=int(args.continuation_seed_mesh_points),
            )
        selected_prediction = seed_solution

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

        # The local-defect rescue is a certified continuation corrector, not
        # an alternative cold-start globalization strategy. Restricting it
        # to a physically neighbouring certified seed preserves the audited
        # cold baseline while keeping the method general for any continuation
        # coordinate (phi, pressure, inlet temperature, or enthalpy).
        run_opts.local_jacobian_refresh = bool(
            run_opts.local_jacobian_refresh and x0 is not None
        )

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
            # The tight stage checks spatial convergence, not a parameter
            # continuation transition. Keep its route equal to the audited
            # refinement baseline.
            tight_opts.local_jacobian_refresh = False
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
        report["predictor_build_time_s"] = predictor_build_time_s
        report["local_jacobian_refresh_scope"] = (
            "continuation_corrector" if bool(run_opts.local_jacobian_refresh) else "off"
        )
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

    predictor_kind = (
        str(selected_prediction.get("predictor_kind", "copy"))
        if selected_prediction is not None else "cold"
    )
    prediction_defect = weighted_prediction_defect(
        problem, np.asarray(x_sol, dtype=float), selected_prediction
    )
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
        predictor_kind=predictor_kind,
        prediction_defect=prediction_defect,
    )
    next_solution = {
        "phi": np.array([float(phi)], dtype=float),
        "z": z.copy(),
        "x": np.asarray(x_sol, dtype=float).copy(),
    }
    linearization = getattr(problem, "_continuation_linearization", None)
    if isinstance(linearization, dict):
        # This hand-off is intentionally not serialised as a seed-cache field:
        # it is valid only for the immediately neighbouring target on the
        # current process and configuration.
        next_solution["continuation_linearization"] = linearization
    continuation_trace = {
        "predictor_kind": predictor_kind,
        "predictor_frame": (
            str(selected_prediction.get("predictor_frame", "z"))
            if selected_prediction is not None else "none"
        ),
        "predictor_model": (
            str(selected_prediction.get("predictor_model", "secant"))
            if selected_prediction is not None else "none"
        ),
        "predictor_metadata": (
            selected_prediction.get("predictor_metadata", {})
            if selected_prediction is not None else {}
        ),
        "seed_mesh_transfer": (
            str(selected_prediction.get("seed_mesh_transfer", "none"))
            if selected_prediction is not None else "none"
        ),
        "seed_mesh_source_points": (
            int(selected_prediction.get("seed_mesh_source_points", 0))
            if selected_prediction is not None else 0
        ),
        "seed_mesh_target_points": (
            int(selected_prediction.get("seed_mesh_target_points", 0))
            if selected_prediction is not None else 0
        ),
        "seeded": bool(selected_prediction is not None),
        "prediction_defect": prediction_defect,
        "predictor_build_time_s": float(report.get("predictor_build_time_s", 0.0)),
        "local_jacobian_refresh_scope": str(
            report.get("local_jacobian_refresh_scope", "off")
        ),
        "n_refine_passes": int(len(report.get("passes", []))),
        "n_domain_expansions": int(len(report.get("expansion_events", []))),
        "solve_time_s": float(dt),
        "final_accepted": bool(final_accepted),
        "residual_inf": residual_inf,
        "weighted_step_norm": weighted_step_norm,
    }
    if bool(getattr(args, "profile_solver", False)):
        continuation_trace["solver_profile"] = report.get("profile", {})
    return rec, next_solution, continuation_trace


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
    rec, _, _ = solve_flame_native(
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
            "n_points", "width_m", "solve_time_s", "requested", "bridge",
            "predictor_kind", "prediction_defect",
        ])
        for rec in records:
            wr.writerow([
                rec.phi, rec.Z, rec.solve_ok, rec.final_accepted,
                rec.acceptance_criterion, rec.residual_inf, rec.weighted_step_norm,
                rec.Su_m_per_s, rec.n_points, rec.width_m, rec.solve_time_s,
                rec.requested, rec.bridge, rec.predictor_kind, rec.prediction_defect,
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
            requested=np.array([rec.requested], dtype=bool),
            bridge=np.array([rec.bridge], dtype=bool),
            predictor_kind=np.array([rec.predictor_kind], dtype=object),
            prediction_defect=np.array([rec.prediction_defect], dtype=float),
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
    requested = np.array([r.requested for r in records], dtype=bool)
    bridge = np.array([r.bridge for r in records], dtype=bool)
    predictor_kind = np.array([r.predictor_kind for r in records], dtype=object)
    prediction_defect = np.array([r.prediction_defect for r in records], dtype=float)

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
        "requested": requested, "bridge": bridge,
        "predictor_kind": predictor_kind, "prediction_defect": prediction_defect,
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
    p.add_argument(
        "--profile-solver",
        action="store_true",
        help=(
            "Guarda contadores y tiempos internos por flamelet en la traza. "
            "Usarlo solo en perfiles, no en la campana temporal principal."
        ),
    )
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
    p.add_argument(
        "--local-jacobian-refresh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Ante un damping no contractivo, actualiza solo los bloques "
            "espaciales indicados por el defecto no lineal y exige el mismo "
            "criterio de aceptacion. Solo actua en el corrector de continuacion; "
            "--no-local-jacobian-refresh reproduce el baseline de ablacion."
        ),
    )
    p.add_argument(
        "--local-jacobian-refresh-defect-threshold",
        type=float,
        default=0.20,
        help="Defecto relativo por bloque que activa el refresco local certificado.",
    )
    p.add_argument(
        "--local-jacobian-refresh-max-fraction",
        type=float,
        default=0.35,
        help=(
            "Fraccion maxima de bloques que puede refrescarse localmente; por "
            "encima se conserva la reconstruccion global normal."
        ),
    )
    p.add_argument(
        "--local-jacobian-refresh-min-blocks",
        type=int,
        default=1,
        help="Numero minimo de bloques para intentar el refresco local certificado.",
    )
    p.add_argument("--allow-failed-flamelets", action="store_true",
                   help="Permite guardar la tabla aunque algun flamelet no converja.")
    p.add_argument(
        "--continuation-mode",
        choices=("cold", "fixed", "fixed-bridges", "adaptive-pc"),
        default="fixed",
        help=(
            "Politica paramétrica: cold no reutiliza perfiles vecinos; fixed "
            "reproduce la region de confianza actual; fixed-bridges conserva "
            "un paso fijo con puntos puente; adaptive-pc adapta ese paso en log(phi)."
        ),
    )
    p.add_argument("--disable-continuation", action="store_true",
                   help="Alias heredado de --continuation-mode cold.")
    p.add_argument("--disable-continuation-predictor", action="store_true",
                   help="Desactiva los predictores parametricos entre flamelets.")
    p.add_argument(
        "--continuation-predictor-model",
        choices=("secant", "tangent"),
        default="secant",
        help=(
            "Modelo de semilla: secant reproduce el baseline; tangent reutiliza "
            "la LU estacionaria de la llama anterior para una sensibilidad local."
        ),
    )
    p.add_argument("--continuation-predictor-damping", type=float, default=0.7,
                   help="Amortiguamiento del predictor secante en phi.")
    p.add_argument(
        "--continuation-tangent-damping",
        type=float,
        default=0.3,
        help=(
            "Amortiguamiento global del predictor tangente con Jacobiano. "
            "No modifica el baseline secante."
        ),
    )
    p.add_argument(
        "--continuation-seed-mesh-points",
        type=int,
        default=0,
        help=(
            "Maximo de nodos heredados por una semilla de continuacion; 0 "
            "conserva la malla fuente como baseline. El corrector siempre "
            "vuelve a refinar y certificar su propia malla."
        ),
    )
    p.add_argument(
        "--continuation-trust-ratio",
        type=float,
        default=_CONTINUATION_MAX_PHI_RATIO,
        help=(
            "Razon multiplicativa maxima para reutilizar una semilla en phi; "
            "0 desactiva solo esta cota para la ablacion no acotada."
        ),
    )
    p.add_argument("--pc-initial-ratio", type=float, default=1.15,
                   help="Paso inicial del predictor-corrector: exp(Delta log(phi)).")
    p.add_argument("--pc-min-ratio", type=float, default=1.02,
                   help="Paso multiplicativo mínimo del predictor-corrector.")
    p.add_argument("--pc-max-ratio", type=float, default=1.20,
                   help="Paso multiplicativo máximo del predictor-corrector.")
    p.add_argument("--pc-max-retries", type=int, default=4,
                   help="Reintentos con paso reducido antes de un arranque frío registrado.")
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
    if bool(args.disable_continuation):
        args.continuation_mode = "cold"
    if not 0.0 <= float(args.upwind_factor) <= 1.0:
        raise SystemExit("--upwind-factor debe estar entre 0.0 y 1.0")
    if int(args.continuation_seed_mesh_points) < 0:
        raise SystemExit("--continuation-seed-mesh-points debe ser mayor o igual a cero")
    if float(args.local_jacobian_refresh_defect_threshold) < 0.0:
        raise SystemExit("--local-jacobian-refresh-defect-threshold debe ser no negativo")
    if not 0.0 <= float(args.local_jacobian_refresh_max_fraction) <= 1.0:
        raise SystemExit(
            "--local-jacobian-refresh-max-fraction debe estar entre 0 y 1"
        )
    if int(args.local_jacobian_refresh_min_blocks) < 1:
        raise SystemExit("--local-jacobian-refresh-min-blocks debe ser al menos 1")
    if args.continuation_mode == "adaptive-pc":
        if not (
            1.0 < float(args.pc_min_ratio) <= float(args.pc_initial_ratio)
            <= float(args.pc_max_ratio)
        ):
            raise SystemExit(
                "adaptive-pc requiere 1 < --pc-min-ratio <= "
                "--pc-initial-ratio <= --pc-max-ratio"
            )
    if args.continuation_mode == "fixed-bridges" and float(args.continuation_trust_ratio) <= 1.0:
        raise SystemExit(
            "fixed-bridges requiere --continuation-trust-ratio mayor que uno."
        )
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

    if args.continuation_mode in ("adaptive-pc", "fixed-bridges") and (
        np.any(phi_vals <= 0.0) or np.any(np.diff(phi_vals) <= 0.0)
    ):
        raise SystemExit(
            "La continuacion con puentes requiere phi positivos y estrictamente crecientes; "
            "ordene --phi-values de pobre a rico."
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
    print(f"continuation    : {args.continuation_mode}")
    print(f"phis            : {phi_vals}")
    if z_mode:
        print(f"Z target        : {[f'{z:.4f}' for z in z_targets]}")
    print(f"Z (Bilger)      : {[f'{z:.4f}' for z in Z_preview]}")
    print(f"Z_st (phi=1)    : {Z_st:.6f}")

    records: list[FlameRecord] = []
    continuation_trace: list[dict[str, Any]] = []
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
        and args.continuation_mode in ("cold", "fixed")
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
        for index, rec in enumerate(records):
            continuation_trace.append({
                "mode": "cached-parallel",
                "requested_index": int(index),
                "phi_from": None,
                "phi_target": float(rec.phi),
                "phi_trial": float(rec.phi),
                "requested": True,
                "bridge": False,
                "accepted": bool(rec.solve_ok and rec.final_accepted),
                "predictor_kind": rec.predictor_kind,
                "prediction_defect": rec.prediction_defect,
                "solve_time_s": rec.solve_time_s,
                "final_accepted": rec.final_accepted,
            })
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
    elif args.continuation_mode in ("adaptive-pc", "fixed-bridges"):
        # This path is sequential by design. A cache can seed the first
        # corrected state, but every bridge remains a fully certified solve.
        initial_seed = prev_solution
        prev_solution = None
        prev_prev_solution = None
        current_phi: float | None = None
        if args.continuation_mode == "fixed-bridges":
            fixed_ratio = float(args.continuation_trust_ratio)
            controller_config = ContinuationConfig(
                initial_ratio=fixed_ratio,
                min_ratio=fixed_ratio,
                max_ratio=fixed_ratio,
                max_retries=int(args.pc_max_retries),
            )
        else:
            controller_config = ContinuationConfig(
                initial_ratio=float(args.pc_initial_ratio),
                min_ratio=float(args.pc_min_ratio),
                max_ratio=float(args.pc_max_ratio),
                max_retries=int(args.pc_max_retries),
            )
        controller = AdaptiveContinuationController(controller_config)

        def commit_adaptive(rec: FlameRecord, trace: dict[str, Any],
                            requested: bool, bridge: bool) -> None:
            nonlocal species_names, used_progress_species
            rec.requested = bool(requested)
            rec.bridge = bool(bridge)
            records.append(rec)
            continuation_trace.append(trace)
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
                    args, resolved_mech=resolved_mech, rec=rec,
                    species_names=species_names,
                )
                if cached_path is not None:
                    seed_cache_written.append(str(cached_path.resolve()))

        for requested_index, target in enumerate(phi_vals, start=1):
            target_phi = float(target)
            while current_phi is None or not np.isclose(
                np.log(current_phi), np.log(target_phi), rtol=0.0, atol=1.0e-13
            ):
                if current_phi is None:
                    proposal = None
                    trial_phi = target_phi
                    seed, seed_previous, bridge = initial_seed, None, False
                else:
                    proposal = controller.propose(current_phi, target_phi)
                    trial_phi = float(proposal.phi_trial)
                    seed, seed_previous = prev_solution, prev_prev_solution
                    bridge = bool(proposal.is_bridge)

                before = controller.snapshot()
                print(
                    f"\n[{requested_index}/{len(phi_vals)}] {args.continuation_mode} "
                    f"target={target_phi:.4f}, trial={trial_phi:.4f}"
                    f"{' [bridge]' if bridge else ''} ..."
                )
                rec, corrected, solve_trace = solve_flame_native(
                    phi=trial_phi, args=args, mech_data=mech_data, opts=opts,
                    progress_weights=progress_weights, prev_solution=seed,
                    prev_prev_solution=seed_previous, continuation_trust_ratio=0.0,
                )
                accepted = bool(rec.solve_ok and rec.final_accepted)
                trace: dict[str, Any] = {
                    "mode": str(args.continuation_mode),
                    "requested_index": int(requested_index - 1),
                    "phi_from": None if current_phi is None else float(current_phi),
                    "phi_target": target_phi,
                    "phi_trial": trial_phi,
                    "requested": not bridge,
                    "bridge": bridge,
                    "log_step": None if proposal is None else float(proposal.log_step),
                    "retry": 0 if proposal is None else int(proposal.retry),
                    "controller_before": before,
                    **solve_trace,
                    "accepted": accepted,
                }
                if accepted:
                    trace["next_log_step"] = controller.accept(
                        rec.prediction_defect, rec.predictor_kind
                    )
                    trace["controller_after"] = controller.snapshot()
                    commit_adaptive(rec, trace, requested=not bridge, bridge=bridge)
                    print(
                        f"  OK | predictor={rec.predictor_kind} | "
                        f"eta_p={rec.prediction_defect:.3e} | "
                        f"Su={rec.Su_m_per_s:.4f} m/s | n={rec.n_points} | "
                        f"t={rec.solve_time_s:.2f}s"
                    )
                    prev_prev_solution, prev_solution = prev_solution, corrected
                    current_phi, initial_seed = trial_phi, None
                    continue

                # No failed correction is inserted into the FGM table or cache.
                if proposal is not None:
                    next_step, retry, can_retry = controller.reject()
                else:
                    next_step, retry, can_retry = None, 0, False
                trace.update({
                    "accepted": False,
                    "next_log_step": next_step,
                    "retry_after_reject": retry,
                    "controller_after_reject": controller.snapshot(),
                })
                continuation_trace.append(trace)
                if can_retry:
                    print(f"  RETRY | next ratio={np.exp(next_step):.4f} (retry {retry})")
                    continue

                # Recovery is an explicit cold target solve, not a hidden seed.
                print("  COLD FALLBACK | continuation budget exhausted")
                cold_rec, cold_solution, cold_trace = solve_flame_native(
                    phi=target_phi, args=args, mech_data=mech_data, opts=opts,
                    progress_weights=progress_weights, prev_solution=None,
                    prev_prev_solution=None, continuation_trust_ratio=0.0,
                )
                cold_accepted = bool(cold_rec.solve_ok and cold_rec.final_accepted)
                cold_trace_record: dict[str, Any] = {
                    "mode": str(args.continuation_mode),
                    "requested_index": int(requested_index - 1),
                    "phi_from": None if current_phi is None else float(current_phi),
                    "phi_target": target_phi,
                    "phi_trial": target_phi,
                    "requested": True,
                    "bridge": False,
                    "cold_fallback": True,
                    **cold_trace,
                    "accepted": cold_accepted,
                    "controller_after": controller.snapshot(),
                }
                if not cold_accepted and not bool(args.allow_failed_flamelets):
                    raise RuntimeError(
                        f"Flamelet phi={target_phi:.6g} no aceptado tras agotar "
                        "predictor-corrector y arranque frío."
                    )
                commit_adaptive(cold_rec, cold_trace_record, requested=True, bridge=False)
                prev_prev_solution, prev_solution = prev_solution, cold_solution
                current_phi, initial_seed = target_phi, None
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
            rec, next_solution, solve_trace = solve_flame_native(
                phi=float(phi), args=args,
                mech_data=mech_data, opts=opts,
                progress_weights=progress_weights,
                prev_solution=(
                    None if args.continuation_mode == "cold"
                    else (seed_for_phi if seed_for_phi is not None else prev_solution)
                ),
                prev_prev_solution=(
                    None if args.continuation_mode == "cold" or seed_for_phi is not None
                    else prev_prev_solution
                ),
            )
            records.append(rec)
            continuation_trace.append({
                "mode": str(args.continuation_mode),
                "requested_index": int(i - 1),
                "phi_from": None if prev_solution is None else float(np.asarray(prev_solution["phi"]).ravel()[0]),
                "phi_target": float(phi),
                "phi_trial": float(phi),
                "requested": True,
                "bridge": False,
                "accepted": bool(rec.solve_ok and rec.final_accepted),
                **solve_trace,
            })
            status = "OK" if rec.solve_ok else "FAIL"
            print(f"  {status} | Su={rec.Su_m_per_s:.4f} m/s | Z={rec.Z:.4f} | "
                  f"n={rec.n_points} | ||F||inf={rec.residual_inf:.3e} | "
                  f"wstep={rec.weighted_step_norm:.3e} | "
                  f"t={rec.solve_time_s:.2f}s")
            if rec.solve_ok and rec.final_accepted and args.continuation_mode == "fixed":
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
    continuation_schedule = {
        "coordinate": "log(phi)",
        "phi_requested": [float(value) for value in phi_vals],
        "phi_resolved": [float(value) for value in tables["phi_grid"]],
        "requested": [bool(value) for value in tables["requested"]],
        "bridge": [bool(value) for value in tables["bridge"]],
    }
    (out_dir / "continuation_schedule.json").write_text(
        json.dumps(continuation_schedule, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "continuation_trace.json").write_text(
        json.dumps(continuation_trace, indent=2, ensure_ascii=False),
        encoding="utf-8",
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
        "n_phi_requested": len(phi_vals),
        "n_phi_table": len(records),
        "n_phi_bridge": int(sum(bool(rec.bridge) for rec in records)),
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
        "local_jacobian_refresh": {
            "enabled": bool(args.local_jacobian_refresh),
            "scope": "continuation_corrector_only",
            "defect_threshold": float(args.local_jacobian_refresh_defect_threshold),
            "max_fraction": float(args.local_jacobian_refresh_max_fraction),
            "min_blocks": int(args.local_jacobian_refresh_min_blocks),
            "policy": (
                "local_blocks_only_after_failed_standard_contraction; "
                "exact_LU_and_standard_certificate_required"
            ),
        },
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "auto"),
        "continuation_predictor_enabled": not bool(args.disable_continuation_predictor),
        "continuation_predictor_model": str(args.continuation_predictor_model),
        "continuation_predictor_damping": float(args.continuation_predictor_damping),
        "continuation_tangent_damping": float(args.continuation_tangent_damping),
        "continuation_tangent": {
            "coordinate": "log(phi)",
            "linearization": "previous_certified_stationary_block_LU",
            "target_derivative": "residual_chord_on_source_mesh",
            "persistent_cache": False,
        },
        "continuation_seed_mesh_points": int(args.continuation_seed_mesh_points),
        "continuation_trust_ratio": float(args.continuation_trust_ratio),
        "continuation_mode": str(args.continuation_mode),
        "profile_solver": bool(args.profile_solver),
        "continuation_trace": "continuation_trace.json",
        "continuation_schedule": "continuation_schedule.json",
        "continuation_trace_attempts": int(len(continuation_trace)),
        "continuation_trace_rejections": int(
            sum(not bool(item.get("accepted", False)) for item in continuation_trace)
        ),
        "continuation_pc": {
            "coordinate": "log(phi)",
            "initial_ratio": float(args.pc_initial_ratio),
            "min_ratio": float(args.pc_min_ratio),
            "max_ratio": float(args.pc_max_ratio),
            "max_retries": int(args.pc_max_retries),
            "bridge_rows_included": bool(
                args.continuation_mode in ("fixed-bridges", "adaptive-pc")
            ),
        },
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
