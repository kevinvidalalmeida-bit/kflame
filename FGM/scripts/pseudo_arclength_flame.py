"""Fixed-mesh pseudo-arclength seed for certified 1-D flame continuation.

This adapter intentionally stops before a flame is accepted.  It uses the
bordered pseudo-arclength system only to improve a continuation seed; the
ordinary V2 solver must subsequently solve, adapt the mesh and domain, and
apply its complete physical certificate.  Consequently a failed augmented
step has no effect on an FGM table other than falling back to the established
tangent/copy predictor.
"""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
V2_DIRECTORY = ROOT / "V2"
MATERIALS_DIRECTORY = V2_DIRECTORY / "materiales"
for directory in (V2_DIRECTORY, MATERIALS_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from config import FlameCase  # noqa: E402
from equations import (  # noqa: E402
    BlockTridiagJacobian,
    build_jacobian_steady,
    residual,
    solve_linear,
)
from problem import FreeFlameProblem  # noqa: E402
from solver import _state_step_scales, bound_step_limit  # noqa: E402
from species_backend_native import NativeSpeciesBackend  # noqa: E402

from fgm_continuation import (  # noqa: E402
    bordered_pseudo_arclength_direct_update,
    normalise_arc_tangent,
    pseudo_arclength_corrector,
)


def _fixed_mesh_problem(
    case: FlameCase,
    log_phi: float,
    z: np.ndarray,
    anchor_index: int,
    mech_data: Any,
) -> FreeFlameProblem:
    """Create the physical residual at one log-equivalence-ratio on one mesh."""
    phi = float(math.exp(float(log_phi)))
    if not math.isfinite(phi) or phi <= 0.0:
        raise ValueError("Pseudo-arclength requires a finite positive phi.")
    local_case = replace(case, phi=phi)
    problem = FreeFlameProblem(local_case, n_points=max(2, int(z.size)))
    problem.assume_finite_y = True
    problem.z = np.asarray(z, dtype=float).copy()
    problem.n_points = int(problem.z.size)
    problem.width = float(problem.z[-1] - problem.z[0])
    problem.solve_energy = True
    problem.jacobian_mode = "block_tridiag"
    problem.j_fixed = int(np.clip(anchor_index, 1, problem.n_points - 2))
    # The phase row remains on the same mesh node while lambda varies.  The
    # prescribed temperature follows the physical equilibrium state, exactly
    # as in the existing tangent chord predictor.
    problem.T_fixed_point = float(problem.anchor_T)
    problem.backend_factory = lambda prob: NativeSpeciesBackend(prob, mech_data=mech_data)
    problem.backend = problem.backend_factory(problem)
    return problem


def _block_tridiag_sparse(jacobian: BlockTridiagJacobian) -> sparse.bsr_matrix:
    """Materialize only the existing block-tridiagonal entries as BSR."""
    n_blocks = int(jacobian.n_blocks)
    block_size = int(jacobian.block_size)
    data: list[np.ndarray] = []
    indices: list[int] = []
    indptr = [0]
    for row in range(n_blocks):
        if row > 0:
            data.append(np.asarray(jacobian.lower[row - 1], dtype=float))
            indices.append(row - 1)
        data.append(np.asarray(jacobian.diag[row], dtype=float))
        indices.append(row)
        if row < n_blocks - 1:
            data.append(np.asarray(jacobian.upper[row], dtype=float))
            indices.append(row + 1)
        indptr.append(len(indices))
    return sparse.bsr_matrix(
        (
            np.asarray(data, dtype=float),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(n_blocks * block_size, n_blocks * block_size),
    )


def pseudo_arclength_flame_seed(
    *,
    case: FlameCase,
    target_phi: float,
    source: Mapping[str, Any],
    mech_data: Any,
    residual_guard_inf: float = 1.0e4,
    parameter_fd_step: float = 1.0e-4,
    max_iterations: int = 4,
    max_damping: int = 6,
) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    """Return an augmented-corrected seed, never a certified flame result.

    The source must carry V2's immediately preceding exact stationary LU.  A
    centered :math:`F_\\lambda` provides the first tangent; each augmented
    correction thereafter rebuilds an exact fixed-mesh Jacobian.  This avoids
    the mathematically invalid assumption that a chemical Jacobian changes by
    a low-rank matrix when the mixture changes.
    """
    target_phi = float(target_phi)
    metadata: dict[str, Any] = {
        "predictor_kind": "pseudo_arclength",
        "used": False,
        "reason": "invalid_source",
        "target_phi": target_phi,
    }
    if not math.isfinite(target_phi) or target_phi <= 0.0:
        return None, metadata
    try:
        source_phi = float(np.asarray(source["phi"], dtype=float).ravel()[0])
        z = np.asarray(source["z"], dtype=float)
        state_source = np.asarray(source["x"], dtype=float)
        handoff = source["continuation_linearization"]
        source_lu = handoff["lu"]
        anchor_index = int(handoff["anchor_index"])
        linear_state = np.asarray(handoff["state"], dtype=float)
        source_residual = np.asarray(handoff["residual"], dtype=float)
    except (KeyError, TypeError, ValueError, IndexError):
        return None, metadata
    if (
        not math.isfinite(source_phi)
        or source_phi <= 0.0
        or z.ndim != 1
        or z.size < 3
        or state_source.size == 0
        or state_source.shape != linear_state.shape
        or source_residual.shape != state_source.shape
        or not np.all(np.isfinite(state_source))
        or not np.all(np.isfinite(source_residual))
        or not np.allclose(state_source, linear_state, rtol=1.0e-11, atol=1.0e-13)
        or not isinstance(source_lu, dict)
        or source_lu.get("method") != "block_tridiag"
        or not 0 < anchor_index < z.size - 1
    ):
        return None, metadata

    lambda_source = float(math.log(source_phi))
    lambda_target = float(math.log(target_phi))
    delta_lambda = lambda_target - lambda_source
    if not math.isfinite(delta_lambda) or abs(delta_lambda) <= 1.0e-14:
        metadata["reason"] = "zero_parameter_step"
        return None, metadata
    fd_step = min(abs(delta_lambda) * 0.25, max(float(parameter_fd_step), 1.0e-6))
    if fd_step <= 0.0:
        metadata["reason"] = "invalid_fd_step"
        return None, metadata

    def make_problem(log_phi: float) -> FreeFlameProblem:
        return _fixed_mesh_problem(case, log_phi, z, anchor_index, mech_data)

    def residual_at(state: np.ndarray, log_phi: float) -> np.ndarray:
        problem = make_problem(log_phi)
        return np.asarray(residual(np.asarray(state, dtype=float), problem), dtype=float)

    def bordered_update_at(
        residual_value: np.ndarray,
        parameter_derivative: np.ndarray,
        state: np.ndarray,
        log_phi: float,
    ):
        problem = make_problem(log_phi)
        jacobian, _ = build_jacobian_steady(residual, np.asarray(state, dtype=float), problem)
        if not isinstance(jacobian, BlockTridiagJacobian):
            raise TypeError("Pseudo-arclength requires the V2 block-tridiagonal Jacobian.")
        return bordered_pseudo_arclength_direct_update(
            stationary_jacobian=_block_tridiag_sparse(jacobian),
            residual=residual_value,
            parameter_residual_derivative=parameter_derivative,
            state=state,
            parameter=log_phi,
            predicted_state=predicted_state,
            predicted_parameter=predicted_parameter,
            tangent_state=tangent_state,
            tangent_parameter=tangent_parameter,
            weights=weights,
        )

    try:
        source_problem = make_problem(lambda_source)
        # The source LU was assembled for the retained phase row and its
        # associated operational residual.  Use that paired parameter chord
        # for the *first* tangent, exactly as the established V2 tangent
        # predictor does.  A centered derivative here would compare a newly
        # reconstructed phase convention with the retained LU and can inject
        # the residual left by the source certificate into dF/dlambda.
        target_residual_at_source = residual_at(state_source, lambda_target)
        f_lambda = (target_residual_at_source - source_residual) / delta_lambda
        sensitivity = np.asarray(solve_linear(source_lu, -f_lambda), dtype=float)
        scales = _state_step_scales(state_source, source_problem, rdt=0.0)
        weights = 1.0 / np.maximum(scales, 1.0e-300)
        tangent_state, tangent_parameter, tangent_norm = normalise_arc_tangent(
            sensitivity, 1.0, weights
        )
        if delta_lambda * tangent_parameter < 0.0:
            tangent_state, tangent_parameter = -tangent_state, -tangent_parameter
        arclength_step = delta_lambda / tangent_parameter
        predicted_state = state_source + arclength_step * tangent_state
        predicted_parameter = lambda_source + arclength_step * tangent_parameter
        # Project a predictor only once. The arclength constraint is then
        # evaluated from that physical prediction; no hidden clipping occurs
        # inside a Newton step.
        predicted_state = make_problem(predicted_parameter).reset_bad_values(predicted_state)

        def bounded_step(state, step, log_phi, parameter_step):
            problem = make_problem(log_phi)
            alpha, _ = bound_step_limit(state, step, problem)
            return float(alpha if alpha >= 1.0 else 0.99 * alpha)

        corrected = pseudo_arclength_corrector(
            predicted_state=predicted_state,
            predicted_parameter=predicted_parameter,
            tangent_state=tangent_state,
            tangent_parameter=tangent_parameter,
            weights=weights,
            residual_at=residual_at,
            bordered_update_at=bordered_update_at,
            bound_step=bounded_step,
            parameter_fd_step=fd_step,
            residual_tolerance=float(residual_guard_inf),
            constraint_tolerance=1.0e-8,
            max_iterations=int(max_iterations),
            max_damping=int(max_damping),
        )
    except Exception as exc:
        metadata["reason"] = f"setup_error:{type(exc).__name__}"
        return None, metadata

    parameter_error = float(corrected.parameter - lambda_target)
    metadata.update({
        "reason": str(corrected.reason),
        "converged": bool(corrected.converged),
        "source_phi": source_phi,
        "source_log_phi": lambda_source,
        "target_log_phi": lambda_target,
        "corrected_log_phi": float(corrected.parameter),
        "corrected_phi": float(math.exp(corrected.parameter)),
        "parameter_error_log_phi": parameter_error,
        "tangent_norm": float(tangent_norm),
        "arclength_step": float(arclength_step),
        "iterations": int(corrected.iterations),
        "residual_inf": float(corrected.residual_inf),
        "constraint": float(corrected.constraint),
        "trace": [dict(item) for item in corrected.trace],
    })
    # A true pseudo-arclength step can land before the originally requested
    # coordinate.  That is a legitimate *bridge* when it lies on the directed
    # interval: it must be certified as its own flamelet and retained in the
    # FGM sequence.  Only a step outside the requested interval is rejected.
    interval_tolerance = max(5.0e-5, 0.01 * abs(delta_lambda))
    lower = min(lambda_source, lambda_target) - interval_tolerance
    upper = max(lambda_source, lambda_target) + interval_tolerance
    if not corrected.converged or not lower <= corrected.parameter <= upper:
        metadata["reason"] = (
            "outside_requested_interval" if corrected.converged else str(corrected.reason)
        )
        return None, metadata
    is_bridge = abs(parameter_error) > 1.0e-10
    metadata["bridge"] = bool(is_bridge)
    metadata["requested_phi"] = target_phi
    metadata["used"] = True
    return {
        "phi": np.array([float(math.exp(corrected.parameter))], dtype=float),
        "z": z.copy(),
        "x": np.asarray(corrected.state, dtype=float).copy(),
    }, metadata
