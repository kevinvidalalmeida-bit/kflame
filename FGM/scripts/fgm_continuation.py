"""Solver-independent control logic for parameter continuation.

The controller deliberately knows nothing about flames, chemistry, meshes, or
the nonlinear solver.  A caller supplies an accepted corrected state and its
predictor defect; this module decides the next trial point in ``log(phi)``.
Keeping this policy separate makes the same continuation path replayable with
V2 and with an independent reference solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import median
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


@dataclass(frozen=True)
class ContinuationConfig:
    """Numerical policy for adaptive continuation in ``lambda = log(phi)``."""

    initial_ratio: float = 1.15
    min_ratio: float = 1.02
    max_ratio: float = 1.20
    safety: float = 0.90
    min_growth: float = 0.50
    max_growth: float = 1.25
    calibration_samples: int = 3
    max_retries: int = 4
    epsilon: float = 1.0e-300

    def __post_init__(self) -> None:
        if self.min_ratio <= 1.0:
            raise ValueError("min_ratio must be greater than one.")
        if not self.min_ratio <= self.initial_ratio <= self.max_ratio:
            raise ValueError("Require min_ratio <= initial_ratio <= max_ratio.")
        if not 0.0 < self.safety <= 1.0:
            raise ValueError("safety must lie in (0, 1].")
        if not 0.0 < self.min_growth <= self.max_growth:
            raise ValueError("Invalid growth limits.")
        if self.calibration_samples < 1 or self.max_retries < 0:
            raise ValueError("Invalid calibration or retry count.")

    @property
    def initial_log_step(self) -> float:
        return math.log(self.initial_ratio)

    @property
    def min_log_step(self) -> float:
        return math.log(self.min_ratio)

    @property
    def max_log_step(self) -> float:
        return math.log(self.max_ratio)


@dataclass(frozen=True)
class ContinuationProposal:
    """A proposed corrected flamelet between an accepted state and a target."""

    phi_from: float
    phi_target: float
    phi_trial: float
    log_step: float
    is_bridge: bool
    retry: int


@dataclass
class AdaptiveContinuationController:
    """Accept/reject and step-size policy for a corrected continuation path."""

    config: ContinuationConfig = field(default_factory=ContinuationConfig)
    log_step: float | None = None
    defect_reference: float | None = None
    _secant_defects: list[float] = field(default_factory=list)
    _retry: int = 0

    def __post_init__(self) -> None:
        if self.log_step is None:
            self.log_step = self.config.initial_log_step
        self.log_step = self._clip_step(float(self.log_step))

    def _clip_step(self, value: float) -> float:
        return min(self.config.max_log_step, max(self.config.min_log_step, value))

    def propose(self, phi_from: float, phi_target: float) -> ContinuationProposal:
        """Advance at most one adaptive log-step toward ``phi_target``."""
        if phi_from <= 0.0 or phi_target <= 0.0:
            raise ValueError("Continuation requires strictly positive phi values.")
        lam_from = math.log(float(phi_from))
        lam_target = math.log(float(phi_target))
        remaining = lam_target - lam_from
        direction = 1.0 if remaining >= 0.0 else -1.0
        used_step = min(abs(remaining), float(self.log_step))
        lam_trial = lam_from + direction * used_step
        # Preserve an explicitly requested endpoint bit-for-bit when reached.
        is_bridge = used_step + 1.0e-14 < abs(remaining)
        phi_trial = float(math.exp(lam_trial)) if is_bridge else float(phi_target)
        return ContinuationProposal(
            phi_from=float(phi_from),
            phi_target=float(phi_target),
            phi_trial=phi_trial,
            log_step=float(used_step),
            is_bridge=bool(is_bridge),
            retry=int(self._retry),
        )

    def accept(self, prediction_defect: float, predictor_kind: str) -> float:
        """Record a certified correction and return the next log-step.

        The controller does not reject a physically certified flame merely
        because its predictor was poor.  The defect changes only the *next*
        step.  Early copy predictions do not calibrate the local-predictor reference.
        """
        defect = float(prediction_defect)
        if (
            predictor_kind.startswith(("secant", "tangent"))
            and math.isfinite(defect)
            and defect > 0.0
        ):
            if self.defect_reference is None:
                self._secant_defects.append(defect)
                if len(self._secant_defects) >= self.config.calibration_samples:
                    values = self._secant_defects[: self.config.calibration_samples]
                    self.defect_reference = max(float(median(values)), self.config.epsilon)

        if self.defect_reference is not None and math.isfinite(defect) and defect > 0.0:
            factor = self.config.safety * math.sqrt(self.defect_reference / max(defect, self.config.epsilon))
            factor = min(self.config.max_growth, max(self.config.min_growth, factor))
            self.log_step = self._clip_step(float(self.log_step) * factor)
        self._retry = 0
        return float(self.log_step)

    def reject(self) -> tuple[float, int, bool]:
        """Shrink the next trial and report whether the retry budget remains."""
        self.log_step = self._clip_step(float(self.log_step) * 0.5)
        self._retry += 1
        can_retry = (
            self._retry < self.config.max_retries
            and float(self.log_step) > self.config.min_log_step + 1.0e-14
        )
        return float(self.log_step), int(self._retry), bool(can_retry)

    def snapshot(self) -> dict[str, float | int | None]:
        return {
            "log_step": float(self.log_step),
            "ratio": float(math.exp(float(self.log_step))),
            "defect_reference": self.defect_reference,
            "secant_calibration_count": int(len(self._secant_defects)),
            "retry": int(self._retry),
        }


@dataclass(frozen=True)
class BorderedArcLengthUpdate:
    """One exact Schur-complement update of a bordered continuation system.

    The stationary flame Jacobian is never treated as a low-rank update.  At
    a current augmented iterate ``(x, lambda)``, the bordered Newton system is

    ``[J  F_lambda; t_x^T  t_lambda] [dx; dlambda] = -[F; g]``.

    With an already factored ``J``, two ordinary right-hand sides and a scalar
    Schur complement are sufficient.  This object records the resulting
    correction together with the scalar denominator so a caller can reject a
    near-singular augmented step explicitly.
    """

    state_step: np.ndarray
    parameter_step: float
    constraint: float
    schur_denominator: float
    tangent_norm: float


@dataclass(frozen=True)
class PseudoArcLengthCorrectorResult:
    """Outcome of a damped pseudo-arclength corrector on a fixed mesh.

    It is deliberately not a physical-flame certificate. A caller must still
    apply its ordinary nonlinear solve, mesh/domain checks, and acceptance
    predicate before retaining the state in an FGM table.
    """

    state: np.ndarray
    parameter: float
    converged: bool
    iterations: int
    residual_inf: float
    constraint: float
    reason: str
    trace: tuple[dict[str, float], ...]


def _weighted_inner(
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Dimensionless mean inner product used by pseudo-arclength.

    ``weights`` are inverse physical scales, e.g. the same per-variable
    absolute-plus-relative scales used by the nonlinear solver.  Applying the
    metric here makes the arclength condition independent of whether a state
    component is represented in kelvin, velocity, or mass fraction.
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if left.shape != right.shape or left.shape != weights.shape or left.size == 0:
        raise ValueError("Pseudo-arclength vectors must have the same nonempty shape.")
    value = float(np.mean((weights * left) * (weights * right)))
    if not math.isfinite(value):
        raise ValueError("Non-finite pseudo-arclength metric.")
    return value


def normalise_arc_tangent(
    state_sensitivity: np.ndarray,
    parameter_sensitivity: float,
    weights: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Return a unit tangent under the physical continuation metric.

    A first-order flame sensitivity normally has ``parameter_sensitivity=1``.
    The separate parameter contribution deliberately remains dimensionless;
    the state part is scaled by ``weights`` before normalisation.
    """
    state_sensitivity = np.asarray(state_sensitivity, dtype=float)
    weights = np.asarray(weights, dtype=float)
    parameter_sensitivity = float(parameter_sensitivity)
    if (
        state_sensitivity.shape != weights.shape
        or state_sensitivity.size == 0
        or not np.all(np.isfinite(state_sensitivity))
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
        or not math.isfinite(parameter_sensitivity)
    ):
        raise ValueError("Invalid tangent or physical scales for pseudo-arclength.")
    norm_sq = _weighted_inner(state_sensitivity, state_sensitivity, weights)
    norm_sq += parameter_sensitivity * parameter_sensitivity
    tangent_norm = math.sqrt(norm_sq)
    if not math.isfinite(tangent_norm) or tangent_norm <= np.finfo(float).tiny:
        raise ValueError("Degenerate pseudo-arclength tangent.")
    return (
        state_sensitivity / tangent_norm,
        parameter_sensitivity / tangent_norm,
        float(tangent_norm),
    )


def bordered_pseudo_arclength_update(
    *,
    solve_jacobian,
    residual: np.ndarray,
    parameter_residual_derivative: np.ndarray,
    state: np.ndarray,
    parameter: float,
    predicted_state: np.ndarray,
    predicted_parameter: float,
    tangent_state: np.ndarray,
    tangent_parameter: float,
    weights: np.ndarray,
    denominator_floor: float = 1.0e-12,
) -> BorderedArcLengthUpdate:
    """Solve one bordered pseudo-arclength Newton correction exactly.

    ``solve_jacobian(rhs)`` must solve an *exact current stationary* Jacobian
    system.  The function does not factorise a second global matrix and never
    assumes that changing chemistry or transport is low rank.  A caller may
    use the update only after checking bounds and the usual physical flame
    certificate.

    The returned formula is

    ``a = J^-1(-F)``, ``b = J^-1(F_lambda)``,
    ``dlambda = (-g - <t_x,a>) / (t_lambda - <t_x,b>)``,
    ``dx = a - b dlambda``.
    """
    residual = np.asarray(residual, dtype=float)
    f_lambda = np.asarray(parameter_residual_derivative, dtype=float)
    state = np.asarray(state, dtype=float)
    predicted_state = np.asarray(predicted_state, dtype=float)
    tangent_state = np.asarray(tangent_state, dtype=float)
    weights = np.asarray(weights, dtype=float)
    parameter = float(parameter)
    predicted_parameter = float(predicted_parameter)
    tangent_parameter = float(tangent_parameter)
    denominator_floor = float(denominator_floor)

    vectors = (residual, f_lambda, state, predicted_state, tangent_state, weights)
    if (
        any(vector.shape != state.shape for vector in vectors)
        or state.size == 0
        or not all(np.all(np.isfinite(vector)) for vector in vectors)
        or not math.isfinite(parameter)
        or not math.isfinite(predicted_parameter)
        or not math.isfinite(tangent_parameter)
        or not math.isfinite(denominator_floor)
        or denominator_floor <= 0.0
    ):
        raise ValueError("Invalid bordered pseudo-arclength inputs.")

    tangent_norm_sq = _weighted_inner(tangent_state, tangent_state, weights)
    tangent_norm_sq += tangent_parameter * tangent_parameter
    if tangent_norm_sq <= np.finfo(float).tiny:
        raise ValueError("Degenerate bordered pseudo-arclength tangent.")

    constraint = _weighted_inner(
        tangent_state, state - predicted_state, weights
    ) + tangent_parameter * (parameter - predicted_parameter)
    a = np.asarray(solve_jacobian(-residual), dtype=float)
    b = np.asarray(solve_jacobian(f_lambda), dtype=float)
    if (
        a.shape != state.shape
        or b.shape != state.shape
        or not np.all(np.isfinite(a))
        or not np.all(np.isfinite(b))
    ):
        raise ValueError("Jacobian solve failed inside bordered pseudo-arclength.")

    tangent_a = _weighted_inner(tangent_state, a, weights)
    tangent_b = _weighted_inner(tangent_state, b, weights)
    denominator = tangent_parameter - tangent_b
    if not math.isfinite(denominator) or abs(denominator) <= denominator_floor:
        raise ValueError("Near-singular pseudo-arclength Schur complement.")
    parameter_step = (-constraint - tangent_a) / denominator
    state_step = a - b * parameter_step
    if not math.isfinite(parameter_step) or not np.all(np.isfinite(state_step)):
        raise ValueError("Non-finite bordered pseudo-arclength correction.")
    return BorderedArcLengthUpdate(
        state_step=np.asarray(state_step, dtype=float),
        parameter_step=float(parameter_step),
        constraint=float(constraint),
        schur_denominator=float(denominator),
        tangent_norm=float(math.sqrt(tangent_norm_sq)),
    )


def bordered_pseudo_arclength_direct_update(
    *,
    stationary_jacobian,
    residual: np.ndarray,
    parameter_residual_derivative: np.ndarray,
    state: np.ndarray,
    parameter: float,
    predicted_state: np.ndarray,
    predicted_parameter: float,
    tangent_state: np.ndarray,
    tangent_parameter: float,
    weights: np.ndarray,
) -> BorderedArcLengthUpdate:
    """Solve the full sparse bordered system, including at a simple fold.

    Unlike :func:`bordered_pseudo_arclength_update`, this implementation does
    not require an invertible stationary ``J``. It factorizes the augmented
    sparse matrix directly,

    ``[[J, F_lambda], [t_x^T W^2 / n, t_lambda]]``.

    This is the mathematically appropriate pseudo-arclength form near a fold:
    the bordered matrix can be nonsingular even when ``J`` is singular. It is
    an experimental robustness route, not a claim that the flame Jacobian
    itself receives a low-rank update.
    """
    residual = np.asarray(residual, dtype=float)
    f_lambda = np.asarray(parameter_residual_derivative, dtype=float)
    state = np.asarray(state, dtype=float)
    predicted_state = np.asarray(predicted_state, dtype=float)
    tangent_state = np.asarray(tangent_state, dtype=float)
    weights = np.asarray(weights, dtype=float)
    parameter = float(parameter)
    predicted_parameter = float(predicted_parameter)
    tangent_parameter = float(tangent_parameter)
    if (
        residual.shape != state.shape
        or f_lambda.shape != state.shape
        or predicted_state.shape != state.shape
        or tangent_state.shape != state.shape
        or weights.shape != state.shape
        or state.size == 0
        or not all(np.all(np.isfinite(v)) for v in (
            residual, f_lambda, state, predicted_state, tangent_state, weights
        ))
        or np.any(weights <= 0.0)
        or not math.isfinite(parameter)
        or not math.isfinite(predicted_parameter)
        or not math.isfinite(tangent_parameter)
    ):
        raise ValueError("Invalid direct bordered pseudo-arclength inputs.")
    matrix = sparse.csc_matrix(stationary_jacobian, dtype=float)
    if matrix.shape != (state.size, state.size):
        raise ValueError("Bordered stationary Jacobian has incompatible shape.")
    tangent_row = sparse.csr_matrix(
        ((weights * weights * tangent_state / state.size).reshape(1, -1))
    )
    augmented = sparse.bmat(
        [
            [matrix, sparse.csc_matrix(f_lambda.reshape(-1, 1))],
            [tangent_row, sparse.csr_matrix([[tangent_parameter]])],
        ],
        format="csc",
    )
    constraint = _weighted_inner(tangent_state, state - predicted_state, weights) + (
        tangent_parameter * (parameter - predicted_parameter)
    )
    rhs = np.concatenate((-residual, np.array([-constraint], dtype=float)))
    solution = np.asarray(splu(augmented).solve(rhs), dtype=float)
    if solution.shape != (state.size + 1,) or not np.all(np.isfinite(solution)):
        raise ValueError("Non-finite direct bordered pseudo-arclength solution.")
    tangent_norm_sq = _weighted_inner(tangent_state, tangent_state, weights)
    tangent_norm_sq += tangent_parameter * tangent_parameter
    return BorderedArcLengthUpdate(
        state_step=solution[:-1],
        parameter_step=float(solution[-1]),
        constraint=float(constraint),
        # There is no scalar Schur denominator in the direct bordered solve.
        schur_denominator=float("nan"),
        tangent_norm=float(math.sqrt(tangent_norm_sq)),
    )


def pseudo_arclength_corrector(
    *,
    predicted_state: np.ndarray,
    predicted_parameter: float,
    tangent_state: np.ndarray,
    tangent_parameter: float,
    weights: np.ndarray,
    residual_at: Callable[[np.ndarray, float], np.ndarray],
    linear_solve_at: Callable[[np.ndarray, float], Callable[[np.ndarray], np.ndarray]] | None = None,
    bordered_update_at: Callable[[np.ndarray, np.ndarray, np.ndarray, float], BorderedArcLengthUpdate] | None = None,
    bound_step: Callable[[np.ndarray, np.ndarray, float, float], float] | None = None,
    parameter_fd_step: float = 1.0e-4,
    residual_tolerance: float = 1.0e4,
    constraint_tolerance: float = 1.0e-8,
    max_iterations: int = 5,
    max_damping: int = 6,
) -> PseudoArcLengthCorrectorResult:
    """Apply a damped bordered corrector without prescribing a solver type.

    ``residual_at`` together with either a stationary-Jacobian solve callback
    or a direct bordered-update callback define the physical family
    :math:`F(x,\\lambda)=0`, making the routine applicable to equivalence
    ratio, pressure, inlet temperature, or enthalpy. A fresh exact stationary
    Jacobian is requested at each augmented iteration. That conservative
    choice makes this first branch-following experiment auditable; it does
    not assume that a flame Jacobian changes by a low-rank update.

    The returned state remains only a seed. Mesh adaptation and the physical
    certificate are intentionally outside this routine.
    """
    state = np.asarray(predicted_state, dtype=float).copy()
    predicted_state = np.asarray(predicted_state, dtype=float)
    tangent_state = np.asarray(tangent_state, dtype=float)
    weights = np.asarray(weights, dtype=float)
    parameter = float(predicted_parameter)
    tangent_parameter = float(tangent_parameter)
    fd_step = float(parameter_fd_step)
    residual_tolerance = float(residual_tolerance)
    constraint_tolerance = float(constraint_tolerance)
    if (
        state.size == 0
        or state.shape != predicted_state.shape
        or tangent_state.shape != state.shape
        or weights.shape != state.shape
        or not np.all(np.isfinite(state))
        or not np.all(np.isfinite(tangent_state))
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
        or not math.isfinite(parameter)
        or not math.isfinite(tangent_parameter)
        or not math.isfinite(fd_step)
        or fd_step <= 0.0
        or not math.isfinite(residual_tolerance)
        or residual_tolerance <= 0.0
        or not math.isfinite(constraint_tolerance)
        or constraint_tolerance <= 0.0
        or int(max_iterations) < 1
        or int(max_damping) < 0
        or (linear_solve_at is None and bordered_update_at is None)
        or (linear_solve_at is not None and bordered_update_at is not None)
    ):
        raise ValueError("Invalid pseudo-arclength corrector configuration.")

    trace: list[dict[str, float]] = []
    last_constraint = float("inf")
    last_residual_inf = float("inf")

    def constraint_at(x_value: np.ndarray, lambda_value: float) -> float:
        return _weighted_inner(tangent_state, x_value - predicted_state, weights) + (
            tangent_parameter * (lambda_value - predicted_parameter)
        )

    for iteration in range(1, int(max_iterations) + 1):
        try:
            residual = np.asarray(residual_at(state, parameter), dtype=float)
        except Exception as exc:
            return PseudoArcLengthCorrectorResult(
                state, parameter, False, iteration - 1, last_residual_inf,
                last_constraint, f"residual_error:{type(exc).__name__}", tuple(trace),
            )
        if residual.shape != state.shape or not np.all(np.isfinite(residual)):
            return PseudoArcLengthCorrectorResult(
                state, parameter, False, iteration - 1, last_residual_inf,
                last_constraint, "nonfinite_residual", tuple(trace),
            )
        constraint = constraint_at(state, parameter)
        residual_inf = float(np.linalg.norm(residual, ord=np.inf))
        last_constraint, last_residual_inf = constraint, residual_inf
        if residual_inf <= residual_tolerance and abs(constraint) <= constraint_tolerance:
            return PseudoArcLengthCorrectorResult(
                state, parameter, True, iteration - 1, residual_inf,
                constraint, "converged", tuple(trace),
            )

        try:
            f_plus = np.asarray(residual_at(state, parameter + fd_step), dtype=float)
            f_minus = np.asarray(residual_at(state, parameter - fd_step), dtype=float)
            f_lambda = (f_plus - f_minus) / (2.0 * fd_step)
            if (
                f_plus.shape != state.shape
                or f_minus.shape != state.shape
                or not np.all(np.isfinite(f_lambda))
            ):
                raise ValueError("Invalid parameter residual derivative.")
            if bordered_update_at is not None:
                update = bordered_update_at(residual, f_lambda, state, parameter)
            else:
                assert linear_solve_at is not None
                solve_jacobian = linear_solve_at(state, parameter)
                update = bordered_pseudo_arclength_update(
                    solve_jacobian=solve_jacobian,
                    residual=residual,
                    parameter_residual_derivative=f_lambda,
                    state=state,
                    parameter=parameter,
                    predicted_state=predicted_state,
                    predicted_parameter=predicted_parameter,
                    tangent_state=tangent_state,
                    tangent_parameter=tangent_parameter,
                    weights=weights,
                )
        except Exception as exc:
            return PseudoArcLengthCorrectorResult(
                state, parameter, False, iteration - 1, residual_inf,
                constraint, f"linearization_error:{type(exc).__name__}", tuple(trace),
            )

        maximum_alpha = 1.0
        if bound_step is not None:
            try:
                maximum_alpha = float(
                    bound_step(state, update.state_step, parameter, update.parameter_step)
                )
            except Exception as exc:
                return PseudoArcLengthCorrectorResult(
                    state, parameter, False, iteration - 1, residual_inf,
                    constraint, f"bounds_error:{type(exc).__name__}", tuple(trace),
                )
        if not math.isfinite(maximum_alpha) or maximum_alpha <= 0.0:
            return PseudoArcLengthCorrectorResult(
                state, parameter, False, iteration - 1, residual_inf,
                constraint, "no_bounded_step", tuple(trace),
            )
        maximum_alpha = min(1.0, maximum_alpha)
        old_merit = max(residual_inf / residual_tolerance, abs(constraint) / constraint_tolerance)
        accepted = False
        for level in range(int(max_damping) + 1):
            alpha = maximum_alpha * (2.0 ** (-0.5 * level))
            trial_state = state + alpha * update.state_step
            trial_parameter = parameter + alpha * update.parameter_step
            if not np.all(np.isfinite(trial_state)) or not math.isfinite(trial_parameter):
                continue
            try:
                trial_residual = np.asarray(
                    residual_at(trial_state, trial_parameter), dtype=float
                )
            except Exception:
                continue
            if trial_residual.shape != state.shape or not np.all(np.isfinite(trial_residual)):
                continue
            trial_constraint = constraint_at(trial_state, trial_parameter)
            trial_residual_inf = float(np.linalg.norm(trial_residual, ord=np.inf))
            trial_merit = max(
                trial_residual_inf / residual_tolerance,
                abs(trial_constraint) / constraint_tolerance,
            )
            if trial_merit < old_merit:
                trace.append({
                    "iteration": float(iteration),
                    "damping_alpha": float(alpha),
                    "residual_inf": trial_residual_inf,
                    "constraint": float(trial_constraint),
                    "schur_denominator": float(update.schur_denominator),
                    "parameter": float(trial_parameter),
                })
                state, parameter = trial_state, float(trial_parameter)
                last_constraint, last_residual_inf = trial_constraint, trial_residual_inf
                accepted = True
                break
        if not accepted:
            return PseudoArcLengthCorrectorResult(
                state, parameter, False, iteration, residual_inf,
                constraint, "damping_rejected", tuple(trace),
            )

    return PseudoArcLengthCorrectorResult(
        state, parameter, False, int(max_iterations), last_residual_inf,
        last_constraint, "iteration_limit", tuple(trace),
    )
