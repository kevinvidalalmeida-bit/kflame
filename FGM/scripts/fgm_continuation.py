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

import numpy as np


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
