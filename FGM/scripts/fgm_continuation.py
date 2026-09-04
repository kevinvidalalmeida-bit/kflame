"""Solver-independent control logic for parameter continuation.

The controller deliberately knows nothing about flames, chemistry, meshes, or
the nonlinear solver. A caller supplies an accepted corrected state and its
predictor defect; this module decides the next trial point in ``log(phi)``.
Keeping this policy separate makes the same continuation path replayable with
V2 and with an independent reference solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import median


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

        A physically certified flame is never rejected merely because its
        predictor was poor. The defect changes only the *next* step.
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
            factor = self.config.safety * math.sqrt(
                self.defect_reference / max(defect, self.config.epsilon)
            )
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
