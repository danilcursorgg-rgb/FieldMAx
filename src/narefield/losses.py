from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import numpy as np

_EPS = 1e-12

@dataclass(frozen=True)
class CognitiveInvariant:
    """Compact state descriptor used as the invariant penalty input."""
    norm: float
    variance: float
    sparsity: float

    @classmethod
    def from_state(cls, state: np.ndarray, zero_threshold: float = 1e-6) -> "CognitiveInvariant":
        values = np.asarray(state, dtype=float)
        return cls(
            norm=float(np.linalg.norm(values) / np.sqrt(max(values.size, 1))),
            variance=float(np.var(values)),
            sparsity=float(np.mean(np.abs(values) <= zero_threshold)),
        )

    def as_array(self) -> np.ndarray:
        return np.array([self.norm, self.variance, self.sparsity], dtype=float)


@dataclass(frozen=True)
class LossBreakdown:
    reconstruction: float
    invariant: float
    memory: float
    routing_balance: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return {
            "reconstruction": self.reconstruction,
            "invariant": self.invariant,
            "memory": self.memory,
            "routing_balance": self.routing_balance,
            "total": self.total,
        }

@dataclass(frozen=True)
class PredictionEnergyLoss:
    """Prediction-error objective with information and routing regularizers."""
    invariant_weight: float = 0.01
    memory_weight: float = 0.01
    routing_weight: float = 0.01

    def __call__(
        self,
        target: np.ndarray,
        prediction: np.ndarray,
        *,
        invariant: CognitiveInvariant | np.ndarray | None = None,
        memory_stats: Mapping[str, float] | None = None,
        routing_load: np.ndarray | None = None,
    ) -> LossBreakdown:
        target_values = np.asarray(target, dtype=float)
        prediction_values = np.asarray(prediction, dtype=float)
        if target_values.shape != prediction_values.shape:
            raise ValueError(
                f"target and prediction must have the same shape, got "
                f"{target_values.shape} and {prediction_values.shape}"
            )

        reconstruction = float(np.mean(np.square(target_values - prediction_values)))
        invariant_penalty = self._invariant_penalty(invariant)
        memory_penalty = float(memory_stats.get("capacity_usage", 0.0)) if memory_stats else 0.0
        routing_penalty = self.load_balance_loss(routing_load) if routing_load is not None else 0.0
        total = (
            reconstruction
            + self.invariant_weight * invariant_penalty
            + self.memory_weight * memory_penalty
            + self.routing_weight * routing_penalty
        )
        return LossBreakdown(
            reconstruction=reconstruction,
            invariant=invariant_penalty,
            memory=memory_penalty,
            routing_balance=routing_penalty,
            total=float(total),
        )

    @staticmethod
    def load_balance_loss(load: np.ndarray) -> float:
        values = np.asarray(load, dtype=float).reshape(-1)
        if values.size == 0:
            return 0.0
        total = float(np.sum(values))
        if total < _EPS:
            return float(np.log(max(values.size, 1)))
        probabilities = np.clip(values / total, _EPS, 1.0)
        uniform = 1.0 / values.size
        kl_div = float(np.sum(probabilities * np.log(probabilities / uniform)))
        return kl_div

    @staticmethod
    def _invariant_penalty(invariant: CognitiveInvariant | np.ndarray | None) -> float:
        if invariant is None:
            return 0.0
        values = invariant.as_array() if isinstance(invariant, CognitiveInvariant) else np.asarray(invariant, dtype=float)
        return float(np.mean(np.square(values)))
