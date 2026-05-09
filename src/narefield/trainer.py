from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import NAREFieldModel, NAREStepResult


@dataclass(frozen=True)
class FreeEnergyUpdate:
    step: int
    total_energy: float
    prediction_error: float
    invariant_penalty: float
    memory_capacity: float
    routing_balance: float
    cognitive_temperature: float
    routing_load_std: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "step": self.step,
            "total_energy": self.total_energy,
            "prediction_error": self.prediction_error,
            "invariant_penalty": self.invariant_penalty,
            "memory_capacity": self.memory_capacity,
            "routing_balance": self.routing_balance,
            "cognitive_temperature": self.cognitive_temperature,
            "routing_load_std": self.routing_load_std,
        }


@dataclass
class FreeEnergyTrainer:
    """Runs local NARE updates against the explicit free-energy objective."""

    model: NAREFieldModel
    router_learning_rate: float = 0.01
    history: list[FreeEnergyUpdate] = field(default_factory=list)

    def train_step(self, inputs: np.ndarray, target: np.ndarray | None = None) -> tuple[NAREStepResult, FreeEnergyUpdate]:
        result = self.model.step(inputs, target, learn=True)
        target_load = np.full(self.model.config.num_experts, 1.0 / self.model.config.num_experts)
        self.model.anthill.router.local_update(result.latent, target_load, learning_rate=self.router_learning_rate)
        update = FreeEnergyUpdate(
            step=len(self.history),
            total_energy=result.loss.total,
            prediction_error=result.loss.reconstruction,
            invariant_penalty=result.loss.invariant,
            memory_capacity=result.loss.memory,
            routing_balance=result.loss.routing_balance,
            cognitive_temperature=float(result.metrics["cognitive_temperature"]),
            routing_load_std=float(result.metrics["routing_load_std"]),
        )
        self.history.append(update)
        return result, update

    def fit(self, batches: list[np.ndarray] | tuple[np.ndarray, ...]) -> list[FreeEnergyUpdate]:
        updates: list[FreeEnergyUpdate] = []
        for batch in batches:
            _, update = self.train_step(batch)
            updates.append(update)
        return updates
