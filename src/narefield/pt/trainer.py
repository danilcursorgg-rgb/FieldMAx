"""PyTorch FreeEnergyTrainer — gradient-based optimization of the full NARE-Field model."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.optim as optim

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


class FreeEnergyTrainer:
    """Runs gradient-based optimization against the explicit free-energy objective."""

    def __init__(
        self,
        model: NAREFieldModel,
        lr: float = 1e-3,
        router_lr: float = 3e-3,
    ):
        self.model = model
        self.history: list[FreeEnergyUpdate] = []

        # Separate parameter groups: router gate gets higher lr for load-balancing
        router_params = list(model.anthill.router.parameters())
        router_ids = {id(p) for p in router_params}
        other_params = [p for p in model.parameters() if id(p) not in router_ids]

        self.optimizer = optim.Adam([
            {"params": other_params, "lr": lr},
            {"params": router_params, "lr": router_lr},
        ])

    def train_step(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[NAREStepResult, FreeEnergyUpdate]:
        self.model.train()
        self.optimizer.zero_grad()

        result = self.model(inputs, target, learn=True)

        # Backward pass through the differentiable loss
        result.loss.total.backward()
        self.optimizer.step()

        update = FreeEnergyUpdate(
            step=len(self.history),
            total_energy=result.loss.total.item(),
            prediction_error=result.loss.reconstruction.item(),
            invariant_penalty=result.loss.invariant.item(),
            memory_capacity=result.loss.memory.item(),
            routing_balance=result.loss.routing_balance.item(),
            cognitive_temperature=result.metrics["cognitive_temperature"],
            routing_load_std=result.metrics["routing_load_std"],
        )
        self.history.append(update)
        return result, update

    def fit(
        self,
        batches: list[torch.Tensor] | tuple[torch.Tensor, ...],
        targets: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
    ) -> list[FreeEnergyUpdate]:
        updates: list[FreeEnergyUpdate] = []
        for i, batch in enumerate(batches):
            t = targets[i] if targets is not None else None
            _, update = self.train_step(batch, t)
            updates.append(update)
        return updates

    @torch.no_grad()
    def evaluate(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> NAREStepResult:
        self.model.eval()
        return self.model(inputs, target, learn=False)
