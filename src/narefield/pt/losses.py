from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-12

@dataclass(frozen=True)
class CognitiveInvariant:
    """Compact state descriptor used as the invariant penalty input."""
    norm: torch.Tensor
    variance: torch.Tensor
    sparsity: torch.Tensor

    @classmethod
    def from_state(cls, state: torch.Tensor, zero_threshold: float = 1e-6) -> "CognitiveInvariant":
        """
        Computes invariants from batched state (batch, dim).
        """
        device = state.device
        # Average L2 norm per token
        norm = torch.norm(state, p=2, dim=-1).mean() / torch.sqrt(
            torch.tensor(max(state.size(-1), 1), dtype=torch.float32, device=device)
        )
        # Average variance per token
        variance = torch.var(state, dim=-1).mean()
        # Average sparsity per token
        sparsity = (torch.abs(state) <= zero_threshold).float().mean()
        
        return cls(norm=norm, variance=variance, sparsity=sparsity)

    def as_tensor(self) -> torch.Tensor:
        return torch.stack([self.norm, self.variance, self.sparsity])


@dataclass(frozen=True)
class LossBreakdown:
    reconstruction: torch.Tensor
    invariant: torch.Tensor
    memory: torch.Tensor
    routing_balance: torch.Tensor
    total: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "reconstruction": self.reconstruction.item(),
            "invariant": self.invariant.item(),
            "memory": self.memory.item(),
            "routing_balance": self.routing_balance.item(),
            "total": self.total.item(),
        }

class PredictionEnergyLoss(nn.Module):
    """
    Prediction-error objective with information and routing regularizers (PyTorch).
    Implements: L = L_recon + lambda * L_invariant + gamma * L_memory + beta * L_routing
    """
    
    def __init__(
        self,
        invariant_weight: float = 0.01,
        memory_weight: float = 0.01,
        routing_weight: float = 0.01,
    ):
        super().__init__()
        self.invariant_weight = invariant_weight
        self.memory_weight = memory_weight
        self.routing_weight = routing_weight

    def forward(
        self,
        target: torch.Tensor,
        prediction: torch.Tensor,
        *,
        invariant: CognitiveInvariant | torch.Tensor | None = None,
        memory_stats: Mapping[str, float] | None = None,
        routing_loss: torch.Tensor | None = None,
    ) -> LossBreakdown:
        if target.shape != prediction.shape:
            raise ValueError(
                f"target and prediction must have the same shape, got "
                f"{target.shape} and {prediction.shape}"
            )

        # 1. Prediction error (Energy)
        reconstruction = F.mse_loss(prediction, target)
        
        # 2. Invariant penalty
        invariant_penalty = self._invariant_penalty(invariant, target.device)
        
        # 3. Memory penalty (if capacity is full, encourages compression)
        memory_penalty = torch.tensor(
            memory_stats.get("capacity_usage", 0.0) if memory_stats else 0.0,
            dtype=torch.float32,
            device=target.device
        )
        
        # 4. Routing penalty (load balancing loss from MoE router)
        if routing_loss is None:
            routing_penalty = torch.tensor(0.0, dtype=torch.float32, device=target.device)
        else:
            routing_penalty = routing_loss

        # Total free energy
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
            total=total,
        )

    def _invariant_penalty(self, invariant: CognitiveInvariant | torch.Tensor | None, device: torch.device) -> torch.Tensor:
        if invariant is None:
            return torch.tensor(0.0, dtype=torch.float32, device=device)
        if isinstance(invariant, CognitiveInvariant):
            values = invariant.as_tensor()
        else:
            values = invariant
        return torch.mean(torch.square(values))
