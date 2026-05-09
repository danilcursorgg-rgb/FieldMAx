from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-12

@dataclass(frozen=True)
class AnthillRouterConfig:
    input_dim: int
    num_experts: int
    top_k: int = 2
    temperature: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


@dataclass(frozen=True)
class RoutingDecision:
    probabilities: torch.Tensor
    selected_experts: torch.Tensor
    load: torch.Tensor
    load_balance_loss: torch.Tensor
    entropy: torch.Tensor


class AnthillRouter(nn.Module):
    """Top-k softmax router with load-balancing regularization (PyTorch)."""
    
    def __init__(self, config: AnthillRouterConfig):
        super().__init__()
        self.config = config
        
        if config.seed is not None:
            torch.manual_seed(config.seed)
            
        # Gate matrix (input_dim, num_experts)
        self.gate = nn.Parameter(
            torch.randn(config.input_dim, config.num_experts) / math.sqrt(config.input_dim)
        )

    def forward(self, inputs: torch.Tensor) -> RoutingDecision:
        """
        Args:
            inputs: (batch, dim)
        Returns:
            RoutingDecision
        """
        # logits: (batch, num_experts)
        logits = torch.matmul(inputs, self.gate) / self.config.temperature
        
        # probabilities: (batch, num_experts)
        probabilities = F.softmax(logits, dim=-1)
        
        # Select top-k experts
        topk_probs, topk_indices = torch.topk(probabilities, self.config.top_k, dim=-1)
        
        # Create masked probabilities
        masked = torch.zeros_like(probabilities)
        masked.scatter_(1, topk_indices, topk_probs)
        
        # Re-normalize masked probabilities so they sum to 1
        masked_sum = masked.sum(dim=-1, keepdim=True).clamp(min=_EPS)
        masked = masked / masked_sum
        
        # Load balancing
        # load: (num_experts,)
        load = masked.mean(dim=0)
        
        # Load balancing loss: L_bal = \sum_j (load_j) * log(load_j * num_experts)
        # Forces uniform distribution across experts
        target_prob = 1.0 / self.config.num_experts
        load_balance_loss = torch.sum(load * torch.log(load.clamp(min=_EPS) / target_prob))
        
        # Entropy of routing (average across batch)
        entropy = -torch.mean(torch.sum(masked * torch.log(masked.clamp(min=_EPS, max=1.0)), dim=-1))
        
        return RoutingDecision(
            probabilities=masked,
            selected_experts=topk_indices,
            load=load,
            load_balance_loss=load_balance_loss,
            entropy=entropy,
        )


class LinearExpert(nn.Module):
    """Simple linear expert for Anthill."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        # Initialize identity or close to it
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class AnthillLayer(nn.Module):
    """
    Anthill (Mixture of Experts) layer integrating the router and multiple experts.
    """
    
    def __init__(self, config: AnthillRouterConfig, experts: nn.ModuleList):
        super().__init__()
        self.config = config
        self.router = AnthillRouter(config)
        self.experts = experts
        
        if len(self.experts) != self.config.num_experts:
            raise ValueError("number of experts must match router config")

    @classmethod
    def random(cls, config: AnthillRouterConfig) -> "AnthillLayer":
        if config.seed is not None:
            torch.manual_seed(config.seed)
        experts = nn.ModuleList([LinearExpert(config.input_dim) for _ in range(config.num_experts)])
        return cls(config, experts)

    @classmethod
    def identity(cls, config: AnthillRouterConfig) -> "AnthillLayer":
        # LinearExpert initializes to identity by default in our implementation
        experts = nn.ModuleList([LinearExpert(config.input_dim) for _ in range(config.num_experts)])
        return cls(config, experts)

    @property
    def total_parameter_count(self) -> int:
        return sum(expert.parameter_count for expert in self.experts)
        
    @property
    def active_parameter_count(self) -> int:
        # Expected active parameters per token
        if len(self.experts) == 0:
            return 0
        return self.experts[0].parameter_count * self.config.top_k

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, RoutingDecision]:
        decision = self.router(inputs)
        outputs = torch.zeros_like(inputs)
        
        # Compute only for selected experts (sparse execution)
        # In a real large scale MoE we'd use scatter/gather, but for prototyping
        # we can iterate experts and only compute if selected by any token.
        for expert_idx, expert in enumerate(self.experts):
            # Mask for tokens that selected this expert
            expert_prob = decision.probabilities[:, expert_idx:expert_idx+1] # (batch, 1)
            
            # Optimization: only run expert if prob > 0
            if expert_prob.max() > 0:
                expert_output = expert(inputs)
                outputs = outputs + expert_prob * expert_output
                
        return outputs, decision
