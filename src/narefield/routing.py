from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .losses import PredictionEnergyLoss

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
    probabilities: np.ndarray
    selected_experts: np.ndarray
    load: np.ndarray
    load_balance_loss: float
    entropy: float


@dataclass
class AnthillRouter:
    """Top-k softmax router with load-balancing regularization."""

    config: AnthillRouterConfig

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.config.seed)
        scale = 1.0 / np.sqrt(self.config.input_dim)
        self.gate = rng.normal(0.0, scale, size=(self.config.input_dim, self.config.num_experts))

    def route(self, inputs: np.ndarray) -> RoutingDecision:
        batch = _as_batch(inputs, self.config.input_dim)
        logits = (batch @ self.gate) / self.config.temperature
        probabilities = _row_softmax(logits)
        selected = _top_k_indices(probabilities, self.config.top_k)
        masked = np.zeros_like(probabilities)
        rows = np.arange(probabilities.shape[0])[:, None]
        masked[rows, selected] = probabilities[rows, selected]
        masked = masked / np.clip(masked.sum(axis=1, keepdims=True), _EPS, None)
        load = masked.mean(axis=0)
        entropy = float(-np.mean(np.sum(masked * np.log(np.clip(masked, _EPS, 1.0)), axis=1)))
        return RoutingDecision(
            probabilities=masked,
            selected_experts=selected,
            load=load,
            load_balance_loss=PredictionEnergyLoss.load_balance_loss(load),
            entropy=entropy,
        )

    def local_update(self, inputs: np.ndarray, target_load: np.ndarray, learning_rate: float = 0.01) -> RoutingDecision:
        """Predictive coding update: gate ← gate - η·error·input^T (postsynaptic error × presynaptic activity)"""
        batch = _as_batch(inputs, self.config.input_dim)
        decision = self.route(batch)
        target = np.asarray(target_load, dtype=float).reshape(-1)
        if target.shape != (self.config.num_experts,):
            raise ValueError(f"target_load must have shape ({self.config.num_experts},)")
        target = target / np.clip(target.sum(), _EPS, None)
        error = decision.load - target
        # Predictive coding: Δgate = -η·error·input^T (postsynaptic error × presynaptic activity)
        mean_input = batch.mean(axis=0, keepdims=True)  # (1, input_dim)
        gate_delta = mean_input.T @ error[None, :]  # (input_dim, num_experts)
        self.gate -= learning_rate * gate_delta
        return self.route(batch)


@dataclass(frozen=True)
class LinearExpert:
    name: str
    weight: np.ndarray
    bias: np.ndarray

    @classmethod
    def random(cls, name: str, dim: int, rng: np.random.Generator) -> "LinearExpert":
        scale = 1.0 / np.sqrt(dim)
        return cls(
            name=name,
            weight=rng.normal(0.0, scale, size=(dim, dim)),
            bias=np.zeros(dim, dtype=float),
        )

    @classmethod
    def identity(cls, name: str, dim: int) -> "LinearExpert":
        return cls(
            name=name,
            weight=np.eye(dim, dtype=float),
            bias=np.zeros(dim, dtype=float),
        )

    @property
    def parameter_count(self) -> int:
        return int(self.weight.size + self.bias.size)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        batch = _as_batch(inputs, self.weight.shape[0])
        return batch @ self.weight + self.bias


@dataclass
class AnthillLayer:
    router: AnthillRouter
    experts: Sequence[LinearExpert]

    @classmethod
    def random(cls, config: AnthillRouterConfig) -> "AnthillLayer":
        rng = np.random.default_rng(config.seed)
        experts = [LinearExpert.random(f"expert_{idx}", config.input_dim, rng) for idx in range(config.num_experts)]
        return cls(router=AnthillRouter(config), experts=experts)

    @classmethod
    def identity(cls, config: AnthillRouterConfig) -> "AnthillLayer":
        experts = [LinearExpert.identity(f"expert_{idx}", config.input_dim) for idx in range(config.num_experts)]
        return cls(router=AnthillRouter(config), experts=experts)

    def __post_init__(self) -> None:
        if len(self.experts) != self.router.config.num_experts:
            raise ValueError("number of experts must match router config")

    @property
    def active_parameter_count(self) -> int:
        return int(sum(expert.parameter_count for expert in self.experts[: self.router.config.top_k]))

    @property
    def total_parameter_count(self) -> int:
        return int(sum(expert.parameter_count for expert in self.experts))

    def __call__(self, inputs: np.ndarray) -> tuple[np.ndarray, RoutingDecision]:
        batch = _as_batch(inputs, self.router.config.input_dim)
        decision = self.router.route(batch)
        outputs = np.zeros_like(batch)
        expert_outputs = [expert(batch) for expert in self.experts]
        for expert_idx, expert_output in enumerate(expert_outputs):
            outputs += decision.probabilities[:, [expert_idx]] * expert_output
        return outputs, decision


def _as_batch(inputs: np.ndarray, dim: int) -> np.ndarray:
    values = np.asarray(inputs, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != dim:
        raise ValueError(f"inputs must have shape (batch, {dim}), got {values.shape}")
    return values


def _row_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def _top_k_indices(probabilities: np.ndarray, top_k: int) -> np.ndarray:
    unsorted = np.argpartition(-probabilities, kth=top_k - 1, axis=1)[:, :top_k]
    row_indices = np.arange(probabilities.shape[0])[:, None]
    order = np.argsort(-probabilities[row_indices, unsorted], axis=1)
    return unsorted[row_indices, order]
