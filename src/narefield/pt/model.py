"""Complete PyTorch NAREFieldModel — operational composition of all NARE-Field theory primitives."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .attention import FieldAttentionResult, SubQAttentionConfig, SubQFieldAttention
from .losses import CognitiveInvariant, LossBreakdown, PredictionEnergyLoss
from .memory import MemoryField, MemoryFieldConfig, MemoryTrace
from .routing import AnthillLayer, AnthillRouterConfig, RoutingDecision
from .temperature import CognitiveTemperature, CognitiveTemperatureConfig, TemperatureState
from .trace import (
    NonParametricMemory,
    NonParametricMemoryConfig,
    RetrievalResult,
    TraceMemory,
    TraceMemoryConfig,
    TraceReplay,
)


@dataclass(frozen=True)
class NAREConfig:
    dim: int
    memory_capacity: int = 128
    num_experts: int = 4
    top_k: int = 2
    seed: int | None = None
    invariant_weight: float = 0.01
    memory_weight: float = 0.01
    routing_weight: float = 0.01
    memory_learning_rate: float = 0.05
    memory_inference_rate: float = 0.25
    memory_inference_steps: int = 8
    attention_buckets: int = 8
    attention_top_buckets: int = 2
    attention_mix: float = 0.2
    anthill_mix: float = 0.35
    memory_mix: float = 0.25
    retrieval_mix: float = 0.2
    trace_mix: float = 0.2
    retrieval_capacity: int = 256
    retrieval_top_k: int = 3
    trace_capacity: int = 128
    trace_top_k: int = 2
    min_temperature: float = 0.2
    max_temperature: float = 3.0


@dataclass(frozen=True)
class NAREStepResult:
    prediction: torch.Tensor
    latent: torch.Tensor
    loss: LossBreakdown
    memory_trace: MemoryTrace
    routing: RoutingDecision
    attention: FieldAttentionResult
    retrieval: RetrievalResult
    trace_replay: TraceReplay
    temperature: TemperatureState
    metrics: dict[str, float | str]


class NAREFieldModel(nn.Module):
    """Operational composition of NARE-Field theory primitives (PyTorch)."""

    def __init__(self, config: NAREConfig):
        super().__init__()
        self.config = config

        self.memory = MemoryField(MemoryFieldConfig(
            dim=config.dim,
            capacity=config.memory_capacity,
            learning_rate=config.memory_learning_rate,
            inference_rate=config.memory_inference_rate,
            inference_steps=config.memory_inference_steps,
        ))
        self.anthill = AnthillLayer.identity(AnthillRouterConfig(
            input_dim=config.dim,
            num_experts=config.num_experts,
            top_k=config.top_k,
            seed=config.seed,
        ))
        self.attention = SubQFieldAttention(SubQAttentionConfig(
            dim=config.dim,
            num_buckets=config.attention_buckets,
            top_buckets=config.attention_top_buckets,
            seed=config.seed,
        ))
        self.retrieval_memory = NonParametricMemory(NonParametricMemoryConfig(
            dim=config.dim,
            capacity=config.retrieval_capacity,
            top_k=config.retrieval_top_k,
        ))
        self.trace_memory = TraceMemory(TraceMemoryConfig(
            dim=config.dim,
            capacity=config.trace_capacity,
            top_k=config.trace_top_k,
        ))
        self.temperature_ctrl = CognitiveTemperature(CognitiveTemperatureConfig(
            min_temperature=config.min_temperature,
            max_temperature=config.max_temperature,
        ))
        self.energy_loss = PredictionEnergyLoss(
            invariant_weight=config.invariant_weight,
            memory_weight=config.memory_weight,
            routing_weight=config.routing_weight,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        learn: bool = True,
    ) -> NAREStepResult:
        batch = self._as_batch(inputs)
        target_batch = batch if target is None else self._as_batch(target)

        # 1. SubQ field attention
        attention_result = self.attention(batch)
        attended = self._mix(batch, attention_result.output, self.config.attention_mix)

        # 2. Anthill MoE routing
        anthill_prediction, routing = self.anthill(attended)

        # 3. Non-parametric retrieval
        retrieval = self.retrieval_memory.retrieve(attended)

        # 4. Trace replay
        trace_replay = self.trace_memory.replay(attended)

        # 5. Memory field inference + optional learning
        memory_trace = self.memory(attended, learn=learn)
        memory_prediction = memory_trace.prediction

        # 6. Combine predictions
        prediction = self._combine_prediction(anthill_prediction, memory_prediction, retrieval, trace_replay)

        # 7. Cognitive invariant
        latent = memory_trace.state
        invariant = CognitiveInvariant.from_state(latent)

        # 8. Loss
        memory_stats = self.memory.stats()
        loss = self.energy_loss(
            target_batch,
            prediction,
            invariant=invariant,
            memory_stats=memory_stats,
            routing_loss=routing.load_balance_loss,
        )

        # 9. Write to external memories
        if learn:
            self.retrieval_memory.write(latent.detach(), target_batch.detach())
            self.trace_memory.record(memory_trace)

        # 10. Update cognitive temperature
        temperature_state = self.temperature_ctrl.update(loss.total.item(), latent)

        # Metrics
        metrics = self._compute_metrics(loss, routing, memory_stats, memory_trace, attention_result,
                                         retrieval, trace_replay, temperature_state)

        return NAREStepResult(
            prediction=prediction,
            latent=latent,
            loss=loss,
            memory_trace=memory_trace,
            routing=routing,
            attention=attention_result,
            retrieval=retrieval,
            trace_replay=trace_replay,
            temperature=temperature_state,
            metrics=metrics,
        )

    def _combine_prediction(
        self,
        anthill_prediction: torch.Tensor,
        memory_prediction: torch.Tensor,
        retrieval: RetrievalResult,
        trace_replay: TraceReplay,
    ) -> torch.Tensor:
        components: list[tuple[float, torch.Tensor]] = [
            (self.config.anthill_mix, anthill_prediction),
            (self.config.memory_mix, memory_prediction),
        ]
        if retrieval.hit_rate > 0.0:
            components.append((self.config.retrieval_mix, retrieval.values))
        if trace_replay.hit_rate > 0.0:
            components.append((self.config.trace_mix, trace_replay.reconstruction))
        total_weight = sum(w for w, _ in components)
        result = sum(w * v for w, v in components) / max(total_weight, 1e-12)
        return result

    def _compute_metrics(
        self,
        loss: LossBreakdown,
        routing: RoutingDecision,
        memory_stats: dict[str, float],
        memory_trace: MemoryTrace,
        attention: FieldAttentionResult,
        retrieval: RetrievalResult,
        trace_replay: TraceReplay,
        temperature: TemperatureState,
    ) -> dict[str, float | str]:
        active_ratio = self.anthill.active_parameter_count / max(self.anthill.total_parameter_count, 1)
        return {
            "loss_total": loss.total.item(),
            "loss_reconstruction": loss.reconstruction.item(),
            "loss_routing_balance": routing.load_balance_loss.item(),
            "routing_entropy": routing.entropy.item(),
            "routing_load_std": float(routing.load.std().item()),
            "memory_capacity_usage": memory_stats["capacity_usage"],
            "memory_weight_norm": memory_stats["weight_norm"],
            "memory_trace_energy": memory_trace.energy,
            "retrieval_hit_rate": retrieval.hit_rate,
            "trace_hit_rate": trace_replay.hit_rate,
            "trace_replay_energy": trace_replay.mean_energy,
            "attention_active_tokens": float(attention.active_tokens),
            "attention_complexity_estimate": attention.complexity_estimate,
            "cognitive_temperature": temperature.temperature,
            "cognitive_mode": temperature.mode,
            "active_parameter_ratio": float(active_ratio),
        }

    def _as_batch(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[1] != self.config.dim:
            raise ValueError(f"values must have shape (batch, {self.config.dim}), got {values.shape}")
        return values

    @staticmethod
    def _mix(left: torch.Tensor, right: torch.Tensor, weight: float) -> torch.Tensor:
        w = max(0.0, min(1.0, weight))
        return (1.0 - w) * left + w * right
