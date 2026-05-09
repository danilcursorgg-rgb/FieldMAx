from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attention import FieldAttentionResult, SubQAttentionConfig, SubQFieldAttention
from .losses import CognitiveInvariant, LossBreakdown, PredictionEnergyLoss
from .memory import MemoryField, MemoryFieldConfig, MemoryTrace
from .pipeline import NAREPipelineTrace, build_pipeline_trace
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
    prediction: np.ndarray
    latent: np.ndarray
    loss: LossBreakdown
    memory_traces: tuple[MemoryTrace, ...]
    routing: RoutingDecision
    attention: FieldAttentionResult
    retrieval: RetrievalResult
    trace_replay: TraceReplay
    temperature: TemperatureState
    pipeline_trace: NAREPipelineTrace
    metrics: dict[str, float | np.ndarray | str]


@dataclass
class NAREFieldModel:
    """Operational composition of NARE-Field theory primitives."""

    config: NAREConfig

    def __post_init__(self) -> None:
        self.memory = MemoryField(
            MemoryFieldConfig(
                dim=self.config.dim,
                capacity=self.config.memory_capacity,
                learning_rate=self.config.memory_learning_rate,
                inference_rate=self.config.memory_inference_rate,
                inference_steps=self.config.memory_inference_steps,
            )
        )
        self.anthill = AnthillLayer.identity(
            AnthillRouterConfig(
                input_dim=self.config.dim,
                num_experts=self.config.num_experts,
                top_k=self.config.top_k,
                seed=self.config.seed,
            )
        )
        self.attention = SubQFieldAttention(
            SubQAttentionConfig(
                dim=self.config.dim,
                num_buckets=self.config.attention_buckets,
                top_buckets=self.config.attention_top_buckets,
                seed=self.config.seed,
            )
        )
        self.retrieval_memory = NonParametricMemory(
            NonParametricMemoryConfig(
                dim=self.config.dim,
                capacity=self.config.retrieval_capacity,
                top_k=self.config.retrieval_top_k,
            )
        )
        self.trace_memory = TraceMemory(
            TraceMemoryConfig(dim=self.config.dim, capacity=self.config.trace_capacity, top_k=self.config.trace_top_k)
        )
        self.temperature = CognitiveTemperature(
            CognitiveTemperatureConfig(
                min_temperature=self.config.min_temperature,
                max_temperature=self.config.max_temperature,
            )
        )
        self.energy_loss = PredictionEnergyLoss(
            invariant_weight=self.config.invariant_weight,
            memory_weight=self.config.memory_weight,
            routing_weight=self.config.routing_weight,
        )

    def step(self, inputs: np.ndarray, target: np.ndarray | None = None, *, previous_inputs: np.ndarray | None = None, learn: bool = True) -> NAREStepResult:
        batch = self._as_batch(inputs)

        # Compute delta-based logic
        delta = np.zeros_like(batch)
        if batch.shape[0] > 1:
            delta[1:] = batch[1:] - batch[:-1]
            if previous_inputs is not None:
                prev = self._as_batch(previous_inputs)
                delta[0] = batch[0] - prev[-1]
            else:
                delta[0] = batch[0]
        else:
            if previous_inputs is not None:
                prev = self._as_batch(previous_inputs)
                delta[0] = batch[0] - prev[-1]
            else:
                delta[0] = batch[0]

        target_batch = delta if target is None else self._as_batch(target)
        if target_batch.shape != delta.shape:
            raise ValueError("target must have the same shape as inputs")

        attention_result = self.attention(delta)
        attended = self._mix(delta, attention_result.output, self.config.attention_mix)
        anthill_prediction, routing = self.anthill(attended)
        retrieval = self.retrieval_memory.retrieve(attended)
        trace_replay = self.trace_memory.replay(attended)
        memory_traces = tuple(self.memory.write(row) if learn else self.memory.infer(row) for row in attended)
        memory_prediction = np.vstack([trace.prediction for trace in memory_traces])
        prediction = self._combine_prediction(anthill_prediction, memory_prediction, retrieval, trace_replay)
        latent = np.vstack([trace.state for trace in memory_traces])
        invariant = CognitiveInvariant.from_state(latent)
        memory_stats = self.memory.stats()
        loss = self.energy_loss(
            target_batch,
            prediction,
            invariant=invariant,
            memory_stats=memory_stats,
            routing_load=routing.load,
        )
        if learn:
            self.retrieval_memory.write(latent, target_batch)
            self.trace_memory.record(memory_traces)
        temperature_state = self.temperature.update(loss.total, latent)
        metrics = self._metrics(
            loss,
            routing,
            memory_stats,
            memory_traces,
            attention_result,
            retrieval,
            trace_replay,
            temperature_state,
        )
        pipeline_trace = build_pipeline_trace(
            reconstruction=loss.reconstruction,
            invariant=loss.invariant,
            memory=loss.memory,
            routing_balance=loss.routing_balance,
            invariant_weight=self.config.invariant_weight,
            memory_weight=self.config.memory_weight,
            routing_weight=self.config.routing_weight,
            total_energy=loss.total,
            attention_complexity=attention_result.complexity_estimate,
            routing_load_std=float(metrics["routing_load_std"]),
            memory_energy=float(metrics["mean_memory_trace_energy"]),
            retrieval_hit_rate=retrieval.hit_rate,
            trace_hit_rate=trace_replay.hit_rate,
            trace_energy=trace_replay.mean_energy,
            cognitive_temperature=temperature_state.temperature,
            cognitive_mode=temperature_state.mode,
        )
        return NAREStepResult(
            prediction=prediction,
            latent=latent,
            loss=loss,
            memory_traces=memory_traces,
            routing=routing,
            attention=attention_result,
            retrieval=retrieval,
            trace_replay=trace_replay,
            temperature=temperature_state,
            pipeline_trace=pipeline_trace,
            metrics=metrics,
        )

    def reset(self) -> None:
        self.temperature.reset()

    def validation_record(self, result: NAREStepResult) -> dict[str, float | np.ndarray]:
        return {
            "loss_total": result.loss.total,
            "loss_reconstruction": result.loss.reconstruction,
            "loss_routing_balance": result.loss.routing_balance,
            "routing_load": result.routing.load,
            "memory_capacity_usage": result.metrics["memory_capacity_usage"],
            "active_parameter_ratio": result.metrics["active_parameter_ratio"],
            "attention_complexity": result.metrics["attention_complexity_estimate"],
            "retrieval_hit_rate": result.metrics["retrieval_hit_rate"],
            "trace_hit_rate": result.metrics["trace_hit_rate"],
        }

    def _combine_prediction(
        self,
        anthill_prediction: np.ndarray,
        memory_prediction: np.ndarray,
        retrieval: RetrievalResult,
        trace_replay: TraceReplay,
    ) -> np.ndarray:
        components: list[tuple[float, np.ndarray]] = [
            (self.config.anthill_mix, anthill_prediction),
            (self.config.memory_mix, memory_prediction),
        ]
        if retrieval.hit_rate > 0.0:
            components.append((self.config.retrieval_mix, retrieval.values))
        if trace_replay.hit_rate > 0.0:
            components.append((self.config.trace_mix, trace_replay.reconstruction))
        total = sum(weight for weight, _ in components)
        return sum(weight * value for weight, value in components) / max(total, 1e-12)

    def _metrics(
        self,
        loss: LossBreakdown,
        routing: RoutingDecision,
        memory_stats: dict[str, float],
        memory_traces: tuple[MemoryTrace, ...],
        attention: FieldAttentionResult,
        retrieval: RetrievalResult,
        trace_replay: TraceReplay,
        temperature: TemperatureState,
    ) -> dict[str, float | np.ndarray | str]:
        active_ratio = self.anthill.active_parameter_count / max(self.anthill.total_parameter_count, 1)
        retrieval_stats = self.retrieval_memory.stats()
        trace_stats = self.trace_memory.stats()
        mean_memory_energy = float(np.mean([trace.energy for trace in memory_traces])) if memory_traces else 0.0
        return {
            **loss.as_dict(),
            "routing_entropy": routing.entropy,
            "routing_load_std": float(np.std(routing.load)),
            "routing_load": routing.load,
            "memory_capacity_usage": memory_stats["capacity_usage"],
            "memory_weight_norm": memory_stats["weight_norm"],
            "mean_memory_trace_energy": mean_memory_energy,
            "retrieval_capacity_usage": retrieval_stats["capacity_usage"],
            "trace_capacity_usage": trace_stats["capacity_usage"],
            "retrieval_hit_rate": retrieval.hit_rate,
            "trace_hit_rate": trace_replay.hit_rate,
            "trace_replay_energy": trace_replay.mean_energy,
            "attention_active_tokens": float(attention.active_tokens),
            "attention_complexity_estimate": attention.complexity_estimate,
            "cognitive_temperature": temperature.temperature,
            "cognitive_entropy": temperature.entropy,
            "cognitive_mode": temperature.mode,
            "active_parameter_ratio": float(active_ratio),
        }

    def _as_batch(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != self.config.dim:
            raise ValueError(f"values must have shape (batch, {self.config.dim}), got {array.shape}")
        return array

    @staticmethod
    def _mix(left: np.ndarray, right: np.ndarray, weight: float) -> np.ndarray:
        clipped = float(np.clip(weight, 0.0, 1.0))
        return (1.0 - clipped) * left + clipped * right
