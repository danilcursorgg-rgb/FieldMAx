from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class ValidationReport:
    mean_loss: float
    mean_reconstruction: float
    mean_routing_balance: float
    routing_load_std: float
    active_experts: int
    memory_capacity_usage: float
    active_parameter_ratio: float
    mean_attention_complexity: float
    mean_retrieval_hit_rate: float
    mean_trace_hit_rate: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mean_loss": self.mean_loss,
            "mean_reconstruction": self.mean_reconstruction,
            "mean_routing_balance": self.mean_routing_balance,
            "routing_load_std": self.routing_load_std,
            "active_experts": self.active_experts,
            "memory_capacity_usage": self.memory_capacity_usage,
            "active_parameter_ratio": self.active_parameter_ratio,
            "mean_attention_complexity": self.mean_attention_complexity,
            "mean_retrieval_hit_rate": self.mean_retrieval_hit_rate,
            "mean_trace_hit_rate": self.mean_trace_hit_rate,
        }


@dataclass(frozen=True)
class AblationReport:
    baseline: str
    variant: str
    loss_delta: float
    reconstruction_delta: float
    active_parameter_delta: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "baseline": self.baseline,
            "variant": self.variant,
            "loss_delta": self.loss_delta,
            "reconstruction_delta": self.reconstruction_delta,
            "active_parameter_delta": self.active_parameter_delta,
        }


def summarize_validation(records: Iterable[Mapping[str, float | np.ndarray]]) -> ValidationReport:
    rows = list(records)
    if not rows:
        raise ValueError("records must not be empty")

    losses = np.array([float(row["loss_total"]) for row in rows], dtype=float)
    reconstruction = np.array([float(row["loss_reconstruction"]) for row in rows], dtype=float)
    routing_balance = np.array([float(row["loss_routing_balance"]) for row in rows], dtype=float)
    loads = np.vstack([np.asarray(row["routing_load"], dtype=float) for row in rows])
    capacity_usage = np.array([float(row["memory_capacity_usage"]) for row in rows], dtype=float)
    active_ratio = np.array([float(row["active_parameter_ratio"]) for row in rows], dtype=float)
    attention_complexity = _optional_array(rows, "attention_complexity")
    retrieval_hit = _optional_array(rows, "retrieval_hit_rate")
    trace_hit = _optional_array(rows, "trace_hit_rate")
    mean_load = loads.mean(axis=0)

    return ValidationReport(
        mean_loss=float(losses.mean()),
        mean_reconstruction=float(reconstruction.mean()),
        mean_routing_balance=float(routing_balance.mean()),
        routing_load_std=float(np.std(mean_load)),
        active_experts=int(np.sum(mean_load > 0.0)),
        memory_capacity_usage=float(capacity_usage.mean()),
        active_parameter_ratio=float(active_ratio.mean()),
        mean_attention_complexity=float(attention_complexity.mean()),
        mean_retrieval_hit_rate=float(retrieval_hit.mean()),
        mean_trace_hit_rate=float(trace_hit.mean()),
    )


def compare_ablation(baseline: ValidationReport, variant: ValidationReport, variant_name: str) -> AblationReport:
    return AblationReport(
        baseline="baseline",
        variant=variant_name,
        loss_delta=float(variant.mean_loss - baseline.mean_loss),
        reconstruction_delta=float(variant.mean_reconstruction - baseline.mean_reconstruction),
        active_parameter_delta=float(variant.active_parameter_ratio - baseline.active_parameter_ratio),
    )


def _optional_array(rows: list[Mapping[str, float | np.ndarray]], key: str) -> np.ndarray:
    return np.array([float(row.get(key, 0.0)) for row in rows], dtype=float)
