"""PyTorch trace memory and non-parametric retrieval for NARE-Field."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import MemoryTrace

_EPS = 1e-12


@dataclass(frozen=True)
class TraceMemoryConfig:
    dim: int
    capacity: int = 128
    top_k: int = 2
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")


@dataclass(frozen=True)
class TraceReplay:
    reconstruction: torch.Tensor
    selected_indices: torch.Tensor
    weights: torch.Tensor
    mean_energy: float
    hit_rate: float


class TraceMemory(nn.Module):
    """Stores latent reasoning trajectories (PyTorch)."""

    def __init__(self, config: TraceMemoryConfig):
        super().__init__()
        self.config = config
        self.register_buffer("keys", torch.empty(0, config.dim))
        self.register_buffer("deltas", torch.empty(0, config.dim))
        self._energies: list[float] = []

    def record(self, trace: MemoryTrace) -> None:
        with torch.no_grad():
            state = trace.state.detach()
            delta = trace.delta.detach()
            norm = torch.norm(state, dim=-1, keepdim=True).clamp(min=_EPS)
            key = state / norm
            if key.ndim == 1:
                key = key.unsqueeze(0)
                delta = delta.unsqueeze(0)
            self.keys = torch.cat([self.keys, key], dim=0)
            self.deltas = torch.cat([self.deltas, delta], dim=0)
            self._energies.extend([trace.energy] * key.shape[0])
            if self.keys.shape[0] > self.config.capacity:
                excess = self.keys.shape[0] - self.config.capacity
                self.keys = self.keys[excess:]
                self.deltas = self.deltas[excess:]
                self._energies = self._energies[excess:]

    def replay(self, queries: torch.Tensor) -> TraceReplay:
        if queries.ndim == 1:
            queries = queries.unsqueeze(0)
        batch_size = queries.shape[0]
        if self.keys.shape[0] == 0:
            return TraceReplay(
                reconstruction=torch.zeros_like(queries),
                selected_indices=torch.zeros(batch_size, self.config.top_k, dtype=torch.long, device=queries.device),
                weights=torch.zeros(batch_size, self.config.top_k, device=queries.device),
                mean_energy=0.0,
                hit_rate=0.0,
            )
        q_norm = F.normalize(queries, dim=-1)
        k_norm = F.normalize(self.keys, dim=-1)
        similarities = torch.matmul(q_norm, k_norm.T)
        count = min(self.config.top_k, self.keys.shape[0])
        topk_vals, topk_idx = torch.topk(similarities, count, dim=-1)
        weights = F.softmax(topk_vals / self.config.temperature, dim=-1)
        selected_deltas = self.deltas[topk_idx]
        replay_delta = torch.einsum("bk,bkd->bd", weights, selected_deltas)
        energies_tensor = torch.tensor(self._energies, device=queries.device)
        selected_energies = energies_tensor[topk_idx]
        max_sim = similarities.max(dim=-1).values
        hit_rate = float((max_sim > 0.0).float().mean().item())
        return TraceReplay(
            reconstruction=queries + replay_delta,
            selected_indices=topk_idx,
            weights=weights,
            mean_energy=float(selected_energies.mean().item()),
            hit_rate=hit_rate,
        )

    def stats(self) -> dict[str, float]:
        return {
            "capacity": float(self.config.capacity),
            "stored_traces": float(self.keys.shape[0]),
            "capacity_usage": float(self.keys.shape[0] / self.config.capacity),
        }


@dataclass(frozen=True)
class NonParametricMemoryConfig:
    dim: int
    capacity: int = 256
    top_k: int = 3
    temperature: float = 1.0


@dataclass(frozen=True)
class RetrievalResult:
    values: torch.Tensor
    selected_indices: torch.Tensor
    weights: torch.Tensor
    hit_rate: float


class NonParametricMemory(nn.Module):
    """External key-value memory for context-over-parameters (PyTorch)."""

    def __init__(self, config: NonParametricMemoryConfig):
        super().__init__()
        self.config = config
        self.register_buffer("keys", torch.empty(0, config.dim))
        self.register_buffer("values", torch.empty(0, config.dim))

    def write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        with torch.no_grad():
            k = keys.detach()
            v = values.detach()
            if k.ndim == 1:
                k = k.unsqueeze(0)
                v = v.unsqueeze(0)
            self.keys = torch.cat([self.keys, k], dim=0)
            self.values = torch.cat([self.values, v], dim=0)
            if self.keys.shape[0] > self.config.capacity:
                excess = self.keys.shape[0] - self.config.capacity
                self.keys = self.keys[excess:]
                self.values = self.values[excess:]

    def retrieve(self, queries: torch.Tensor) -> RetrievalResult:
        if queries.ndim == 1:
            queries = queries.unsqueeze(0)
        batch_size = queries.shape[0]
        if self.keys.shape[0] == 0:
            return RetrievalResult(
                values=torch.zeros_like(queries),
                selected_indices=torch.zeros(batch_size, self.config.top_k, dtype=torch.long, device=queries.device),
                weights=torch.zeros(batch_size, self.config.top_k, device=queries.device),
                hit_rate=0.0,
            )
        q_norm = F.normalize(queries, dim=-1)
        k_norm = F.normalize(self.keys, dim=-1)
        similarities = torch.matmul(q_norm, k_norm.T)
        count = min(self.config.top_k, self.keys.shape[0])
        topk_vals, topk_idx = torch.topk(similarities, count, dim=-1)
        weights = F.softmax(topk_vals / self.config.temperature, dim=-1)
        selected_values = self.values[topk_idx]
        result = torch.einsum("bk,bkd->bd", weights, selected_values)
        max_sim = similarities.max(dim=-1).values
        hit_rate = float((max_sim > 0.0).float().mean().item())
        return RetrievalResult(
            values=result,
            selected_indices=topk_idx,
            weights=weights,
            hit_rate=hit_rate,
        )

    def stats(self) -> dict[str, float]:
        return {
            "capacity": float(self.config.capacity),
            "stored_entries": float(self.keys.shape[0]),
            "capacity_usage": float(self.keys.shape[0] / self.config.capacity),
        }
