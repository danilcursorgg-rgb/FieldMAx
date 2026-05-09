from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

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
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


@dataclass(frozen=True)
class LatentTrace:
    states: np.ndarray
    delta: np.ndarray
    energy: float

    @property
    def key(self) -> np.ndarray:
        return self.states[0]


@dataclass(frozen=True)
class TraceReplay:
    reconstruction: np.ndarray
    selected_indices: np.ndarray
    weights: np.ndarray
    mean_energy: float
    hit_rate: float


@dataclass
class TraceMemory:
    """Stores latent reasoning trajectories instead of textual chain-of-thought."""

    config: TraceMemoryConfig
    traces: list[LatentTrace] = field(default_factory=list)
    _session_traces: list[LatentTrace] = field(default_factory=list)

    def reset_context(self) -> None:
        self._session_traces.clear()

    def record(self, memory_traces: tuple[MemoryTrace, ...] | list[MemoryTrace]) -> None:
        for trace in memory_traces:
            state = trace.state
            norm = np.linalg.norm(state)
            key = state / max(norm, _EPS)
            t = LatentTrace(states=key.reshape(1, -1), delta=trace.delta, energy=trace.energy)
            self.traces.append(t)
            self._session_traces.append(t)
            
        if len(self.traces) > self.config.capacity:
            self.traces = self.traces[-self.config.capacity :]
            
        if len(self._session_traces) > self.config.capacity:
            self._session_traces = self._session_traces[-self.config.capacity :]

    def replay(self, queries: np.ndarray) -> TraceReplay:
        batch = _as_batch(queries, self.config.dim)
        if not self.traces and not self._session_traces:
            return TraceReplay(
                reconstruction=np.zeros_like(batch),
                selected_indices=np.zeros((batch.shape[0], self.config.top_k), dtype=int),
                weights=np.zeros((batch.shape[0], self.config.top_k), dtype=float),
                mean_energy=0.0,
                hit_rate=0.0,
            )

        all_traces = self.traces + self._session_traces
        keys = np.vstack([trace.states for trace in all_traces])
        deltas = np.vstack([trace.delta for trace in all_traces])
        energies = np.array([trace.energy for trace in all_traces], dtype=float)
        similarities = _cosine(batch, keys)
        count = min(self.config.top_k, len(all_traces))
        selected = _top_k(similarities, count)
        row = np.arange(batch.shape[0])[:, None]
        selected_scores = similarities[row, selected] / self.config.temperature
        weights = _row_softmax(selected_scores)
        replay_delta = np.einsum("bk,bkd->bd", weights, deltas[selected])
        selected_energy = energies[selected]
        return TraceReplay(
            reconstruction=batch + replay_delta,
            selected_indices=selected,
            weights=weights,
            mean_energy=float(np.mean(selected_energy)),
            hit_rate=float(np.mean(np.max(similarities, axis=1) > 0.0)),
        )

    def stats(self) -> dict[str, float]:
        return {
            "capacity": float(self.config.capacity),
            "stored_traces": float(len(self.traces)),
            "capacity_usage": float(len(self.traces) / self.config.capacity),
        }


@dataclass(frozen=True)
class NonParametricMemoryConfig:
    dim: int
    capacity: int = 256
    top_k: int = 3
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


@dataclass(frozen=True)
class RetrievalResult:
    values: np.ndarray
    selected_indices: np.ndarray
    weights: np.ndarray
    hit_rate: float


@dataclass
class NonParametricMemory:
    """External key-value memory for context-over-parameters reconstruction."""

    config: NonParametricMemoryConfig

    def __post_init__(self) -> None:
        self.keys = np.empty((0, self.config.dim), dtype=float)
        self.values = np.empty((0, self.config.dim), dtype=float)

    def write(self, keys: np.ndarray, values: np.ndarray) -> None:
        key_batch = _as_batch(keys, self.config.dim)
        value_batch = _as_batch(values, self.config.dim)
        if key_batch.shape != value_batch.shape:
            raise ValueError("keys and values must have matching shapes")
        self.keys = np.vstack([self.keys, key_batch])
        self.values = np.vstack([self.values, value_batch])
        if len(self.keys) > self.config.capacity:
            self.keys = self.keys[-self.config.capacity :]
            self.values = self.values[-self.config.capacity :]

    def retrieve(self, queries: np.ndarray) -> RetrievalResult:
        batch = _as_batch(queries, self.config.dim)
        if len(self.keys) == 0:
            return RetrievalResult(
                values=np.zeros_like(batch),
                selected_indices=np.empty((batch.shape[0], 0), dtype=int),
                weights=np.empty((batch.shape[0], 0), dtype=float),
                hit_rate=0.0,
            )
        similarities = _cosine(batch, self.keys)
        count = min(self.config.top_k, len(self.keys))
        selected = _top_k(similarities, count)
        row = np.arange(batch.shape[0])[:, None]
        weights = _row_softmax(similarities[row, selected] / self.config.temperature)
        values = np.einsum("bk,bkd->bd", weights, self.values[selected])
        return RetrievalResult(
            values=values,
            selected_indices=selected,
            weights=weights,
            hit_rate=float(np.mean(np.max(similarities, axis=1) > 0.0)),
        )

    def stats(self) -> dict[str, float]:
        return {
            "capacity": float(self.config.capacity),
            "stored_items": float(len(self.keys)),
            "capacity_usage": float(len(self.keys) / self.config.capacity),
        }


def _as_batch(values: np.ndarray, dim: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != dim:
        raise ValueError(f"values must have shape (batch, {dim}), got {array.shape}")
    return array


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = left / np.clip(np.linalg.norm(left, axis=1, keepdims=True), _EPS, None)
    right_norm = right / np.clip(np.linalg.norm(right, axis=1, keepdims=True), _EPS, None)
    return left_norm @ right_norm.T


def _top_k(scores: np.ndarray, count: int) -> np.ndarray:
    unsorted = np.argpartition(-scores, kth=count - 1, axis=1)[:, :count]
    row = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[row, unsorted], axis=1)
    return unsorted[row, order]


def _row_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)
