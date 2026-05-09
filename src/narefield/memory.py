from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class MemoryFieldConfig:
    dim: int
    capacity: int = 128
    learning_rate: float = 0.05
    inference_rate: float = 0.25
    inference_steps: int = 8
    decay: float = 0.995
    temperature: float = 1.0
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.inference_steps <= 0:
            raise ValueError("inference_steps must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


@dataclass(frozen=True)
class MemoryTrace:
    initial_state: np.ndarray
    state: np.ndarray
    prediction: np.ndarray
    error: np.ndarray
    delta: np.ndarray
    energy_history: tuple[float, ...]

    @property
    def energy(self) -> float:
        return self.energy_history[-1] if self.energy_history else 0.0


@dataclass
class MemoryField:
    """Attractor memory with predictive inference and Hebbian writes."""

    config: MemoryFieldConfig
    weights: np.ndarray = field(init=False)
    attractors: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.weights = np.zeros((self.config.dim, self.config.dim), dtype=float)
        self.attractors = np.empty((0, self.config.dim), dtype=float)

    def read(self, query: np.ndarray) -> np.ndarray:
        vector = self._as_vector(query)
        linear_prediction = self.weights @ vector
        if self.attractors.size == 0:
            return linear_prediction

        similarities = (self.attractors @ vector) / (
            np.sqrt(self.config.dim) * self.config.temperature
        )
        coefficients = _softmax(similarities)
        attractor_prediction = coefficients @ self.attractors
        return 0.5 * linear_prediction + 0.5 * attractor_prediction

    def infer(self, state: np.ndarray) -> MemoryTrace:
        initial = self._as_vector(state)
        current = initial.copy()
        history: list[float] = []
        prediction = self.read(current)
        error = current - prediction

        for _ in range(self.config.inference_steps):
            prediction = self.read(current)
            error = current - prediction
            history.append(float(np.mean(np.square(error))))
            current = current - self.config.inference_rate * error
            current = self._normalize(current)

        prediction = self.read(current)
        error = current - prediction
        history.append(float(np.mean(np.square(error))))
        return MemoryTrace(
            initial_state=initial,
            state=current,
            prediction=prediction,
            error=error,
            delta=current - initial,
            energy_history=tuple(history),
        )

    def learn(self, trace: MemoryTrace) -> None:
        z = self._normalize(np.asarray(trace.delta, dtype=float))
        error = self._normalize(np.asarray(trace.error, dtype=float))
        # Predictive coding: W ← decay·W + η·error·z^T (postsynaptic error × presynaptic activity)
        self.weights = (
            self.config.decay * self.weights
            + self.config.learning_rate * np.outer(error, z)
        )
        self._append_attractor(trace.delta)

    def write(self, state: np.ndarray) -> MemoryTrace:
        trace = self.infer(state)
        self.learn(trace)
        return trace

    def batch_write(self, states: np.ndarray) -> list[MemoryTrace]:
        values = np.asarray(states, dtype=float)
        if values.ndim == 1:
            return [self.write(values)]
        return [self.write(row) for row in values]

    def stats(self) -> dict[str, float]:
        return {
            "capacity": float(self.config.capacity),
            "stored_attractors": float(len(self.attractors)),
            "capacity_usage": float(len(self.attractors) / self.config.capacity),
            "weight_norm": float(np.linalg.norm(self.weights)),
        }

    def _append_attractor(self, state: np.ndarray) -> None:
        vector = self._normalize(self._as_vector(state))
        self.attractors = np.vstack([self.attractors, vector])
        if len(self.attractors) > self.config.capacity:
            self.attractors = self.attractors[-self.config.capacity :]

    def _as_vector(self, state: np.ndarray) -> np.ndarray:
        vector = np.asarray(state, dtype=float).reshape(-1)
        if vector.shape != (self.config.dim,):
            raise ValueError(
                f"state must have shape ({self.config.dim},), got {vector.shape}"
            )
        return self._normalize(vector)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        if not self.config.normalize:
            return np.asarray(vector, dtype=float)
        norm = np.linalg.norm(vector)
        if norm <= _EPS:
            return np.asarray(vector, dtype=float)
        return np.asarray(vector, dtype=float) / norm


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)
