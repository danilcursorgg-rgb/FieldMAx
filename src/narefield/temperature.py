from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class CognitiveTemperatureConfig:
    min_temperature: float = 0.2
    max_temperature: float = 3.0
    energy_scale: float = 0.5
    entropy_scale: float = 0.5
    smoothing: float = 0.8

    def __post_init__(self) -> None:
        if self.min_temperature <= 0:
            raise ValueError("min_temperature must be positive")
        if self.max_temperature < self.min_temperature:
            raise ValueError("max_temperature must be >= min_temperature")
        if not 0.0 <= self.smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")


@dataclass(frozen=True)
class TemperatureState:
    temperature: float
    entropy: float
    energy: float
    mode: str


@dataclass
class CognitiveTemperature:
    """Exploration/exploitation controller derived from energy and entropy."""

    config: CognitiveTemperatureConfig = CognitiveTemperatureConfig()

    def __post_init__(self) -> None:
        self._temperature = self.config.min_temperature

    def update(self, energy: float, state: np.ndarray) -> TemperatureState:
        entropy = state_entropy(state)
        raw = self.config.min_temperature + self.config.energy_scale * np.log1p(max(energy, 0.0))
        raw += self.config.entropy_scale * entropy
        clipped = float(np.clip(raw, self.config.min_temperature, self.config.max_temperature))
        self._temperature = self.config.smoothing * self._temperature + (1.0 - self.config.smoothing) * clipped
        midpoint = 0.5 * (self.config.min_temperature + self.config.max_temperature)
        mode = "explore" if self._temperature >= midpoint else "exploit"
        return TemperatureState(
            temperature=float(self._temperature),
            entropy=float(entropy),
            energy=float(energy),
            mode=mode,
        )

    @property
    def value(self) -> float:
        return float(self._temperature)


def state_entropy(state: np.ndarray) -> float:
    values = np.abs(np.asarray(state, dtype=float)).reshape(-1)
    total = float(np.sum(values))
    if total <= _EPS:
        return 0.0
    probabilities = np.clip(values / total, _EPS, 1.0)
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(probabilities.size + 1.0))
