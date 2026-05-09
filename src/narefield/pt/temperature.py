"""Cognitive temperature controller for explore/exploit switching (PyTorch)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

_EPS = 1e-12


@dataclass(frozen=True)
class CognitiveTemperatureConfig:
    min_temperature: float = 0.2
    max_temperature: float = 3.0
    energy_scale: float = 0.5
    entropy_scale: float = 0.5
    smoothing: float = 0.8


@dataclass(frozen=True)
class TemperatureState:
    temperature: float
    entropy: float
    energy: float
    mode: str


def state_entropy(state: torch.Tensor) -> float:
    values = torch.abs(state).reshape(-1)
    total = values.sum().item()
    if total <= _EPS:
        return 0.0
    probabilities = (values / total).clamp(min=_EPS)
    import math
    return float((-probabilities * torch.log(probabilities)).sum().item() / math.log(probabilities.numel() + 1.0))


class CognitiveTemperature(nn.Module):
    """Exploration/exploitation controller from energy and entropy."""

    def __init__(self, config: CognitiveTemperatureConfig | None = None):
        super().__init__()
        self.config = config or CognitiveTemperatureConfig()
        self._temperature = self.config.min_temperature

    def update(self, energy: float, state: torch.Tensor) -> TemperatureState:
        import math
        entropy = state_entropy(state)
        raw = self.config.min_temperature + self.config.energy_scale * math.log1p(max(energy, 0.0))
        raw += self.config.entropy_scale * entropy
        clipped = max(self.config.min_temperature, min(self.config.max_temperature, raw))
        self._temperature = self.config.smoothing * self._temperature + (1.0 - self.config.smoothing) * clipped
        midpoint = 0.5 * (self.config.min_temperature + self.config.max_temperature)
        mode = "explore" if self._temperature >= midpoint else "exploit"
        return TemperatureState(
            temperature=self._temperature,
            entropy=entropy,
            energy=energy,
            mode=mode,
        )

    def reset(self) -> None:
        self._temperature = self.config.min_temperature
