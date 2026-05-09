"""PyTorch implementations of NARE-Field components."""

from .attention import FieldAttentionResult, SubQAttentionConfig, SubQFieldAttention
from .losses import CognitiveInvariant, LossBreakdown, PredictionEnergyLoss
from .memory import MemoryField, MemoryFieldConfig, MemoryTrace
from .model import NAREConfig, NAREFieldModel, NAREStepResult
from .routing import AnthillLayer, AnthillRouter, AnthillRouterConfig, LinearExpert, RoutingDecision
from .temperature import CognitiveTemperature, CognitiveTemperatureConfig, TemperatureState, state_entropy
from .trace import (
    NonParametricMemory,
    NonParametricMemoryConfig,
    RetrievalResult,
    TraceMemory,
    TraceMemoryConfig,
    TraceReplay,
)
from .trainer import FreeEnergyTrainer, FreeEnergyUpdate

__all__ = [
    "AnthillLayer",
    "AnthillRouter",
    "AnthillRouterConfig",
    "CognitiveInvariant",
    "CognitiveTemperature",
    "CognitiveTemperatureConfig",
    "FieldAttentionResult",
    "FreeEnergyTrainer",
    "FreeEnergyUpdate",
    "LinearExpert",
    "LossBreakdown",
    "MemoryField",
    "MemoryFieldConfig",
    "MemoryTrace",
    "NAREConfig",
    "NAREFieldModel",
    "NAREStepResult",
    "NonParametricMemory",
    "NonParametricMemoryConfig",
    "PredictionEnergyLoss",
    "RetrievalResult",
    "RoutingDecision",
    "SubQAttentionConfig",
    "SubQFieldAttention",
    "TemperatureState",
    "TraceMemory",
    "TraceMemoryConfig",
    "TraceReplay",
    "state_entropy",
]
