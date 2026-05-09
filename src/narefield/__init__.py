"""Operational NARE-Field components."""

from .architecture import ArchitectureStage, NAREArchitectureSpec, NARE_FIELD_ARCHITECTURE, theory_alignment_table
from .attention import FieldAttentionResult, SubQAttentionConfig, SubQFieldAttention
from .benchmark import (
    AblationMatrixReport,
    AblationVariant,
    BenchmarkRunConfig,
    BenchmarkRunReport,
    BenchmarkRunner,
    BenchmarkSuite,
    BenchmarkTask,
    DEFAULT_ABLATIONS,
    TaskEncoder,
)
from .evidence import EvidenceRunner, EvidenceSeries, HypothesisEvidence, NAREEvidenceReport
from .losses import CognitiveInvariant, LossBreakdown, PredictionEnergyLoss
from .memory import MemoryField, MemoryFieldConfig, MemoryTrace
from .metrics import AblationReport, ValidationReport, compare_ablation, summarize_validation
from .model import NAREConfig, NAREFieldModel, NAREStepResult
from .pipeline import EnergyLedgerTerm, NAREPipelineTrace, PipelineStageTrace, build_pipeline_trace
from .routing import AnthillLayer, AnthillRouter, AnthillRouterConfig, LinearExpert, RoutingDecision
from .temperature import CognitiveTemperature, CognitiveTemperatureConfig, TemperatureState, state_entropy
from .trace import (
    LatentTrace,
    NonParametricMemory,
    NonParametricMemoryConfig,
    RetrievalResult,
    TraceMemory,
    TraceMemoryConfig,
    TraceReplay,
)
from .trainer import FreeEnergyTrainer, FreeEnergyUpdate

__all__ = [
    "AblationMatrixReport",
    "AblationReport",
    "AblationVariant",
    "ArchitectureStage",
    "AnthillLayer",
    "AnthillRouter",
    "AnthillRouterConfig",
    "CognitiveInvariant",
    "CognitiveTemperature",
    "CognitiveTemperatureConfig",
    "DEFAULT_ABLATIONS",
    "EnergyLedgerTerm",
    "EvidenceRunner",
    "EvidenceSeries",
    "FieldAttentionResult",
    "FreeEnergyTrainer",
    "FreeEnergyUpdate",
    "HypothesisEvidence",
    "LatentTrace",
    "LinearExpert",
    "LossBreakdown",
    "MemoryField",
    "MemoryFieldConfig",
    "MemoryTrace",
    "NAREArchitectureSpec",
    "NAREConfig",
    "NAREFieldModel",
    "NARE_FIELD_ARCHITECTURE",
    "NAREEvidenceReport",
    "NAREPipelineTrace",
    "NAREStepResult",
    "NonParametricMemory",
    "NonParametricMemoryConfig",
    "PipelineStageTrace",
    "PredictionEnergyLoss",
    "RetrievalResult",
    "RoutingDecision",
    "SubQAttentionConfig",
    "SubQFieldAttention",
    "BenchmarkRunConfig",
    "BenchmarkRunReport",
    "BenchmarkRunner",
    "BenchmarkSuite",
    "BenchmarkTask",
    "TemperatureState",
    "TraceMemory",
    "TraceMemoryConfig",
    "TraceReplay",
    "ValidationReport",
    "TaskEncoder",
    "build_pipeline_trace",
    "compare_ablation",
    "state_entropy",
    "theory_alignment_table",
    "summarize_validation",
]
