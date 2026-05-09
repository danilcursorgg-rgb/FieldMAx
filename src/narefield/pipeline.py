from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnergyLedgerTerm:
    name: str
    raw_value: float
    weight: float

    @property
    def weighted_value(self) -> float:
        return self.raw_value * self.weight

    def as_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "raw_value": self.raw_value,
            "weight": self.weight,
            "weighted_value": self.weighted_value,
        }


@dataclass(frozen=True)
class PipelineStageTrace:
    name: str
    signal: float
    metric: str
    interpretation: str

    def as_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "signal": self.signal,
            "metric": self.metric,
            "interpretation": self.interpretation,
        }


@dataclass(frozen=True)
class NAREPipelineTrace:
    stages: tuple[PipelineStageTrace, ...]
    energy_ledger: tuple[EnergyLedgerTerm, ...]
    total_energy: float
    cognitive_mode: str

    def as_dict(self) -> dict[str, float | str | list[dict[str, float | str]]]:
        return {
            "total_energy": self.total_energy,
            "cognitive_mode": self.cognitive_mode,
            "stages": [stage.as_dict() for stage in self.stages],
            "energy_ledger": [term.as_dict() for term in self.energy_ledger],
        }

    def stage_names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages)

    def weighted_regularization(self) -> float:
        return sum(term.weighted_value for term in self.energy_ledger if term.name != "reconstruction")


def build_pipeline_trace(
    *,
    reconstruction: float,
    invariant: float,
    memory: float,
    routing_balance: float,
    invariant_weight: float,
    memory_weight: float,
    routing_weight: float,
    total_energy: float,
    attention_complexity: float,
    routing_load_std: float,
    memory_energy: float,
    retrieval_hit_rate: float,
    trace_hit_rate: float,
    trace_energy: float,
    cognitive_temperature: float,
    cognitive_mode: str,
) -> NAREPipelineTrace:
    return NAREPipelineTrace(
        stages=(
            PipelineStageTrace(
                name="SubQ field attention",
                signal=attention_complexity,
                metric="attention_complexity_estimate",
                interpretation="lower than full pairwise attention when active buckets stay sparse",
            ),
            PipelineStageTrace(
                name="Anthill routing",
                signal=routing_load_std,
                metric="routing_load_std",
                interpretation="lower values mean less expert collapse",
            ),
            PipelineStageTrace(
                name="Memory field inference",
                signal=memory_energy,
                metric="mean_memory_trace_energy",
                interpretation="local predictive-coding energy after attractor inference",
            ),
            PipelineStageTrace(
                name="Non-parametric retrieval",
                signal=retrieval_hit_rate,
                metric="retrieval_hit_rate",
                interpretation="external context usage instead of weight storage",
            ),
            PipelineStageTrace(
                name="Trace replay",
                signal=trace_hit_rate,
                metric="trace_hit_rate",
                interpretation="reuse of compressed latent reasoning routes",
            ),
            PipelineStageTrace(
                name="Trace replay energy",
                signal=trace_energy,
                metric="trace_replay_energy",
                interpretation="energy of selected latent route memories",
            ),
            PipelineStageTrace(
                name="Cognitive temperature",
                signal=cognitive_temperature,
                metric="cognitive_temperature",
                interpretation="explore/exploit controller driven by energy and entropy",
            ),
        ),
        energy_ledger=(
            EnergyLedgerTerm(name="reconstruction", raw_value=reconstruction, weight=1.0),
            EnergyLedgerTerm(name="invariant", raw_value=invariant, weight=invariant_weight),
            EnergyLedgerTerm(name="memory_capacity", raw_value=memory, weight=memory_weight),
            EnergyLedgerTerm(name="routing_balance", raw_value=routing_balance, weight=routing_weight),
        ),
        total_energy=total_energy,
        cognitive_mode=cognitive_mode,
    )
