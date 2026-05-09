from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchitectureStage:
    name: str
    theory: str
    implementation: str
    energy_role: str

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "theory": self.theory,
            "implementation": self.implementation,
            "energy_role": self.energy_role,
        }


@dataclass(frozen=True)
class NAREArchitectureSpec:
    name: str
    principle: str
    stages: tuple[ArchitectureStage, ...]

    def as_dict(self) -> dict[str, str | list[dict[str, str]]]:
        return {
            "name": self.name,
            "principle": self.principle,
            "stages": [stage.as_dict() for stage in self.stages],
        }

    def pipeline(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages)


NARE_FIELD_ARCHITECTURE = NAREArchitectureSpec(
    name="NARE-Field benchmark prototype",
    principle="Minimize prediction energy while reconstructing knowledge from compact invariants, field memory, latent traces, and external context.",
    stages=(
        ArchitectureStage(
            name="SubQ field attention",
            theory="Tokens interact through high-potential regions instead of full pairwise attention.",
            implementation="SubQFieldAttention buckets latent states and attends only active potential buckets.",
            energy_role="Reduces attention work from O(n^2) toward O(n log n + b^2).",
        ),
        ArchitectureStage(
            name="Anthill routing",
            theory="A swarm of micro-experts specializes without routing collapse.",
            implementation="AnthillRouter applies top-k softmax routing with KL load-balancing.",
            energy_role="Activates only a sparse expert subset and penalizes uneven load.",
        ),
        ArchitectureStage(
            name="Memory field inference",
            theory="Memory is an attractor field updated by local prediction-error dynamics.",
            implementation="MemoryField iteratively moves latent state toward field predictions and writes Hebbian attractors.",
            energy_role="Turns reconstruction error into local state and field updates.",
        ),
        ArchitectureStage(
            name="Non-parametric retrieval",
            theory="Parameters store invariants; specific context is retrieved at runtime.",
            implementation="NonParametricMemory retrieves key-value context by cosine similarity.",
            energy_role="Moves factual storage out of weights and into external memory.",
        ),
        ArchitectureStage(
            name="Trace replay",
            theory="Reasoning is amortized by replaying compressed latent routes.",
            implementation="TraceMemory stores latent deltas and reconstructs similar trajectories.",
            energy_role="Reduces repeated computation for similar tasks.",
        ),
        ArchitectureStage(
            name="Cognitive temperature",
            theory="Exploration rises with cognitive energy and state entropy.",
            implementation="CognitiveTemperature maps energy and entropy to explore/exploit mode.",
            energy_role="Controls whether the system stabilizes or searches new routes.",
        ),
        ArchitectureStage(
            name="Free-energy update",
            theory="Learning minimizes reconstruction error plus information, memory, and routing costs.",
            implementation="FreeEnergyTrainer coordinates model step, memory writes, retrieval writes, trace writes, and router balancing.",
            energy_role="Produces the benchmarkable total prediction/free-energy objective.",
        ),
    ),
)


def theory_alignment_table() -> list[dict[str, str]]:
    return [stage.as_dict() for stage in NARE_FIELD_ARCHITECTURE.stages]
