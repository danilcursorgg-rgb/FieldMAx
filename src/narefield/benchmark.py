from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .metrics import AblationReport, ValidationReport, compare_ablation, summarize_validation
from .model import NAREConfig, NAREFieldModel
from .trainer import FreeEnergyTrainer, FreeEnergyUpdate


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    prompt: str
    target: str
    category: str = "reasoning"

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "prompt": self.prompt,
            "target": self.target,
            "category": self.category,
        }


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    tasks: tuple[BenchmarkTask, ...]

    @classmethod
    def synthetic_reasoning(cls, count: int = 24) -> "BenchmarkSuite":
        tasks: list[BenchmarkTask] = []
        for idx in range(count):
            left = (idx * 7 + 3) % 19
            right = (idx * 5 + 11) % 23
            if idx % 3 == 0:
                answer = left + right
                prompt = f"If a={left} and b={right}, compute a+b."
                target = f"rule:add answer:{answer}"
                category = "arithmetic_add"
            elif idx % 3 == 1:
                answer = left * right
                prompt = f"If a={left} and b={right}, compute a*b."
                target = f"rule:multiply answer:{answer}"
                category = "arithmetic_multiply"
            else:
                answer = (left + right) % 7
                prompt = f"If a={left} and b={right}, compute (a+b) mod 7."
                target = f"rule:modular answer:{answer}"
                category = "modular_reasoning"
            tasks.append(BenchmarkTask(name=f"synthetic_{idx:03d}", prompt=prompt, target=target, category=category))
        return cls(name="synthetic_reasoning", tasks=tuple(tasks))

    @classmethod
    def from_jsonl(cls, path: str | Path, *, name: str | None = None) -> "BenchmarkSuite":
        source = Path(path)
        tasks: list[BenchmarkTask] = []
        with source.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("each JSONL row must be an object")
                prompt = _string_field(item, "prompt", "question", "input")
                target = _string_field(item, "target", "answer", "output", default=prompt)
                category = _string_field(item, "category", "task", default="reasoning")
                task_name = _string_field(item, "name", "id", default=f"task_{idx:05d}")
                tasks.append(BenchmarkTask(name=task_name, prompt=prompt, target=target, category=category))
        if not tasks:
            raise ValueError("benchmark suite must contain at least one task")
        return cls(name=name or source.stem, tasks=tuple(tasks))

    def as_dict(self) -> dict[str, str | list[dict[str, str]]]:
        return {"name": self.name, "tasks": [task.as_dict() for task in self.tasks]}


@dataclass(frozen=True)
class TaskEncoder:
    dim: int

    def encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=float)
        payload = text.encode("utf-8")
        if not payload:
            return vector
        for index, value in enumerate(payload):
            bucket = (value + 31 * index) % self.dim
            sign = 1.0 if value % 2 == 0 else -1.0
            vector[bucket] += sign * (1.0 + (value % 17) / 17.0)
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            return vector
        return vector / norm

    def encode_tasks(self, tasks: Iterable[BenchmarkTask]) -> tuple[np.ndarray, np.ndarray]:
        rows = list(tasks)
        inputs = np.vstack([self.encode(task.prompt) for task in rows])
        targets = np.vstack([self.encode(task.target) for task in rows])
        return inputs, targets


@dataclass(frozen=True)
class BenchmarkRunConfig:
    epochs: int = 3
    batch_size: int = 4

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def as_dict(self) -> dict[str, int]:
        return {"epochs": self.epochs, "batch_size": self.batch_size}


@dataclass(frozen=True)
class BenchmarkRunReport:
    suite: str
    variant: str
    task_count: int
    config: BenchmarkRunConfig
    validation: ValidationReport
    updates: tuple[FreeEnergyUpdate, ...]

    def as_dict(self) -> dict[str, str | int | dict[str, int] | dict[str, float | int] | list[dict[str, float | int]]]:
        return {
            "suite": self.suite,
            "variant": self.variant,
            "task_count": self.task_count,
            "config": self.config.as_dict(),
            "validation": self.validation.as_dict(),
            "updates": [update.as_dict() for update in self.updates],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class AblationVariant:
    name: str
    attention_mix: float | None = None
    retrieval_mix: float | None = None
    trace_mix: float | None = None
    memory_mix: float | None = None
    anthill_mix: float | None = None

    def apply(self, config: NAREConfig) -> NAREConfig:
        return replace(
            config,
            attention_mix=config.attention_mix if self.attention_mix is None else self.attention_mix,
            retrieval_mix=config.retrieval_mix if self.retrieval_mix is None else self.retrieval_mix,
            trace_mix=config.trace_mix if self.trace_mix is None else self.trace_mix,
            memory_mix=config.memory_mix if self.memory_mix is None else self.memory_mix,
            anthill_mix=config.anthill_mix if self.anthill_mix is None else self.anthill_mix,
        )


@dataclass(frozen=True)
class AblationMatrixReport:
    baseline: BenchmarkRunReport
    variants: tuple[BenchmarkRunReport, ...]
    comparisons: tuple[AblationReport, ...]

    def as_dict(
        self,
    ) -> dict[str, dict[str, str | int | dict[str, int] | dict[str, float | int] | list[dict[str, float | int]]] | list[dict[str, str | float]] | list[dict[str, str | int | dict[str, int] | dict[str, float | int] | list[dict[str, float | int]]]]]:
        return {
            "baseline": self.baseline.as_dict(),
            "variants": [variant.as_dict() for variant in self.variants],
            "comparisons": [comparison.as_dict() for comparison in self.comparisons],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)


DEFAULT_ABLATIONS: tuple[AblationVariant, ...] = (
    AblationVariant(name="no_attention", attention_mix=0.0),
    AblationVariant(name="no_retrieval", retrieval_mix=0.0),
    AblationVariant(name="no_trace", trace_mix=0.0),
    AblationVariant(name="memory_only", attention_mix=0.0, retrieval_mix=0.0, trace_mix=0.0, anthill_mix=0.0),
)


@dataclass(frozen=True)
class BenchmarkRunner:
    model_config: NAREConfig
    run_config: BenchmarkRunConfig = BenchmarkRunConfig()

    def run(self, suite: BenchmarkSuite, *, variant: str = "baseline") -> BenchmarkRunReport:
        encoder = TaskEncoder(dim=self.model_config.dim)
        inputs, targets = encoder.encode_tasks(suite.tasks)
        model = NAREFieldModel(self.model_config)
        trainer = FreeEnergyTrainer(model)
        records: list[Mapping[str, float | np.ndarray]] = []
        updates: list[FreeEnergyUpdate] = []
        for _ in range(self.run_config.epochs):
            for batch_inputs, batch_targets in _batches(inputs, targets, self.run_config.batch_size):
                result, update = trainer.train_step(batch_inputs, batch_targets)
                records.append(model.validation_record(result))
                updates.append(update)
        return BenchmarkRunReport(
            suite=suite.name,
            variant=variant,
            task_count=len(suite.tasks),
            config=self.run_config,
            validation=summarize_validation(records),
            updates=tuple(updates),
        )

    def run_ablation_matrix(
        self,
        suite: BenchmarkSuite,
        variants: tuple[AblationVariant, ...] = DEFAULT_ABLATIONS,
    ) -> AblationMatrixReport:
        baseline = self.run(suite, variant="baseline")
        variant_reports: list[BenchmarkRunReport] = []
        comparisons: list[AblationReport] = []
        for variant in variants:
            runner = BenchmarkRunner(model_config=variant.apply(self.model_config), run_config=self.run_config)
            report = runner.run(suite, variant=variant.name)
            variant_reports.append(report)
            comparisons.append(compare_ablation(baseline.validation, report.validation, variant.name))
        return AblationMatrixReport(
            baseline=baseline,
            variants=tuple(variant_reports),
            comparisons=tuple(comparisons),
        )


def _batches(inputs: np.ndarray, targets: np.ndarray, batch_size: int) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    for start in range(0, inputs.shape[0], batch_size):
        stop = start + batch_size
        yield inputs[start:stop], targets[start:stop]


def _string_field(item: Mapping[object, object], *keys: str, default: str = "") -> str:
    for key in keys:
        value = item.get(key)
        if value is not None:
            return str(value)
    return default
