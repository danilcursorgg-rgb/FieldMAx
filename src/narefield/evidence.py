from __future__ import annotations

import json
from dataclasses import dataclass, replace

import numpy as np

from .attention import SubQAttentionConfig, SubQFieldAttention
from .benchmark import BenchmarkRunConfig, BenchmarkRunner, BenchmarkSuite, TaskEncoder
from .model import NAREConfig, NAREFieldModel
from .trainer import FreeEnergyTrainer


@dataclass(frozen=True)
class HypothesisEvidence:
    name: str
    status: str
    metric: str
    observed: float
    threshold: float
    details: str

    def as_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "status": self.status,
            "metric": self.metric,
            "observed": self.observed,
            "threshold": self.threshold,
            "details": self.details,
        }


@dataclass(frozen=True)
class EvidenceSeries:
    reconstruction: tuple[float, ...]
    routing_load_std: tuple[float, ...]
    retrieval_hit_rate: tuple[float, ...]
    trace_hit_rate: tuple[float, ...]

    def tail_mean(self, values: tuple[float, ...]) -> float:
        if not values:
            return 0.0
        split = max(len(values) // 2, 1)
        return float(np.mean(values[-split:]))

    def head_mean(self, values: tuple[float, ...]) -> float:
        if not values:
            return 0.0
        split = max(len(values) // 2, 1)
        return float(np.mean(values[:split]))


@dataclass(frozen=True)
class NAREEvidenceReport:
    suite: str
    checks: tuple[HypothesisEvidence, ...]

    @property
    def score(self) -> float:
        if not self.checks:
            return 0.0
        points = 0.0
        for check in self.checks:
            if check.status == "pass":
                points += 1.0
            elif check.status == "warn":
                points += 0.5
        return points / len(self.checks)

    def as_dict(self) -> dict[str, float | str | list[dict[str, float | str]]]:
        return {
            "suite": self.suite,
            "score": self.score,
            "checks": [check.as_dict() for check in self.checks],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class EvidenceRunner:
    model_config: NAREConfig
    run_config: BenchmarkRunConfig = BenchmarkRunConfig(epochs=4, batch_size=4)

    def run(self, suite: BenchmarkSuite) -> NAREEvidenceReport:
        series = self._train_series(suite, self.model_config)
        checks = (
            self._check_energy_trend(series),
            self._check_ablation_value(suite),
            self._check_memory_reuse(series),
            self._check_routing_stability(series),
            self._check_subq_scaling(),
        )
        return NAREEvidenceReport(suite=suite.name, checks=checks)

    def _train_series(self, suite: BenchmarkSuite, config: NAREConfig) -> EvidenceSeries:
        encoder = TaskEncoder(dim=config.dim)
        inputs, targets = encoder.encode_tasks(suite.tasks)
        model = NAREFieldModel(config)
        trainer = FreeEnergyTrainer(model)
        reconstruction: list[float] = []
        routing_load_std: list[float] = []
        retrieval_hit_rate: list[float] = []
        trace_hit_rate: list[float] = []
        for _ in range(self.run_config.epochs):
            for start in range(0, inputs.shape[0], self.run_config.batch_size):
                stop = start + self.run_config.batch_size
                result, _ = trainer.train_step(inputs[start:stop], targets[start:stop])
                reconstruction.append(result.loss.reconstruction)
                routing_load_std.append(float(result.metrics["routing_load_std"]))
                retrieval_hit_rate.append(float(result.metrics["retrieval_hit_rate"]))
                trace_hit_rate.append(float(result.metrics["trace_hit_rate"]))
        return EvidenceSeries(
            reconstruction=tuple(reconstruction),
            routing_load_std=tuple(routing_load_std),
            retrieval_hit_rate=tuple(retrieval_hit_rate),
            trace_hit_rate=tuple(trace_hit_rate),
        )

    def _check_energy_trend(self, series: EvidenceSeries) -> HypothesisEvidence:
        early = series.head_mean(series.reconstruction)
        late = series.tail_mean(series.reconstruction)
        ratio = late / max(early, 1e-12)
        if ratio <= 1.0:
            status = "pass"
        elif ratio <= 1.10:
            status = "warn"
        else:
            status = "fail"
        return HypothesisEvidence(
            name="prediction_energy_trend",
            status=status,
            metric="late_reconstruction / early_reconstruction",
            observed=ratio,
            threshold=1.0,
            details="Theory expects repeated exposure plus memory/retrieval writes to reduce reconstruction energy.",
        )

    def _check_ablation_value(self, suite: BenchmarkSuite) -> HypothesisEvidence:
        runner = BenchmarkRunner(model_config=self.model_config, run_config=self.run_config)
        baseline = runner.run(suite)
        memory_only_config = replace(
            self.model_config,
            attention_mix=0.0,
            retrieval_mix=0.0,
            trace_mix=0.0,
            anthill_mix=0.0,
        )
        memory_only = BenchmarkRunner(model_config=memory_only_config, run_config=self.run_config).run(suite, variant="memory_only")
        delta = memory_only.validation.mean_loss - baseline.validation.mean_loss
        if delta > 0.0:
            status = "pass"
        elif delta > -0.02:
            status = "warn"
        else:
            status = "fail"
        return HypothesisEvidence(
            name="full_architecture_beats_memory_only_ablation",
            status=status,
            metric="memory_only_mean_loss - full_mean_loss",
            observed=delta,
            threshold=0.0,
            details="Full architecture should outperform a field-memory-only ablation if attention/retrieval/trace add value.",
        )

    def _check_memory_reuse(self, series: EvidenceSeries) -> HypothesisEvidence:
        retrieval = series.tail_mean(series.retrieval_hit_rate)
        trace = series.tail_mean(series.trace_hit_rate)
        combined = 0.5 * (retrieval + trace)
        if combined >= 0.30:
            status = "pass"
        elif combined > 0.0:
            status = "warn"
        else:
            status = "fail"
        return HypothesisEvidence(
            name="non_parametric_and_trace_memory_reuse",
            status=status,
            metric="mean_tail_hit_rate",
            observed=combined,
            threshold=0.30,
            details="Repeated tasks should produce non-zero external retrieval and latent trace replay hit rates.",
        )

    def _check_routing_stability(self, series: EvidenceSeries) -> HypothesisEvidence:
        load_std = series.tail_mean(series.routing_load_std)
        if load_std <= 0.30:
            status = "pass"
        elif load_std <= 0.45:
            status = "warn"
        else:
            status = "fail"
        return HypothesisEvidence(
            name="anthill_routing_stability",
            status=status,
            metric="tail_routing_load_std",
            observed=load_std,
            threshold=0.30,
            details="Top-k Anthill routing should avoid expert collapse under load-balancing updates.",
        )

    def _check_subq_scaling(self) -> HypothesisEvidence:
        ratios: list[float] = []
        for tokens in (16, 32, 64):
            rng = np.random.default_rng(tokens + self.model_config.dim)
            values = rng.normal(size=(tokens, self.model_config.dim))
            attention = SubQFieldAttention(
                SubQAttentionConfig(
                    dim=self.model_config.dim,
                    num_buckets=max(4, self.model_config.attention_buckets),
                    top_buckets=min(2, max(1, self.model_config.attention_top_buckets)),
                    seed=self.model_config.seed,
                )
            )
            result = attention(values)
            ratios.append(result.complexity_estimate / float(tokens * tokens))
        worst_ratio = float(max(ratios))
        if worst_ratio < 0.75:
            status = "pass"
        elif worst_ratio < 1.0:
            status = "warn"
        else:
            status = "fail"
        return HypothesisEvidence(
            name="subquadratic_attention_scaling",
            status=status,
            metric="max(complexity_estimate / n^2)",
            observed=worst_ratio,
            threshold=0.75,
            details="Field attention should remain below dense pairwise work on longer synthetic contexts.",
        )
