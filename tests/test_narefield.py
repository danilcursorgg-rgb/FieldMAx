import numpy as np

from narefield import (
    AnthillLayer,
    AnthillRouterConfig,
    BenchmarkRunConfig,
    BenchmarkRunner,
    BenchmarkSuite,
    CognitiveInvariant,
    EvidenceRunner,
    CognitiveTemperature,
    FreeEnergyTrainer,
    MemoryField,
    MemoryFieldConfig,
    NAREConfig,
    NAREFieldModel,
    NARE_FIELD_ARCHITECTURE,
    NonParametricMemory,
    NonParametricMemoryConfig,
    PredictionEnergyLoss,
    SubQAttentionConfig,
    SubQFieldAttention,
    TraceMemory,
    TraceMemoryConfig,
    compare_ablation,
    summarize_validation,
    theory_alignment_table,
)


def test_prediction_energy_includes_regularizers() -> None:
    loss_fn = PredictionEnergyLoss(invariant_weight=1.0, memory_weight=1.0, routing_weight=1.0)
    target = np.zeros((2, 3))
    prediction = np.ones((2, 3))
    loss = loss_fn(
        target,
        prediction,
        invariant=CognitiveInvariant.from_state(prediction),
        memory_stats={"capacity_usage": 0.5},
        routing_load=np.array([0.9, 0.1]),
    )
    assert loss.reconstruction == 1.0
    assert loss.invariant > 0.0
    assert loss.memory == 0.5
    assert loss.routing_balance > 0.0
    assert loss.total > loss.reconstruction


def test_memory_field_moves_toward_existing_attractor() -> None:
    field = MemoryField(MemoryFieldConfig(dim=3, capacity=4, inference_steps=5, inference_rate=0.4))
    attractor = np.array([1.0, 0.0, 0.0])
    field.write(attractor)
    query = np.array([0.8, 0.2, 0.0])
    trace = field.infer(query)
    initial_distance = np.linalg.norm(query / np.linalg.norm(query) - field.attractors[0])
    final_distance = np.linalg.norm(trace.state - field.attractors[0])
    assert final_distance < initial_distance
    assert field.stats()["capacity_usage"] == 0.25


def test_anthill_router_top_k_and_balance_loss() -> None:
    layer = AnthillLayer.random(AnthillRouterConfig(input_dim=4, num_experts=3, top_k=2, seed=1))
    output, decision = layer(np.ones((5, 4)))
    assert output.shape == (5, 4)
    assert decision.probabilities.shape == (5, 3)
    assert np.allclose(decision.probabilities.sum(axis=1), 1.0)
    assert np.all(np.count_nonzero(decision.probabilities, axis=1) == 2)
    assert decision.load_balance_loss >= 0.0


def test_nare_model_step_and_validation_summary() -> None:
    rng = np.random.default_rng(3)
    model = NAREFieldModel(NAREConfig(dim=5, memory_capacity=10, num_experts=4, top_k=2, seed=3))
    records = []
    for _ in range(3):
        result = model.step(rng.normal(size=(4, 5)), learn=True)
        records.append(model.validation_record(result))
        assert result.prediction.shape == (4, 5)
        assert result.loss.total >= 0.0
        assert result.metrics["active_parameter_ratio"] == 0.5
        assert result.metrics["attention_complexity_estimate"] > 0.0
    report = summarize_validation(records)
    assert report.mean_loss >= 0.0
    assert report.active_experts > 0
    assert report.memory_capacity_usage > 0.0
    assert report.mean_attention_complexity > 0.0


def test_subq_attention_estimates_subquadratic_work() -> None:
    rng = np.random.default_rng(5)
    tokens = rng.normal(size=(16, 6))
    attention = SubQFieldAttention(SubQAttentionConfig(dim=6, num_buckets=4, top_buckets=2, seed=5))
    result = attention(tokens)
    assert result.output.shape == tokens.shape
    assert result.active_tokens > 0
    assert result.complexity_estimate < tokens.shape[0] ** 2


def test_trace_and_nonparametric_memory_reconstruct_context() -> None:
    field = MemoryField(MemoryFieldConfig(dim=4, capacity=8))
    trace = field.write(np.array([1.0, 0.0, 0.0, 0.0]))
    trace_memory = TraceMemory(TraceMemoryConfig(dim=4, capacity=4, top_k=1))
    trace_memory.record([trace])
    replay = trace_memory.replay(np.array([[0.8, 0.2, 0.0, 0.0]]))
    assert replay.reconstruction.shape == (1, 4)
    assert replay.hit_rate > 0.0

    external = NonParametricMemory(NonParametricMemoryConfig(dim=4, capacity=4, top_k=1))
    external.write(np.array([[1.0, 0.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0, 0.0]]))
    retrieval = external.retrieve(np.array([[0.9, 0.1, 0.0, 0.0]]))
    assert retrieval.values.shape == (1, 4)
    assert retrieval.hit_rate > 0.0


def test_cognitive_temperature_and_free_energy_trainer() -> None:
    rng = np.random.default_rng(9)
    controller = CognitiveTemperature()
    state = controller.update(energy=2.0, state=rng.normal(size=(3, 4)))
    assert state.temperature > 0.0
    assert state.mode in {"explore", "exploit"}

    model = NAREFieldModel(NAREConfig(dim=4, memory_capacity=8, num_experts=3, top_k=2, seed=9))
    trainer = FreeEnergyTrainer(model)
    result, update = trainer.train_step(rng.normal(size=(5, 4)))
    assert result.loss.total == update.total_energy
    assert len(trainer.history) == 1


def test_ablation_report_tracks_quality_delta() -> None:
    rng = np.random.default_rng(11)
    full_model = NAREFieldModel(NAREConfig(dim=4, memory_capacity=8, num_experts=3, top_k=2, seed=11))
    ablated_model = NAREFieldModel(
        NAREConfig(dim=4, memory_capacity=8, num_experts=3, top_k=2, seed=11, attention_mix=0.0, trace_mix=0.0)
    )
    records = []
    ablated_records = []
    for _ in range(2):
        batch = rng.normal(size=(4, 4))
        records.append(full_model.validation_record(full_model.step(batch, learn=True)))
        ablated_records.append(ablated_model.validation_record(ablated_model.step(batch, learn=True)))
    report = summarize_validation(records)
    ablated_report = summarize_validation(ablated_records)
    ablation = compare_ablation(report, ablated_report, "without_trace_attention")
    assert ablation.variant == "without_trace_attention"


def test_architecture_spec_maps_theory_to_code() -> None:
    pipeline = NARE_FIELD_ARCHITECTURE.pipeline()
    assert pipeline[0] == "SubQ field attention"
    assert "Free-energy update" in pipeline
    alignment = theory_alignment_table()
    assert len(alignment) == len(pipeline)
    assert all("implementation" in row for row in alignment)


def test_benchmark_runner_prepares_synthetic_ablation_matrix() -> None:
    suite = BenchmarkSuite.synthetic_reasoning(count=6)
    runner = BenchmarkRunner(
        model_config=NAREConfig(dim=8, memory_capacity=16, num_experts=3, top_k=2, seed=13),
        run_config=BenchmarkRunConfig(epochs=1, batch_size=3),
    )
    report = runner.run(suite)
    assert report.task_count == 6
    assert report.validation.mean_loss >= 0.0

    matrix = runner.run_ablation_matrix(suite)
    assert matrix.baseline.variant == "baseline"
    assert len(matrix.variants) > 0
    assert len(matrix.comparisons) == len(matrix.variants)
    assert "no_trace" in {comparison.variant for comparison in matrix.comparisons}


def test_model_step_exposes_pipeline_energy_ledger() -> None:
    rng = np.random.default_rng(17)
    model = NAREFieldModel(NAREConfig(dim=6, memory_capacity=12, num_experts=3, top_k=2, seed=17))
    result = model.step(rng.normal(size=(4, 6)), learn=True)
    assert "SubQ field attention" in result.pipeline_trace.stage_names()
    ledger_total = sum(term.weighted_value for term in result.pipeline_trace.energy_ledger)
    assert abs(ledger_total - result.loss.total) < 1e-9
    assert result.pipeline_trace.weighted_regularization() >= 0.0


def test_evidence_runner_reports_hypotheses_without_hiding_failures() -> None:
    suite = BenchmarkSuite.synthetic_reasoning(count=6)
    runner = EvidenceRunner(
        model_config=NAREConfig(dim=8, memory_capacity=16, num_experts=3, top_k=2, seed=19),
        run_config=BenchmarkRunConfig(epochs=2, batch_size=3),
    )
    report = runner.run(suite)
    assert 0.0 <= report.score <= 1.0
    assert {check.name for check in report.checks} == {
        "prediction_energy_trend",
        "full_architecture_beats_memory_only_ablation",
        "non_parametric_and_trace_memory_reuse",
        "anthill_routing_stability",
        "subquadratic_attention_scaling",
    }
    assert {check.status for check in report.checks} <= {"pass", "warn", "fail"}
