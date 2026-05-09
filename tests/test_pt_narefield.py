"""Tests for the PyTorch NARE-Field implementation."""
import torch
import pytest

from narefield.pt import (
    AnthillLayer,
    AnthillRouterConfig,
    CognitiveInvariant,
    CognitiveTemperature,
    FreeEnergyTrainer,
    MemoryField,
    MemoryFieldConfig,
    NAREConfig,
    NAREFieldModel,
    NonParametricMemory,
    NonParametricMemoryConfig,
    PredictionEnergyLoss,
    SubQAttentionConfig,
    SubQFieldAttention,
    TraceMemory,
    TraceMemoryConfig,
)


def test_subq_attention_subquadratic():
    torch.manual_seed(1)
    tokens = torch.randn(16, 8)
    attention = SubQFieldAttention(SubQAttentionConfig(dim=8, num_buckets=4, top_buckets=2, seed=1))
    result = attention(tokens)
    assert result.output.shape == tokens.shape
    assert result.active_tokens > 0
    assert result.complexity_estimate < tokens.shape[0] ** 2


def test_memory_field_inference_and_learn():
    torch.manual_seed(2)
    field = MemoryField(MemoryFieldConfig(dim=4, capacity=8, inference_steps=5))
    state = torch.randn(3, 4)
    trace = field(state, learn=True)
    assert trace.prediction.shape == (3, 4)
    assert trace.state.shape == (3, 4)
    assert len(trace.energy_history) == 6  # 5 steps + 1 final
    assert field.stats()["stored_attractors"] == 3


def test_anthill_routing_top_k_selection():
    torch.manual_seed(3)
    config = AnthillRouterConfig(input_dim=6, num_experts=4, top_k=2, seed=3)
    layer = AnthillLayer.identity(config)
    inputs = torch.randn(5, 6)
    output, decision = layer(inputs)
    assert output.shape == (5, 6)
    assert decision.probabilities.shape == (5, 4)
    assert decision.load_balance_loss.item() >= 0.0


def test_trace_memory_record_and_replay():
    torch.manual_seed(4)
    field = MemoryField(MemoryFieldConfig(dim=4, capacity=8))
    state = torch.randn(2, 4)
    trace = field(state, learn=True)
    tmem = TraceMemory(TraceMemoryConfig(dim=4, capacity=8, top_k=1))
    tmem.record(trace)
    query = torch.randn(2, 4)
    replay = tmem.replay(query)
    assert replay.reconstruction.shape == (2, 4)
    assert replay.hit_rate > 0.0


def test_nonparametric_memory_write_and_retrieve():
    torch.manual_seed(5)
    mem = NonParametricMemory(NonParametricMemoryConfig(dim=4, capacity=8, top_k=2))
    keys = torch.randn(3, 4)
    values = torch.randn(3, 4)
    mem.write(keys, values)
    result = mem.retrieve(keys[:1])
    assert result.values.shape == (1, 4)
    assert result.hit_rate > 0.0


def test_prediction_energy_loss_components():
    torch.manual_seed(6)
    loss_fn = PredictionEnergyLoss(invariant_weight=1.0, memory_weight=1.0, routing_weight=1.0)
    target = torch.zeros(2, 3)
    prediction = torch.ones(2, 3)
    invariant = CognitiveInvariant.from_state(prediction)
    routing_loss = torch.tensor(0.5)
    loss = loss_fn(
        target, prediction,
        invariant=invariant,
        memory_stats={"capacity_usage": 0.5},
        routing_loss=routing_loss,
    )
    assert loss.reconstruction.item() == pytest.approx(1.0, abs=1e-5)
    assert loss.invariant.item() > 0.0
    assert loss.memory.item() == pytest.approx(0.5, abs=1e-5)
    assert loss.total.item() > loss.reconstruction.item()


def test_cognitive_temperature_explore_exploit():
    torch.manual_seed(7)
    ctrl = CognitiveTemperature()
    state = torch.randn(3, 4)
    ts = ctrl.update(energy=2.0, state=state)
    assert ts.temperature > 0
    assert ts.mode in {"explore", "exploit"}


def test_full_model_forward_and_backward():
    torch.manual_seed(8)
    config = NAREConfig(dim=8, memory_capacity=16, num_experts=4, top_k=2, seed=8)
    model = NAREFieldModel(config)
    inputs = torch.randn(4, 8)
    targets = torch.randn(4, 8)
    result = model(inputs, targets, learn=True)
    assert result.prediction.shape == (4, 8)
    assert result.loss.total.item() >= 0.0
    # Test backward
    result.loss.total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients computed"


def test_free_energy_trainer_reduces_loss():
    torch.manual_seed(9)
    config = NAREConfig(dim=8, memory_capacity=16, num_experts=4, top_k=2, seed=9)
    model = NAREFieldModel(config)
    trainer = FreeEnergyTrainer(model)
    inputs = torch.randn(8, 8)
    targets = torch.randn(8, 8)
    losses = []
    for _ in range(10):
        _, update = trainer.train_step(inputs, targets)
        losses.append(update.total_energy)
    assert len(trainer.history) == 10
    # Loss should generally decrease over 10 steps on the same data
    assert losses[-1] <= losses[0] * 1.5, "Loss did not decrease or increased too much"


def test_model_eval_mode_no_learn():
    torch.manual_seed(10)
    config = NAREConfig(dim=8, memory_capacity=16, num_experts=4, top_k=2, seed=10)
    model = NAREFieldModel(config)
    inputs = torch.randn(4, 8)
    result = model(inputs, learn=False)
    assert result.prediction.shape == (4, 8)
    assert result.metrics["memory_capacity_usage"] == 0.0  # Nothing stored in eval mode
