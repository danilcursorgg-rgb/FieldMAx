"""
Full NARE-Field PyTorch prototype — end-to-end training & evaluation.

Demonstrates all theory components working together:
1. Prediction energy minimization (free-energy objective)
2. Memory field with Hebbian plasticity and attractor dynamics
3. SubQ field attention (subquadratic complexity)
4. Anthill MoE routing with load-balancing
5. Trace memory (latent reasoning trajectories)
6. Non-parametric retrieval memory
7. Cognitive temperature (explore/exploit control)
8. Ablation study proving each component adds value
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import numpy as np

from narefield.pt import (
    FreeEnergyTrainer,
    NAREConfig,
    NAREFieldModel,
)


def generate_synthetic_reasoning(count: int, dim: int, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic reasoning tasks: input -> target mapping with learnable structure."""
    rng = np.random.default_rng(seed)
    inputs_list = []
    targets_list = []
    for idx in range(count):
        left = (idx * 7 + 3) % 19
        right = (idx * 5 + 11) % 23
        # Encode prompt as a structured vector
        inp = np.zeros(dim, dtype=np.float32)
        for i, val in enumerate(f"a={left} b={right}".encode()):
            bucket = (val + 31 * i) % dim
            sign = 1.0 if val % 2 == 0 else -1.0
            inp[bucket] += sign * (1.0 + (val % 17) / 17.0)
        norm = np.linalg.norm(inp)
        if norm > 0:
            inp /= norm

        # Encode target (answer)
        if idx % 3 == 0:
            answer = left + right
            tag = f"add:{answer}"
        elif idx % 3 == 1:
            answer = left * right
            tag = f"mul:{answer}"
        else:
            answer = (left + right) % 7
            tag = f"mod:{answer}"

        tgt = np.zeros(dim, dtype=np.float32)
        for i, val in enumerate(tag.encode()):
            bucket = (val + 31 * i) % dim
            sign = 1.0 if val % 2 == 0 else -1.0
            tgt[bucket] += sign * (1.0 + (val % 17) / 17.0)
        tnorm = np.linalg.norm(tgt)
        if tnorm > 0:
            tgt /= tnorm

        inputs_list.append(inp)
        targets_list.append(tgt)

    return torch.tensor(np.array(inputs_list)), torch.tensor(np.array(targets_list))


def run_training(
    config: NAREConfig,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epochs: int,
    batch_size: int,
    label: str = "full",
) -> tuple[NAREFieldModel, list[dict]]:
    """Train model and return (model, history)."""
    model = NAREFieldModel(config)
    trainer = FreeEnergyTrainer(model, lr=1e-3, router_lr=3e-3)

    history = []
    n = inputs.shape[0]
    for epoch in range(epochs):
        epoch_losses = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_inp = inputs[start:end]
            batch_tgt = targets[start:end]
            result, update = trainer.train_step(batch_inp, batch_tgt)
            epoch_losses.append(update.total_energy)

        mean_loss = float(np.mean(epoch_losses))
        last = update.as_dict()
        history.append({
            "epoch": epoch,
            "mean_energy": mean_loss,
            "prediction_error": last["prediction_error"],
            "routing_balance": last["routing_balance"],
            "routing_load_std": last["routing_load_std"],
            "cognitive_temperature": last["cognitive_temperature"],
        })
        print(f"  [{label}] Epoch {epoch+1}/{epochs}  energy={mean_loss:.4f}  "
              f"recon={last['prediction_error']:.4f}  route_std={last['routing_load_std']:.4f}  "
              f"temp={last['cognitive_temperature']:.3f}")

    return model, history


def evaluate_model(model: NAREFieldModel, inputs: torch.Tensor, targets: torch.Tensor) -> dict:
    """Evaluate model without learning; return metrics."""
    model.eval()
    with torch.no_grad():
        result = model(inputs, targets, learn=False)
    return {
        "loss_total": result.loss.total.item(),
        "loss_reconstruction": result.loss.reconstruction.item(),
        "routing_load_std": result.metrics["routing_load_std"],
        "memory_capacity_usage": result.metrics["memory_capacity_usage"],
        "retrieval_hit_rate": result.metrics["retrieval_hit_rate"],
        "trace_hit_rate": result.metrics["trace_hit_rate"],
        "attention_complexity": result.metrics["attention_complexity_estimate"],
        "cognitive_temperature": result.metrics["cognitive_temperature"],
        "cognitive_mode": result.metrics["cognitive_mode"],
        "active_parameter_ratio": result.metrics["active_parameter_ratio"],
    }


def main() -> None:
    print("=" * 70)
    print("   NARE-Field PyTorch Prototype — End-to-End Training & Evaluation")
    print("=" * 70)

    # Configuration
    dim = 32
    num_tasks = 48
    epochs = 10
    batch_size = 8
    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate data
    print(f"\nGenerating {num_tasks} synthetic reasoning tasks (dim={dim})...")
    inputs, targets = generate_synthetic_reasoning(num_tasks, dim, seed=seed)
    print(f"  Input shape: {inputs.shape}, Target shape: {targets.shape}")

    # Full model config
    full_config = NAREConfig(
        dim=dim,
        memory_capacity=64,
        num_experts=4,
        top_k=2,
        seed=seed,
        attention_buckets=8,
        attention_top_buckets=2,
        retrieval_capacity=128,
        trace_capacity=64,
    )

    # =========================================================================
    # PHASE 1: Train full model
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 1: Training FULL NARE-Field model")
    print("=" * 70)
    full_model, full_history = run_training(full_config, inputs, targets, epochs, batch_size, label="FULL")

    # =========================================================================
    # PHASE 2: Ablation — no attention, no retrieval, no trace
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 2: Ablation — memory-only (no attention/retrieval/trace)")
    print("=" * 70)
    ablated_config = NAREConfig(
        dim=dim,
        memory_capacity=64,
        num_experts=4,
        top_k=2,
        seed=seed,
        attention_buckets=8,
        attention_top_buckets=2,
        attention_mix=0.0,
        retrieval_mix=0.0,
        trace_mix=0.0,
        retrieval_capacity=128,
        trace_capacity=64,
    )
    _, ablated_history = run_training(ablated_config, inputs, targets, epochs, batch_size, label="ABLATED")

    # =========================================================================
    # PHASE 3: Evaluation
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 3: Evaluation on training set (held-state)")
    print("=" * 70)
    full_eval = evaluate_model(full_model, inputs, targets)

    print("\n--- Full Model Evaluation ---")
    for k, v in full_eval.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # =========================================================================
    # PHASE 4: Evidence checks
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 4: Evidence Checks (pass/warn/fail)")
    print("=" * 70)

    checks = []

    # Check 1: Energy trend — does loss decrease?
    early_energy = float(np.mean([h["mean_energy"] for h in full_history[:3]]))
    late_energy = float(np.mean([h["mean_energy"] for h in full_history[-3:]]))
    ratio = late_energy / max(early_energy, 1e-12)
    status = "pass" if ratio <= 1.0 else ("warn" if ratio <= 1.1 else "fail")
    checks.append({
        "name": "prediction_energy_trend",
        "status": status,
        "observed": ratio,
        "threshold": 1.0,
        "detail": f"early={early_energy:.4f} -> late={late_energy:.4f}",
    })
    print(f"\n  [{'PASS' if status=='pass' else status.upper()}] Energy trend: "
          f"ratio={ratio:.4f} (early={early_energy:.4f}, late={late_energy:.4f})")

    # Check 2: Full model beats ablation
    full_final = full_history[-1]["mean_energy"]
    ablated_final = ablated_history[-1]["mean_energy"]
    delta = ablated_final - full_final
    status = "pass" if delta > 0 else ("warn" if delta > -0.02 else "fail")
    checks.append({
        "name": "full_beats_ablation",
        "status": status,
        "observed": delta,
        "threshold": 0.0,
        "detail": f"full={full_final:.4f} vs ablated={ablated_final:.4f}",
    })
    print(f"  [{'PASS' if status=='pass' else status.upper()}] Full vs ablation: "
          f"delta={delta:.4f} (full={full_final:.4f}, ablated={ablated_final:.4f})")

    # Check 3: Memory reuse (retrieval + trace hit rates)
    combined_hit = 0.5 * (full_eval["retrieval_hit_rate"] + full_eval["trace_hit_rate"])
    status = "pass" if combined_hit >= 0.3 else ("warn" if combined_hit > 0 else "fail")
    checks.append({
        "name": "memory_reuse",
        "status": status,
        "observed": combined_hit,
        "threshold": 0.3,
        "detail": f"retrieval={full_eval['retrieval_hit_rate']:.2f}, trace={full_eval['trace_hit_rate']:.2f}",
    })
    print(f"  [{'PASS' if status=='pass' else status.upper()}] Memory reuse: "
          f"combined={combined_hit:.3f} (retrieval={full_eval['retrieval_hit_rate']:.2f}, "
          f"trace={full_eval['trace_hit_rate']:.2f})")

    # Check 4: Routing stability
    load_std = full_eval["routing_load_std"]
    status = "pass" if load_std <= 0.3 else ("warn" if load_std <= 0.45 else "fail")
    checks.append({
        "name": "routing_stability",
        "status": status,
        "observed": load_std,
        "threshold": 0.3,
    })
    print(f"  [{'PASS' if status=='pass' else status.upper()}] Routing stability: "
          f"load_std={load_std:.4f}")

    # Check 5: SubQ attention is subquadratic
    n_tokens = inputs.shape[0]
    complexity = full_eval["attention_complexity"]
    ratio_sq = complexity / (n_tokens ** 2)
    status = "pass" if ratio_sq < 0.75 else ("warn" if ratio_sq < 1.0 else "fail")
    checks.append({
        "name": "subquadratic_attention",
        "status": status,
        "observed": ratio_sq,
        "threshold": 0.75,
        "detail": f"complexity={complexity:.0f} vs n^2={n_tokens**2}",
    })
    print(f"  [{'PASS' if status=='pass' else status.upper()}] SubQ attention: "
          f"ratio={ratio_sq:.4f} (complexity={complexity:.0f}, n^2={n_tokens**2})")

    # Score
    score = sum(1.0 if c["status"] == "pass" else (0.5 if c["status"] == "warn" else 0.0) for c in checks) / len(checks)

    # =========================================================================
    # PHASE 5: Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"  NARE-Field Evidence Score: {score:.0%} ({sum(1 for c in checks if c['status']=='pass')}/{len(checks)} passed)")
    print("=" * 70)

    report = {
        "model": "NARE-Field PyTorch Prototype",
        "dim": dim,
        "num_tasks": num_tasks,
        "epochs": epochs,
        "score": score,
        "training_history": {
            "full": full_history,
            "ablated": ablated_history,
        },
        "evaluation": full_eval,
        "evidence_checks": checks,
        "theory_components": {
            "prediction_energy": "MSE(x, x_hat) + invariant + memory_capacity + routing_balance",
            "memory_field": "Hopfield attractor memory with predictive inference + Hebbian writes",
            "subq_attention": f"bucketed potential attention O(n*k) vs O(n^2), ratio={ratio_sq:.4f}",
            "anthill_routing": f"top-k MoE with KL load-balancing, stability={load_std:.4f}",
            "trace_memory": f"latent reasoning trajectories, hit_rate={full_eval['trace_hit_rate']:.2f}",
            "retrieval_memory": f"non-parametric key-value store, hit_rate={full_eval['retrieval_hit_rate']:.2f}",
            "cognitive_temperature": f"{full_eval['cognitive_temperature']:.3f} ({full_eval['cognitive_mode']})",
        },
    }

    output_path = Path("reports/pt_full_demo_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"\nFull report saved to {output_path}")

    # Print parameter counts
    total_params = sum(p.numel() for p in full_model.parameters())
    trainable_params = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
    print(f"\nModel: {total_params:,} total params, {trainable_params:,} trainable")
    print(f"Active parameter ratio: {full_eval['active_parameter_ratio']:.2f} "
          f"(only {full_eval['active_parameter_ratio']*100:.0f}% of experts active per token)")

    if score < 1.0:
        print("\nSome checks did not pass. See report for details.")
        sys.exit(1)
    else:
        print("\nAll evidence checks passed.")


if __name__ == "__main__":
    main()
