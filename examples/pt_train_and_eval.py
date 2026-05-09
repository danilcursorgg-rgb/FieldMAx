"""Train PyTorch NARE-Field on GSM8K reasoning trajectories, then evaluate.

Theory alignment (all six NARE-Field principles exercised):
  1. Prediction energy (free-energy objective)  — FreeEnergyTrainer
  2. Memory field (Hopfield attractors + Hebbian plasticity)
  3. SubQ attention (bucketed potential, O(n*k))
  4. Anthill routing (MoE + KL load-balancing)
  5. Trace memory (latent trajectory storage & replay)
  6. Non-parametric retrieval (external key-value store)
  7. Cognitive temperature (explore / exploit switching)

Usage::

    python examples/pt_train_and_eval.py                          # defaults
    python examples/pt_train_and_eval.py --model Qwen/Qwen2-0.5B # smaller LLM
    python examples/pt_train_and_eval.py --train-samples 50 --eval-samples 20
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch
from tqdm import tqdm


def extract_gsm8k_answer(answer_text: str) -> str:
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    return answer_text.strip()


def extract_number(text: str) -> str | None:
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PyTorch NARE-Field: LLM Bridge Training & Evaluation")
    p.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct", help="HuggingFace model ID")
    p.add_argument("--nare-dim", type=int, default=256, help="NARE field dimension (compressed)")
    p.add_argument("--train-samples", type=int, default=20, help="Number of GSM8K training samples")
    p.add_argument("--eval-samples", type=int, default=10, help="Number of test samples to evaluate")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs per sample")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate for NARE field + projections")
    p.add_argument("--router-lr", type=float, default=3e-3, help="Learning rate for router gate")
    p.add_argument("--alpha", type=float, default=0.15, help="Blending weight for NARE correction")
    p.add_argument("--max-new-tokens", type=int, default=200, help="Max tokens to generate")
    p.add_argument("--checkpoint", default="pt_nare_bridge.pt", help="Checkpoint path")
    p.add_argument("--report", default="reports/pt_train_eval_report.json", help="Report output path")
    p.add_argument("--num-experts", type=int, default=8, help="Number of Anthill experts")
    p.add_argument("--top-k", type=int, default=2, help="Top-k experts per token")
    p.add_argument("--memory-capacity", type=int, default=1024, help="Memory field capacity")
    p.add_argument("--trace-capacity", type=int, default=512, help="Trace memory capacity")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("  NARE-Field PyTorch Bridge: LLM + Field Training & Evaluation")
    print("=" * 70)
    print(f"  Model:         {args.model}")
    print(f"  NARE dim:      {args.nare_dim}")
    print(f"  Train samples: {args.train_samples}")
    print(f"  Eval samples:  {args.eval_samples}")
    print(f"  Epochs/sample: {args.epochs}")
    print(f"  Experts:       {args.num_experts} (top-{args.top_k})")
    print(f"  Alpha:         {args.alpha}")
    print()

    # ------------------------------------------------------------------
    # 1. Build bridge + field + trainer
    # ------------------------------------------------------------------
    from narefield.pt.bridge import BridgeConfig, NAREBridge
    from narefield.pt.generate import NAREGenerator
    from narefield.pt.model import NAREConfig, NAREFieldModel
    from narefield.pt.trainer import FreeEnergyTrainer

    print("Loading LLM and building NARE bridge...")
    bridge_cfg = BridgeConfig(model_id=args.model, nare_dim=args.nare_dim)
    bridge = NAREBridge(bridge_cfg)
    print(f"  LLM hidden dim: {bridge.llm_dim}")
    print(f"  NARE dim:       {bridge.nare_dim}")

    field_cfg = NAREConfig(
        dim=args.nare_dim,
        memory_capacity=args.memory_capacity,
        trace_capacity=args.trace_capacity,
        num_experts=args.num_experts,
        top_k=args.top_k,
    )
    field = NAREFieldModel(field_cfg)
    # Move field to same device as bridge
    device = bridge.device
    field = field.to(device)

    # Trainer optimises NARE field + bridge projection layers jointly
    bridge_params = list(bridge.proj_down.parameters()) + list(bridge.proj_up.parameters())
    bridge_params += list(bridge.norm_down.parameters()) + list(bridge.norm_up.parameters())

    router_params = list(field.anthill.router.parameters())
    router_ids = {id(p) for p in router_params}
    field_other = [p for p in field.parameters() if id(p) not in router_ids]

    optimizer = torch.optim.Adam([
        {"params": field_other + bridge_params, "lr": args.lr},
        {"params": router_params, "lr": args.router_lr},
    ])

    generator = NAREGenerator(bridge, field, alpha=args.alpha)

    # ------------------------------------------------------------------
    # 2. Load GSM8K
    # ------------------------------------------------------------------
    from datasets import load_dataset

    print("\nLoading GSM8K datasets...")
    train_dataset = load_dataset("gsm8k", "main", split="train")
    test_dataset = load_dataset("gsm8k", "main", split="test")
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # ------------------------------------------------------------------
    # 3. Training Phase
    # ------------------------------------------------------------------
    checkpoint_path = args.checkpoint
    training_history: list[dict] = []

    if os.path.exists(checkpoint_path):
        print(f"\n--- Phase 1: Loading checkpoint from {checkpoint_path} ---")
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        field.load_state_dict(state["field"], strict=False)
        bridge.proj_down.load_state_dict(state["proj_down"])
        bridge.proj_up.load_state_dict(state["proj_up"])
        bridge.norm_down.load_state_dict(state["norm_down"])
        bridge.norm_up.load_state_dict(state["norm_up"])
        training_history = state.get("history", [])
        print(f"  Loaded {len(training_history)} training records")
    else:
        num_train = min(args.train_samples, len(train_dataset))
        print(f"\n--- Phase 1: Training on {num_train} reasoning trajectories ---")
        print(f"    ({args.epochs} epochs per sample, gradient-based)")

        field.train()

        for i in tqdm(range(num_train), desc="Training"):
            item = train_dataset[i]
            text = f"Question: {item['question']}\nAnswer: {item['answer']}"

            # Encode through frozen LLM
            bridge_out = bridge.encode(text)
            hidden = bridge_out.hidden_states  # (seq, llm_dim)

            if hidden.shape[0] < 3:
                continue

            # Project to NARE space
            z = bridge.project_to_nare(hidden)  # (seq, nare_dim)

            # Compute deltas (reasoning trajectories per theory)
            deltas = z[1:] - z[:-1]  # (seq-1, nare_dim)
            if deltas.shape[0] < 2:
                continue

            current_deltas = deltas[:-1]  # input
            next_deltas = deltas[1:]      # target

            sample_energies = []
            for epoch in range(args.epochs):
                optimizer.zero_grad()

                result = field(current_deltas, target=next_deltas, learn=True)
                loss = result.loss.total
                loss.backward()
                optimizer.step()

                sample_energies.append(loss.item())

            record = {
                "sample": i,
                "initial_energy": sample_energies[0],
                "final_energy": sample_energies[-1],
                "routing_load_std": result.metrics["routing_load_std"],
                "temperature": result.metrics["cognitive_temperature"],
                "mode": result.metrics["cognitive_mode"],
            }
            training_history.append(record)

            if (i + 1) % 5 == 0:
                avg_e = sum(r["final_energy"] for r in training_history[-5:]) / 5
                tqdm.write(f"  [{i+1}/{num_train}] avg_energy={avg_e:.4f}  "
                           f"route_std={record['routing_load_std']:.4f}  "
                           f"temp={record['temperature']:.3f} ({record['mode']})")

        # Save checkpoint
        torch.save({
            "field": field.state_dict(),
            "proj_down": bridge.proj_down.state_dict(),
            "proj_up": bridge.proj_up.state_dict(),
            "norm_down": bridge.norm_down.state_dict(),
            "norm_up": bridge.norm_up.state_dict(),
            "config": {
                "nare_dim": args.nare_dim,
                "model_id": args.model,
                "num_experts": args.num_experts,
                "top_k": args.top_k,
            },
            "history": training_history,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    # Training summary
    if training_history:
        init_e = training_history[0]["initial_energy"]
        final_e = training_history[-1]["final_energy"]
        print(f"\nTraining summary: initial_energy={init_e:.4f} -> final_energy={final_e:.4f}")

    # Memory stats
    mem_stats = field.memory.stats()
    print(f"Memory attractors: {int(mem_stats['stored_attractors'])} / {int(mem_stats['capacity'])}")

    # ------------------------------------------------------------------
    # 4. Evaluation Phase
    # ------------------------------------------------------------------
    num_eval = min(args.eval_samples, len(test_dataset))
    print(f"\n--- Phase 2: Evaluation on {num_eval} unseen tasks ---")

    vanilla_correct = 0
    nare_correct = 0
    eval_results: list[dict] = []

    for i in range(num_eval):
        item = test_dataset[i]
        question = item["question"]
        true_answer = extract_gsm8k_answer(item["answer"])
        prompt = f"Question: {question}\nAnswer: Let's solve step by step.\n"

        print(f"\n--- Task {i+1}/{num_eval} ---")
        print(f"Question: {question[:80]}...")
        print(f"True answer: {true_answer}")

        result = generator.generate(prompt, max_new_tokens=args.max_new_tokens)

        # Vanilla
        vanilla_num = extract_number(result.vanilla_answer)
        vanilla_ok = vanilla_num == true_answer
        if vanilla_ok:
            vanilla_correct += 1
        print(f"Vanilla:  {vanilla_num} {'OK' if vanilla_ok else 'WRONG'}")

        # NARE
        nare_num = extract_number(result.nare_answer)
        nare_ok = nare_num == true_answer
        if nare_ok:
            nare_correct += 1
        print(f"NARE:     {nare_num} {'OK' if nare_ok else 'WRONG'}")

        if result.nare_energies:
            print(f"  Final energy: {result.nare_energies[-1]:.4f}  "
                  f"Mode: {result.nare_modes[-1]}  "
                  f"Temp: {result.nare_temperatures[-1]:.3f}")

        eval_results.append({
            "task": i,
            "question": question[:100],
            "true_answer": true_answer,
            "vanilla_answer": vanilla_num,
            "nare_answer": nare_num,
            "vanilla_correct": vanilla_ok,
            "nare_correct": nare_ok,
            "final_energy": result.nare_energies[-1] if result.nare_energies else None,
            "final_mode": result.nare_modes[-1] if result.nare_modes else None,
        })

    # ------------------------------------------------------------------
    # 5. Evidence Checks
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Evidence Checks")
    print("=" * 70)

    checks: list[dict] = []

    # Check 1: Energy trend during training
    if len(training_history) >= 2:
        early = sum(r["final_energy"] for r in training_history[:3]) / min(3, len(training_history))
        late = sum(r["final_energy"] for r in training_history[-3:]) / min(3, len(training_history))
        ratio = late / max(early, 1e-8)
        passed = ratio <= 1.0
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] Energy trend: ratio={ratio:.4f} (early={early:.4f}, late={late:.4f})")
        checks.append({"name": "prediction_energy_trend", "status": status, "ratio": ratio})
    else:
        print("  [SKIP] Energy trend: not enough training data")
        checks.append({"name": "prediction_energy_trend", "status": "SKIP"})

    # Check 2: Routing stability
    if training_history:
        final_std = training_history[-1]["routing_load_std"]
        passed = final_std <= 0.3
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] Routing stability: load_std={final_std:.4f}")
        checks.append({"name": "anthill_routing_stability", "status": status, "load_std": final_std})

    # Check 3: Memory reuse (check retrieval and trace hit rates)
    field.eval()
    if training_history:
        # Run a quick forward to check hit rates
        item = train_dataset[0]
        text = f"Question: {item['question']}\nAnswer: {item['answer']}"
        bridge_out = bridge.encode(text)
        z = bridge.project_to_nare(bridge_out.hidden_states)
        with torch.no_grad():
            test_result = field(z[:8], learn=False)
        r_hit = test_result.metrics.get("retrieval_hit_rate", 0)
        t_hit = test_result.metrics.get("trace_hit_rate", 0)
        combined = (r_hit + t_hit) / 2
        passed = combined > 0.3
        status = "PASS" if passed else ("WARN" if combined > 0.1 else "FAIL")
        print(f"  [{status}] Memory reuse: combined={combined:.3f} "
              f"(retrieval={r_hit:.2f}, trace={t_hit:.2f})")
        checks.append({"name": "memory_reuse", "status": status, "combined": combined})

    # Check 4: SubQ attention complexity
    if training_history:
        n = 8
        complexity = test_result.metrics.get("attention_complexity_estimate", n * n)
        ratio = complexity / (n * n)
        passed = ratio < 0.75
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] SubQ attention: ratio={ratio:.4f} (complexity={complexity}, n^2={n*n})")
        checks.append({"name": "subquadratic_attention", "status": status, "ratio": ratio})

    # Check 5: NARE vs Vanilla accuracy
    if num_eval > 0:
        diff = nare_correct - vanilla_correct
        status = "PASS" if diff >= 0 else "WARN"
        print(f"  [{status}] Accuracy: NARE={nare_correct}/{num_eval} vs Vanilla={vanilla_correct}/{num_eval} (delta={diff:+d})")
        checks.append({"name": "nare_vs_vanilla", "status": status, "nare": nare_correct, "vanilla": vanilla_correct, "delta": diff})

    passed_count = sum(1 for c in checks if c["status"] == "PASS")
    total_checks = len(checks)

    # ------------------------------------------------------------------
    # 6. Final Results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"  Vanilla {args.model}:         {vanilla_correct}/{num_eval} ({vanilla_correct/max(num_eval,1)*100:.0f}%)")
    print(f"  {args.model} + NARE-Field:  {nare_correct}/{num_eval} ({nare_correct/max(num_eval,1)*100:.0f}%)")
    print(f"  Evidence checks:              {passed_count}/{total_checks}")

    diff = nare_correct - vanilla_correct
    if diff > 0:
        print(f"\n  NARE-Field IMPROVED accuracy by +{diff} tasks")
    elif diff == 0:
        print("\n  NARE-Field maintained baseline accuracy")
    else:
        print(f"\n  NARE-Field underperformed by {diff} tasks (needs more training)")

    # Parameter efficiency
    field_params = sum(p.numel() for p in field.parameters())
    bridge_proj_params = sum(p.numel() for p in bridge.proj_down.parameters())
    bridge_proj_params += sum(p.numel() for p in bridge.proj_up.parameters())
    bridge_proj_params += sum(p.numel() for p in bridge.norm_down.parameters())
    bridge_proj_params += sum(p.numel() for p in bridge.norm_up.parameters())
    llm_params = sum(p.numel() for p in bridge._llm.parameters())

    print(f"\n  LLM parameters:    {llm_params:,} (frozen)")
    print(f"  Bridge projections: {bridge_proj_params:,} (trainable)")
    print(f"  NARE field:         {field_params:,} (trainable)")
    print(f"  Total trainable:    {field_params + bridge_proj_params:,}")
    print(f"  Ratio:              {(field_params + bridge_proj_params) / max(llm_params, 1) * 100:.2f}% of LLM")

    # ------------------------------------------------------------------
    # 7. Save Report
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    report = {
        "model": args.model,
        "nare_dim": args.nare_dim,
        "train_samples": args.train_samples,
        "eval_samples": num_eval,
        "training_history": training_history,
        "evaluation": eval_results,
        "evidence_checks": checks,
        "accuracy": {
            "vanilla": vanilla_correct,
            "nare": nare_correct,
            "total": num_eval,
        },
        "parameters": {
            "llm_frozen": llm_params,
            "bridge_trainable": bridge_proj_params,
            "field_trainable": field_params,
            "total_trainable": field_params + bridge_proj_params,
        },
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to {args.report}")


if __name__ == "__main__":
    main()
