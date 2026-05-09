"""Train NARE-Field on GSM8K reasoning trajectories, then evaluate."""
from __future__ import annotations

import re
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from narefield.bridge import NAREBridge
from narefield.generate import NAREGenerator
from narefield.model import NAREConfig, NAREFieldModel


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


def main() -> None:
    print("=" * 60)
    print("AMORTIZED REASONING: Training NARE-Field on GSM8K")
    print("=" * 60)

    bridge = NAREBridge("Qwen/Qwen2-1.5B-Instruct")
    # Using larger capacity for training
    field = NAREFieldModel(NAREConfig(
        dim=bridge.dim,
        memory_capacity=1024,
        trace_capacity=512,
        num_experts=8,
        top_k=2,
    ))
    
    # We will use alpha=0.15 since the field will actually know what it's doing
    generator = NAREGenerator(bridge, field, alpha=0.15)

    print("\nLoading GSM8K datasets...")
    train_dataset = load_dataset("gsm8k", "main", split="train")
    test_dataset = load_dataset("gsm8k", "main", split="test")

    import pickle
    import os
    
    checkpoint_path = "trained_nare_field.pkl"
    
    # 1. TRAINING PHASE
    if os.path.exists(checkpoint_path):
        print(f"\n--- Phase 1: Loading trained NARE-Field from {checkpoint_path} ---")
        with open(checkpoint_path, "rb") as f:
            saved_state = pickle.load(f)
            field.memory.weights = saved_state["memory_weights"]
            field.memory.attractors = saved_state["memory_attractors"]
            field.trace_memory.traces = saved_state["traces"]
            # Fast-forwarding
    else:
        train_samples = 20
        print(f"\n--- Phase 1: Training NARE on {train_samples} reasoning trajectories ---")
        
        total_energy = []
        
        for i in tqdm(range(train_samples), desc="Training"):
            item = train_dataset[i]
            text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            
            inputs = bridge.tokenizer(text, return_tensors="pt").to(bridge.device)
            with torch.no_grad():
                outputs = bridge.model.model(**inputs)
                hidden = outputs.last_hidden_state[0]  # (seq_len, dim)
            
            h = hidden.to(torch.float32).cpu().numpy()
            
            # Compute deltas to isolate logic from factual state
            deltas = h[1:] - h[:-1]
            if deltas.shape[0] < 2:
                continue
                
            current_deltas = deltas[:-1]
            next_deltas = deltas[1:]
            
            result = field.step(current_deltas, target=next_deltas, learn=True)
            total_energy.append(result.loss.total)

        print(f"Training complete. Initial energy: {total_energy[0]:.2f} -> Final energy: {total_energy[-1]:.2f}")
        
        # Save state
        with open(checkpoint_path, "wb") as f:
            pickle.dump({
                "memory_weights": field.memory.weights,
                "memory_attractors": field.memory.attractors,
                "traces": field.trace_memory.traces
            }, f)
        print(f"Saved trained field to {checkpoint_path}")
    
    stats = field.memory.stats()
    print(f"Memory Attractors built: {int(stats['stored_attractors'])} / {int(stats['capacity'])}")

    # 2. EVALUATION PHASE
    num_eval = 10
    print(f"\n--- Phase 2: Evaluation on {num_eval} unseen tasks ---")
    
    vanilla_correct = 0
    nare_correct = 0

    for i in range(num_eval):
        item = test_dataset[i]
        question = item["question"]
        true_answer = extract_gsm8k_answer(item["answer"])

        prompt = f"Question: {question}\nAnswer: Let's solve step by step.\n"

        print(f"\n--- Task {i+1}/{num_eval} ---")
        print(f"Question: {question[:80]}...")
        print(f"True answer: {true_answer}")

        result = generator.generate(prompt, max_new_tokens=200)

        # Vanilla validation
        vanilla_num = extract_number(result.vanilla_answer)
        if vanilla_num == true_answer:
            print(f"Vanilla:  {vanilla_num} \033[92mOK\033[0m")
            vanilla_correct += 1
        else:
            print(f"Vanilla:  {vanilla_num} \033[91mWRONG\033[0m")

        # NARE validation
        nare_num = extract_number(result.nare_answer)
        if nare_num == true_answer:
            print(f"NARE:     {nare_num} \033[92mOK\033[0m")
            nare_correct += 1
        else:
            print(f"NARE:     {nare_num} \033[91mWRONG\033[0m")
            
        print(f"  -> NARE Final Energy: {result.nare_energies[-1]:.2f}")

    print("\n" + "=" * 60)
    print("FINAL RESULTS (TRAINED FIELD)")
    print("=" * 60)
    print(f"Vanilla Qwen2-1.5B:        {vanilla_correct}/{num_eval} ({vanilla_correct/num_eval*100:.0f}%)")
    print(f"Qwen2-1.5B + NARE-Field:   {nare_correct}/{num_eval} ({nare_correct/num_eval*100:.0f}%)")

    diff = nare_correct - vanilla_correct
    if diff > 0:
        print(f"\nSUCCESS: NARE-Field IMPROVED accuracy by +{diff} tasks! 🚀")
    elif diff == 0:
        print("\nNEUTRAL: NARE-Field maintained baseline accuracy.")
    else:
        print(f"\nFAIL: NARE-Field DEGRADED accuracy by {diff} tasks. Needs more training data.")


if __name__ == "__main__":
    main()
