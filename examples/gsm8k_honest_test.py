"""Honest A/B test: vanilla Qwen2-1.5B vs Qwen2-1.5B + NARE-Field on GSM8K."""
from __future__ import annotations

import re

from datasets import load_dataset

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
    print("HONEST GSM8K TEST: Vanilla Qwen2 vs Qwen2 + NARE-Field")
    print("=" * 60)

    bridge = NAREBridge("Qwen/Qwen2-1.5B-Instruct")
    field = NAREFieldModel(NAREConfig(
        dim=bridge.dim,
        memory_capacity=512,
        num_experts=8,
        top_k=2,
    ))
    generator = NAREGenerator(bridge, field, alpha=0.0)

    print("\nLoading GSM8K test set...")
    dataset = load_dataset("gsm8k", "main", split="test")

    num_samples = 20
    vanilla_correct = 0
    nare_correct = 0

    for i in range(min(num_samples, len(dataset))):
        item = dataset[i]
        question = item["question"]
        true_answer = extract_gsm8k_answer(item["answer"])

        prompt = f"Question: {question}\nAnswer: Let's solve step by step.\n"

        print(f"\n--- Task {i+1}/{num_samples} ---")
        print(f"Question: {question[:80]}...")
        print(f"True answer: {true_answer}")

        result = generator.generate(prompt, max_new_tokens=200)

        vanilla_num = extract_number(result.vanilla_answer)
        nare_num = extract_number(result.nare_answer)

        v_ok = vanilla_num == true_answer if vanilla_num else False
        n_ok = nare_num == true_answer if nare_num else False

        if v_ok:
            vanilla_correct += 1
        if n_ok:
            nare_correct += 1

        print(f"Vanilla: {result.vanilla_answer[:100]}...")
        print(f"  -> extracted: {vanilla_num} {'OK' if v_ok else 'WRONG'}")
        print(f"NARE:    {result.nare_answer[:100]}...")
        print(f"  -> extracted: {nare_num} {'OK' if n_ok else 'WRONG'}")
        print(f"  -> energy trend: {result.nare_energies[:5]}")
        print(f"  -> modes: {result.nare_modes[:5]}")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Vanilla Qwen2-1.5B:        {vanilla_correct}/{num_samples} ({vanilla_correct/num_samples:.0%})")
    print(f"Qwen2-1.5B + NARE-Field:   {nare_correct}/{num_samples} ({nare_correct/num_samples:.0%})")
    delta = nare_correct - vanilla_correct
    if delta > 0:
        print(f"\nNARE-Field IMPROVED accuracy by +{delta} tasks.")
    elif delta == 0:
        print(f"\nNo difference. NARE-Field did not help or hurt.")
    else:
        print(f"\nNARE-Field DEGRADED accuracy by {delta} tasks. Alpha tuning needed.")


if __name__ == "__main__":
    main()
