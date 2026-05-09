from __future__ import annotations

import numpy as np

from narefield import FreeEnergyTrainer, NAREConfig, NAREFieldModel, compare_ablation, summarize_validation


def run_model(model: NAREFieldModel, batches: list[np.ndarray]) -> tuple[list[dict], list[dict]]:
    trainer = FreeEnergyTrainer(model)
    records = []
    updates = []
    for batch in batches:
        result, update = trainer.train_step(batch)
        records.append(model.validation_record(result))
        updates.append(update.as_dict())
    return records, updates


def main() -> None:
    rng = np.random.default_rng(42)
    batches = [rng.normal(size=(6, 8)) for _ in range(12)]
    full_model = NAREFieldModel(NAREConfig(dim=8, memory_capacity=32, num_experts=4, top_k=2, seed=42))
    ablated_model = NAREFieldModel(
        NAREConfig(
            dim=8,
            memory_capacity=32,
            num_experts=4,
            top_k=2,
            seed=42,
            attention_mix=0.0,
            retrieval_mix=0.0,
            trace_mix=0.0,
        )
    )

    full_records, updates = run_model(full_model, batches)
    ablated_records, _ = run_model(ablated_model, batches)
    report = summarize_validation(full_records)
    ablated_report = summarize_validation(ablated_records)
    ablation = compare_ablation(report, ablated_report, "no_attention_retrieval_trace")

    print("NARE-Field validation report")
    for key, value in report.as_dict().items():
        print(f"{key}: {value}")
    print("\nLast free-energy update")
    for key, value in updates[-1].items():
        print(f"{key}: {value}")
    print("\nAblation delta")
    for key, value in ablation.as_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
