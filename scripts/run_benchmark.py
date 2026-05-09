from __future__ import annotations

import argparse
from pathlib import Path

from narefield import BenchmarkRunConfig, BenchmarkRunner, BenchmarkSuite, NAREConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark-ready NARE-Field experiments.")
    parser.add_argument("--jsonl", type=Path, default=None, help="Optional JSONL suite with prompt/question and target/answer fields.")
    parser.add_argument("--suite-name", default=None, help="Override benchmark suite name.")
    parser.add_argument("--synthetic-count", type=int, default=24, help="Number of synthetic reasoning tasks when --jsonl is omitted.")
    parser.add_argument("--dim", type=int, default=16, help="Latent dimension.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs over the suite.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--experts", type=int, default=4, help="Anthill expert count.")
    parser.add_argument("--top-k", type=int, default=2, help="Top-k experts per task.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic model seed.")
    parser.add_argument("--ablation", action="store_true", help="Run default ablation matrix.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jsonl is None:
        suite = BenchmarkSuite.synthetic_reasoning(count=args.synthetic_count)
    else:
        suite = BenchmarkSuite.from_jsonl(args.jsonl, name=args.suite_name)

    model_config = NAREConfig(dim=args.dim, num_experts=args.experts, top_k=args.top_k, seed=args.seed)
    run_config = BenchmarkRunConfig(epochs=args.epochs, batch_size=args.batch_size)
    runner = BenchmarkRunner(model_config=model_config, run_config=run_config)
    report = runner.run_ablation_matrix(suite) if args.ablation else runner.run(suite)
    payload = report.to_json()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
