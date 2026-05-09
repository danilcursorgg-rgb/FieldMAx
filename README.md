# NARELabs2: NARE-Field

Операциональная реализация NARE-Field: непараметрических адаптивных реконструктивных энергетических полей.

Этот прототип переводит теорию в вычислимый pipeline: модель минимизирует prediction/free-energy, хранит не весь контент, а инварианты и латентные следы, реконструирует знания через memory-field и внешний retrieval, а Anthill/MoE-роутинг стабилизируется load-balancing loss.

## Что реализовано

- **Prediction/free-energy objective**: `MSE(x, x_hat)` + штрафы за когнитивный инвариант, ёмкость памяти и routing collapse.
- **Memory field**: Hopfield/energy-style поле аттракторов с predictive inference и Hebbian-записью `W <- decay * W + eta * dz dz^T`.
- **SubQ field attention**: bucketed potential attention с оценкой `O(n log n + b^2)` вместо полного `O(n^2)` pairwise attention.
- **Latent trace memory**: хранение траекторий рассуждения `z0 -> z* -> prediction`, а не текстовых chain-of-thought.
- **Non-parametric retrieval memory**: key-value внешний индекс для принципа `context > store`.
- **Anthill routing**: top-k softmax Mixture-of-Experts с KL load-balancing и локальным router update.
- **Cognitive temperature**: контроллер exploration/exploitation от энергии и энтропии состояния.
- **FreeEnergyTrainer**: единая процедура локального update: model step, memory write, trace/retrieval write, router balancing.
- **Validation + ablation**: метрики качества, памяти, routing stability, active parameter ratio, SubQ complexity и ablation delta.

## Установка

```bash
python -m pip install -e '.[dev]'
```

## Быстрый запуск

```bash
python examples/demo.py
```

Демо запускает полную модель и ablation-вариант без attention/retrieval/trace, затем печатает validation report, последний free-energy update и delta по ablation.

## Benchmark-ready запуск

Прототип подготовлен к переходу от теории к бенчмаркам: есть синтетический reasoning-suite, JSONL-загрузчик, encoder задач в латентное пространство, runner, ablation matrix и JSON-отчёты.

```bash
python scripts/run_benchmark.py --synthetic-count 24 --epochs 3 --batch-size 4
python scripts/run_benchmark.py --ablation --output reports/synthetic_ablation.json
```

## Evidence-runner: честная проверка теории

`run_evidence.py` не «доказывает» теорию словами. Он запускает фальсифицируемые проверки и возвращает `pass`/`warn`/`fail`:

```bash
python scripts/run_evidence.py --synthetic-count 24 --epochs 4 --batch-size 4
```

Проверяются пять инженерных гипотез:

1. падает ли prediction/reconstruction energy при повторном обучении;
2. выигрывает ли полная архитектура у memory-only ablation;
3. появляются ли реальные hit-rate у non-parametric retrieval и trace replay;
4. не схлопывается ли Anthill/MoE routing;
5. остаётся ли SubQ attention дешевле dense O(n²) оценки на длинных синтетических входах.

Если гипотеза не выполняется, отчёт покажет `fail` или `warn`; это специально сделано, чтобы не прятать слабые места теории.

Для будущих GSM8K/MATH-подобных наборов используйте JSONL-строки с полями `prompt`/`question` и `target`/`answer`:

```bash
python scripts/run_benchmark.py --jsonl data/gsm8k_like.jsonl --suite-name gsm8k_like --ablation
```

Важно: эти скрипты пока готовят воспроизводимую инфраструктуру измерений; они не заявляют реальное качество на GSM8K/MATH без запуска на настоящих датасетах.

## Архитектура как рабочая система

`NARE_FIELD_ARCHITECTURE` описывает pipeline теории как исполняемую спецификацию: SubQ attention → Anthill routing → memory-field inference → retrieval → trace replay → cognitive temperature → free-energy update.

Но главное теперь не описание, а instrumentation: каждый `NAREFieldModel.step()` возвращает `pipeline_trace` с energy ledger. Это позволяет проверить, из каких членов реально собран loss и какие стадии дали сигнал для evidence/benchmark отчёта.

## Тесты

```bash
python -m pytest
python -m compileall src tests examples
```

## Минимальный пример

```python
import numpy as np
from narefield import FreeEnergyTrainer, NAREConfig, NAREFieldModel

model = NAREFieldModel(NAREConfig(dim=8, memory_capacity=16, num_experts=4, top_k=2, seed=7))
trainer = FreeEnergyTrainer(model)
x = np.random.default_rng(7).normal(size=(4, 8))
result, update = trainer.train_step(x)

print(result.loss.total)
print(result.metrics["cognitive_temperature"])
print(update.as_dict())
```

## Формальная цель оптимизации

Для входа `x`, реконструкции `x_hat`, когнитивного инварианта `c`, состояния памяти `m` и распределения роутинга `p` используется:

```text
L = MSE(x, x_hat)
  + lambda_invariant * ||c||^2
  + lambda_memory * capacity_usage(m)
  + lambda_routing * KL(load(p) || UniformExperts)
```

## Операционный цикл NARE-Field

```text
x
 -> SubQFieldAttention: выделить области высокого когнитивного потенциала
 -> AnthillRouter: top-k эксперты дают реконструктивный прогноз
 -> MemoryField.infer: найти/уточнить ближайший энергетический аттрактор
 -> NonParametricMemory.retrieve: достать внешний контекст
 -> TraceMemory.replay: восстановить латентный маршрут рассуждения
 -> blend predictions: x_hat
 -> PredictionEnergyLoss: free-energy L
 -> learn: Hebbian memory write + trace write + retrieval write + router load-balance update
 -> CognitiveTemperature: выбрать exploit/explore режим
```

## Соответствие теории

| Теоретический блок | Код |
| --- | --- |
| Предсказательная энергия | `PredictionEnergyLoss` |
| Когнитивный инвариант | `CognitiveInvariant` |
| Память как поле аттракторов | `MemoryField` |
| Латентная память рассуждений | `TraceMemory` |
| Непараметрическое знание | `NonParametricMemory` |
| Субквадратичное внимание | `SubQFieldAttention` |
| Anthill/MoE | `AnthillRouter`, `AnthillLayer` |
| Routing collapse prevention | `load_balance_loss`, `FreeEnergyTrainer` router update |
| Когнитивная температура | `CognitiveTemperature` |
| Экспериментальная валидация | `summarize_validation`, `compare_ablation` |
| Benchmark orchestration | `BenchmarkSuite`, `BenchmarkRunner`, `scripts/run_benchmark.py` |
| Архитектура как проверяемая спецификация | `NARE_FIELD_ARCHITECTURE`, `theory_alignment_table` |
| Pipeline instrumentation | `NAREPipelineTrace`, `EnergyLedgerTerm` |
| Evidence checks | `EvidenceRunner`, `scripts/run_evidence.py` |

Это исследовательский NumPy-прототип: он не претендует на SOTA-качество LLM, но реализует формальные правила оптимизации, памяти, роутинга и проверки, нужные для превращения NARE-Field из метафоры в исполняемую архитектуру.

Критическая инженерная позиция: теория считается рабочей только там, где evidence-runner показывает устойчивые метрики. Всё остальное остаётся гипотезой до реального GSM8K/MATH/RAG-сравнения с baseline transformer/LLM.
