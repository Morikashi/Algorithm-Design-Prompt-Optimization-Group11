# Phase 2 Plan — Prompt Optimization (QA + Summarization)

## Objectives (Phase 2 rubric)
- Complete implementation (clean, modular, testable)
- ≥10 real-world inputs per task (QA + summarization)
- Simple/medium/hard inputs + failure case analysis
- Empirical analysis: runtime + memory vs theoretical complexity
- Method comparison: hill climbing vs BFS vs beam (+ optional simulated annealing / GA)
- Final report + system diagram (classic algorithms + LLM modules + data flow)

## Locked Decisions
- Tasks: QA + Summarization only
- Metrics: Exact Match (QA), ROUGE-L (Summ)
- LLM backend: Ollama for experiments; MockLLM for unit tests only
- Fair comparisons: same dataset, same model, same temperature=0, same budget B

## Deliverables
- `scripts/benchmark_phase2.py` generates:
  - `results/bench_summary.csv` (ignored by git)
  - trace CSVs for convergence plots
- ≥10 inputs per task in `src/utils/datasets_phase2.py` (tagged by difficulty)
- Failure-case logs for worst examples per method

## Experiment Protocol
For each task and each method:
- budgets: B ∈ {40, 80, 120} prompt evals
- neighbor cap: max_neighbors ∈ {10, 25}
- record:
  - best score
  - eval_count
  - runtime seconds
  - peak memory MB
  - best prompt text (saved separately, not committed)

## Future Improvements (if time)
- simulated annealing
- genetic algorithm
- embedding-based novelty pruning
- multi-objective score: quality - λ * prompt_cost
