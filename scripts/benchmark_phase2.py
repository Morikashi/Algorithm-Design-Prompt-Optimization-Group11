from __future__ import annotations

import argparse
import csv
import os
import time
import tracemalloc
import uuid

from src.prompt.representation import Prompt
from src.evaluation.ollama_llm import OllamaLLM
from src.evaluation.metrics import ExactMatchMetric, RougeLMetric
from src.evaluation.evaluator import Evaluator
from src.evaluation.llm_interface import LLMConfig
from src.utils.trace import TraceLogger

from src.algorithms.hill_climbing import hill_climb, HillClimbConfig
from src.algorithms.bfs import bfs_search, BFSConfig
from src.algorithms.beam_search import beam_search, BeamConfig
from src.algorithms.heuristics import HeuristicConfig

from src.utils.datasets_phase2 import QA_PHASE2, SUM_PHASE2

from src.analysis.failure_cases import save_worst_examples # this line wasadded/edited in my second commit
from src.algorithms.simulated_annealing import simulated_annealing, SAConfig


def ensure_results_dir():
    os.makedirs("results", exist_ok=True)


def start_prompt(task: str) -> Prompt:
    if task == "qa":
        return Prompt(task="qa", instruction="Answer the question.", constraints=[], output_format="plain")
    return Prompt(task="summarization", instruction="Summarize the text.", constraints=[], output_format="plain")


def run_one(task: str, algo: str, budget: int, max_neighbors: int, max_depth: int, beam_width: int, heuristics: bool):
    run_id = f"p2-{task}-{algo}-{uuid.uuid4().hex[:8]}"

    dataset = QA_PHASE2 if task == "qa" else SUM_PHASE2
    metric = ExactMatchMetric() if task == "qa" else RougeLMetric()

    llm = OllamaLLM()
    llm_cfg = LLMConfig(temperature=0.0, timeout_s=180.0)

    evaluator = Evaluator(llm=llm, metric=metric, lambda_cost=0.0005, llm_config=llm_cfg)

    hcfg = HeuristicConfig(
    enabled=heuristics,
    novelty_threshold=0.92,
    max_prompt_words=250,
    mode=args.heur_mode,
    embedding_cosine_threshold=args.emb_cos,
)


    trace = TraceLogger(out_dir="results", filename=f"trace_{run_id}.csv")

    tracemalloc.start()
    t0 = time.perf_counter()

    if algo == "hill":
        res = hill_climb(
            start=start_prompt(task),
            evaluator=evaluator,
            dataset=dataset,
            config=HillClimbConfig(max_steps=30, max_prompt_evals=budget, max_neighbors=max_neighbors),
            trace_logger=trace,
            run_id=run_id,
        )
        best_score = res.best_eval.final_score
        eval_count = res.eval_count
        worst_path = save_worst_examples(res.best_eval, out_dir="results", run_id=run_id, k=5) # added worst_path in my second commit

    elif algo == "sa":
        res = simulated_annealing(
            start=start_prompt(task),
            evaluator=evaluator,
            dataset=dataset,
            config=SAConfig(
                max_steps=200,
                max_prompt_evals=budget,
                max_neighbors=max_neighbors,
                t_start=1.0,
                t_end=0.05,
                schedule="exp",
                seed=42,
            ),
            trace_logger=trace,
            run_id=run_id,
        )
        best_score = res.best_eval.final_score
        eval_count = res.eval_count
        worst_path = save_worst_examples(res.best_eval, out_dir="results", run_id=run_id, k=5)

    elif algo == "bfs":
        res = bfs_search(
            start=start_prompt(task),
            evaluator=evaluator,
            dataset=dataset,
            config=BFSConfig(
                max_depth=max_depth,
                max_prompt_evals=budget,
                max_neighbors=max_neighbors,
                heuristic_cfg=hcfg,
            ),
            trace_logger=trace,
            run_id=run_id,
        )
        best_score = res.best_eval.final_score
        eval_count = res.eval_count
        worst_path = save_worst_examples(res.best_eval, out_dir="results", run_id=run_id, k=5) # added worst_path in my second commit





    else:
        res = beam_search(
            start=start_prompt(task),
            evaluator=evaluator,
            dataset=dataset,
            config=BeamConfig(
                max_depth=max_depth,
                beam_width=beam_width,
                max_prompt_evals=budget,
                max_neighbors=max_neighbors,
                heuristic_cfg=hcfg,
            ),
            trace_logger=trace,
            run_id=run_id,
        )
        best_score = res.best_eval.final_score
        eval_count = res.eval_count
        worst_path = save_worst_examples(res.best_eval, out_dir="results", run_id=run_id, k=5) # added worst_path in my second commit

    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)

    return {
        "run_id": run_id,
        "task": task,
        "algo": algo,
        "heuristics": int(heuristics),
        "budget": budget,
        "max_neighbors": max_neighbors,
        "max_depth": max_depth,
        "beam_width": beam_width,
        "eval_count": eval_count,
        "best_score": best_score,
        "runtime_s": elapsed,
        "peak_mem_mb": peak_mb,
        "trace_path": trace.path,
        "worst_path": worst_path,
    }


def write_summary(rows, out_path="results/bench_summary.csv"):
    ensure_results_dir()
    header = [
        "run_id", "task", "algo", "heuristics",
        "budget", "max_neighbors", "max_depth", "beam_width",
        "eval_count", "best_score", "runtime_s", "peak_mem_mb", "trace_path", "worst_path"
    ] # added worst_path in my second commit
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("Wrote:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["qa", "summarization", "both"], default="both")
    ap.add_argument("--budgets", default="40,80,120")
    ap.add_argument("--max_neighbors", type=int, default=15)
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--beam_width", type=int, default=5)
    ap.add_argument("--heuristics", action="store_true")
    ap.add_argument("--heur_mode", choices=["jaccard", "embedding"], default="jaccard")
    ap.add_argument("--emb_cos", type=float, default=0.90)
    args = ap.parse_args()

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]

    tasks = ["qa", "summarization"] if args.task == "both" else [args.task]
    algos = ["hill", "bfs", "beam", "sa"]


    rows = []
    for t in tasks:
        for b in budgets:
            for a in algos:
                rows.append(
                    run_one(
                        task=t,
                        algo=a,
                        budget=b,
                        max_neighbors=args.max_neighbors,
                        max_depth=args.max_depth,
                        beam_width=args.beam_width,
                        heuristics=args.heuristics,
                    )
                )

    write_summary(rows)


if __name__ == "__main__":
    main()
