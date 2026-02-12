from __future__ import annotations

import argparse
import uuid

from src.prompt.representation import Prompt
from src.evaluation.mock_llm import MockLLM
from src.evaluation.ollama_llm import OllamaLLM
from src.evaluation.metrics import ExactMatchMetric, RougeLMetric
from src.evaluation.evaluator import Evaluator
from src.evaluation.llm_interface import LLMConfig
from src.utils.trace import TraceLogger
from src.utils.datasets import QA_DATASET, SUMMARIZATION_DATASET
from src.utils.datasets_phase2 import QA_PHASE2, SUM_PHASE2
from src.algorithms.simulated_annealing import simulated_annealing, SAConfig

from src.algorithms.hill_climbing import hill_climb, HillClimbConfig
from src.algorithms.bfs import bfs_search, BFSConfig
from src.algorithms.beam_search import beam_search, BeamConfig
from src.algorithms.heuristics import HeuristicConfig


def make_start_prompt(task: str) -> Prompt:
    if task == "qa":
        return Prompt(task="qa", instruction="Answer the question.", constraints=[], output_format="plain")
    return Prompt(task="summarization", instruction="Summarize the text.", constraints=[], output_format="plain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["qa", "summarization"], required=True)
    ap.add_argument("--backend", choices=["mock", "ollama"], default="mock")
    ap.add_argument("--algo", choices=["hill", "bfs", "beam", "sa"], required=True)


    ap.add_argument("--budget", type=int, default=80)
    ap.add_argument("--max_neighbors", type=int, default=25)

    # depth/beam/steps
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--beam_width", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=20)

    # heuristics
    ap.add_argument("--heuristics", action="store_true")
    ap.add_argument("--novelty", type=float, default=0.92)
    ap.add_argument("--max_prompt_words", type=int, default=250)

    ap.add_argument("--heur_mode", choices=["jaccard", "embedding"], default="jaccard")
    ap.add_argument("--emb_cos", type=float, default=0.90)

    ap.add_argument("--sa_steps", type=int, default=200)
    ap.add_argument("--t_start", type=float, default=1.0)
    ap.add_argument("--t_end", type=float, default=0.05)
    ap.add_argument("--schedule", choices=["exp", "linear"], default="exp")
    ap.add_argument("--seed", type=int, default=42)

    
    # simulated annealing
    

    args = ap.parse_args()

    run_id = f"{args.task}-{args.backend}-{args.algo}-{uuid.uuid4().hex[:8]}"

    # Dataset + metric
    if args.task == "qa":
        dataset = QA_DATASET
        metric = ExactMatchMetric()
    else:
        dataset = SUMMARIZATION_DATASET
        metric = RougeLMetric()

    # Backend
    if args.backend == "mock":
        llm = MockLLM()
        cfg = LLMConfig(temperature=0.0)
    else:
        llm = OllamaLLM()
        cfg = LLMConfig(temperature=0.0, timeout_s=120.0)

    evaluator = Evaluator(llm=llm, metric=metric, lambda_cost=0.0005, llm_config=cfg)
    start = make_start_prompt(args.task)

    hcfg = HeuristicConfig(
    enabled=args.heuristics,
    novelty_threshold=args.novelty,
    max_prompt_words=args.max_prompt_words,
    mode=args.heur_mode,
    embedding_cosine_threshold=args.emb_cos,
    ) # NEWLY ADDED


    trace = TraceLogger(out_dir="results", filename=f"trace_{run_id}.csv")

    if args.algo == "hill":
        res = hill_climb(
            start=start,
            evaluator=evaluator,
            dataset=dataset,
            config=HillClimbConfig(
                max_steps=args.max_steps,
                max_prompt_evals=args.budget,
                max_neighbors=args.max_neighbors,
                require_strict_improvement=True,
            ),
            trace_logger=trace,
            run_id=run_id,
        )
        best_prompt, best_eval, eval_count = res.best_prompt, res.best_eval, res.eval_count

    elif args.algo == "bfs":
        res = bfs_search(
            start=start,
            evaluator=evaluator,
            dataset=dataset,
            config=BFSConfig(
                max_depth=args.max_depth,
                max_prompt_evals=args.budget,
                max_neighbors=args.max_neighbors,
                heuristic_cfg=hcfg,
            ),
            trace_logger=trace,
            run_id=run_id,
        )
        best_prompt, best_eval, eval_count = res.best_prompt, res.best_eval, res.eval_count

    elif args.algo == "sa":
        res = simulated_annealing(
            start=start,
            evaluator=evaluator,
            dataset=dataset,
            config=SAConfig(
                max_steps=args.sa_steps,
                max_prompt_evals=args.budget,
                max_neighbors=args.max_neighbors,
                t_start=args.t_start,
                t_end=args.t_end,
                schedule=args.schedule,
                seed=args.seed,
            ),
            trace_logger=trace,
            run_id=run_id,
        )
        best_prompt, best_eval, eval_count = res.best_prompt, res.best_eval, res.eval_count

    
    else:
        res = beam_search(
            start=start,
            evaluator=evaluator,
            dataset=dataset,
            config=BeamConfig(
                max_depth=args.max_depth,
                beam_width=args.beam_width,
                max_prompt_evals=args.budget,
                max_neighbors=args.max_neighbors,
                heuristic_cfg=hcfg,
            ),
            trace_logger=trace,
            run_id=run_id,
        )
        best_prompt, best_eval, eval_count = res.best_prompt, res.best_eval, res.eval_count

    print("\n=== Search Result ===")
    print("run_id:", run_id)
    print("algo:", args.algo, "task:", args.task, "backend:", args.backend)
    print("eval_count:", eval_count)
    print("best_score:", best_eval.final_score)
    print("best_prompt_id:", best_eval.prompt_id)
    print("trace_path:", trace.path)
    print("\n--- Best Prompt ---")
    print(best_prompt.render())


if __name__ == "__main__":
    main()
