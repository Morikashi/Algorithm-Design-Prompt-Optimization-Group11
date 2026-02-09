from __future__ import annotations

import argparse
import uuid

from src.prompt.representation import Prompt
from src.evaluation.mock_llm import MockLLM
from src.evaluation.ollama_llm import OllamaLLM
from src.evaluation.metrics import ExactMatchMetric, RougeLMetric
from src.evaluation.evaluator import Evaluator
from src.evaluation.llm_interface import LLMConfig
from src.algorithms.hill_climbing import hill_climb, HillClimbConfig
from src.utils.trace import TraceLogger
from src.utils.datasets import QA_DATASET, SUMMARIZATION_DATASET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["qa", "summarization"], required=True)
    ap.add_argument("--backend", choices=["mock", "ollama"], default="mock")
    ap.add_argument("--max_steps", type=int, default=20)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--max_neighbors", type=int, default=25)
    args = ap.parse_args()

    run_id = f"{args.task}-{args.backend}-{uuid.uuid4().hex[:8]}"

    # Choose dataset + metric
    if args.task == "qa":
        dataset = QA_DATASET
        metric = ExactMatchMetric()
        start = Prompt(
            task="qa",
            instruction="Answer the question.",
            constraints=[],
            output_format="plain",
            style=None,
            verification=False,
        )
    else:
        dataset = SUMMARIZATION_DATASET
        metric = RougeLMetric()
        start = Prompt(
            task="summarization",
            instruction="Summarize the text.",
            constraints=[],
            output_format="plain",
            style=None,
            verification=False,
        )

    # Choose backend
    if args.backend == "mock":
        llm = MockLLM()
        llm_cfg = LLMConfig(temperature=0.0)
    else:
        llm = OllamaLLM()
        llm_cfg = LLMConfig(temperature=0.0, timeout_s=120.0)

    evaluator = Evaluator(llm=llm, metric=metric, lambda_cost=0.0, llm_config=llm_cfg)

    trace = TraceLogger(out_dir="results", filename=f"trace_{run_id}.csv")
    cfg = HillClimbConfig(
        max_steps=args.max_steps,
        max_prompt_evals=args.budget,
        max_neighbors=args.max_neighbors,
        require_strict_improvement=True,
    )

    res = hill_climb(
        start=start,
        evaluator=evaluator,
        dataset=dataset,
        config=cfg,
        trace_logger=trace,
        run_id=run_id,
    )

    print("\n=== Hill Climbing Result ===")
    print("run_id:", run_id)
    print("eval_count:", res.eval_count)
    print("steps_taken:", res.steps_taken)
    print("best_score:", res.best_eval.final_score)
    print("best_prompt_id:", res.best_eval.prompt_id)
    print("trace_path:", res.trace_path)
    print("\n--- Best Prompt ---")
    print(res.best_prompt.render())

    # Show a couple predictions for sanity
    print("\n--- Sample Outputs ---")
    for ex in dataset[:2]:
        pred = evaluator.llm.generate(res.best_prompt.render(), ex["input"], llm_cfg)
        print("INPUT:", ex["input"])
        print("PRED :", pred)
        print("REF  :", ex["reference"])
        print("-----")


if __name__ == "__main__":
    main()
