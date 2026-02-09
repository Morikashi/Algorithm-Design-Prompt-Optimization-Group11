from src.prompt.representation import Prompt
from src.evaluation.mock_llm import MockLLM
from src.evaluation.metrics import ExactMatchMetric
from src.evaluation.evaluator import Evaluator
from src.algorithms.hill_climbing import hill_climb, HillClimbConfig


def test_hill_climb_improves_or_matches():
    dataset = [
        {"input": "Capital of France?", "reference": "Paris"},
        {"input": "2+2?", "reference": "4"},
    ]

    # Start prompt is intentionally weak: mock LLM may return "The answer is Paris."
    start = Prompt(
        task="qa",
        instruction="Answer the question.",
        constraints=[],  # no constraint forcing exact answer-only output
        output_format="plain",
        style=None,
        verification=False,
    )

    evaluator = Evaluator(llm=MockLLM(), metric=ExactMatchMetric())

    cfg = HillClimbConfig(max_steps=10, max_prompt_evals=30, max_neighbors=20, require_strict_improvement=True)

    res = hill_climb(start=start, evaluator=evaluator, dataset=dataset, config=cfg, trace_logger=None, run_id="test")

    # Must not crash; should find best prompt with EM=1.0 in this setup
    assert 0.0 <= res.best_eval.quality_score <= 1.0
    assert res.eval_count <= cfg.max_prompt_evals
    assert res.best_eval.quality_score == 1.0
