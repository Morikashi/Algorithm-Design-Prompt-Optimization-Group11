from src.prompt.representation import Prompt
from src.evaluation.mock_llm import MockLLM
from src.evaluation.metrics import ExactMatchMetric
from src.evaluation.evaluator import Evaluator
from src.algorithms.simulated_annealing import simulated_annealing, SAConfig


def test_sa_smoke_runs_and_respects_budget():
    dataset = [
        {"input": "Capital of France?", "reference": "Paris"},
        {"input": "2+2?", "reference": "4"},
    ]

    start = Prompt(task="qa", instruction="Answer the question.", constraints=[], output_format="plain")
    evaluator = Evaluator(llm=MockLLM(), metric=ExactMatchMetric())

    cfg = SAConfig(max_steps=50, max_prompt_evals=20, max_neighbors=10, t_start=1.0, t_end=0.05, seed=1)

    res = simulated_annealing(start=start, evaluator=evaluator, dataset=dataset, config=cfg, trace_logger=None, run_id="test")

    assert res.eval_count <= cfg.max_prompt_evals
    assert 0.0 <= res.best_eval.quality_score <= 1.0
