from src.prompt.representation import Prompt
from src.evaluation.mock_llm import MockLLM
from src.evaluation.metrics import ExactMatchMetric
from src.evaluation.evaluator import Evaluator

from src.algorithms.bfs import bfs_search, BFSConfig
from src.algorithms.beam_search import beam_search, BeamConfig
from src.algorithms.heuristics import HeuristicConfig


def test_bfs_smoke_qa():
    dataset = [
        {"input": "Capital of France?", "reference": "Paris"},
        {"input": "2+2?", "reference": "4"},
    ]
    evaluator = Evaluator(llm=MockLLM(), metric=ExactMatchMetric())
    start = Prompt(task="qa", instruction="Answer the question.", constraints=[], output_format="plain")

    res = bfs_search(
        start=start,
        evaluator=evaluator,
        dataset=dataset,
        config=BFSConfig(max_depth=3, max_prompt_evals=30, max_neighbors=15),
        trace_logger=None,
        run_id="test",
    )
    assert res.eval_count <= 30
    assert res.best_eval.quality_score == 1.0


def test_beam_smoke_qa():
    dataset = [
        {"input": "Capital of France?", "reference": "Paris"},
        {"input": "2+2?", "reference": "4"},
    ]
    evaluator = Evaluator(llm=MockLLM(), metric=ExactMatchMetric())
    start = Prompt(task="qa", instruction="Answer the question.", constraints=[], output_format="plain")

    res = beam_search(
        start=start,
        evaluator=evaluator,
        dataset=dataset,
        config=BeamConfig(max_depth=3, beam_width=3, max_prompt_evals=30, max_neighbors=15),
        trace_logger=None,
        run_id="test",
    )
    assert res.eval_count <= 30
    assert res.best_eval.quality_score == 1.0


def test_heuristics_config_does_not_crash():
    cfg = HeuristicConfig(enabled=True, novelty_threshold=0.95, max_prompt_words=200)
    assert cfg.enabled is True
