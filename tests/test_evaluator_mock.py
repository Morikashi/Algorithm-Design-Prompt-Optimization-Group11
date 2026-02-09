from src.prompt.representation import Prompt
from src.evaluation.mock_llm import MockLLM
from src.evaluation.metrics import ExactMatchMetric, RougeLMetric
from src.evaluation.evaluator import Evaluator


def test_evaluator_qa_exact_match():
    llm = MockLLM()
    metric = ExactMatchMetric()
    evaluator = Evaluator(llm=llm, metric=metric)

    dataset = [
        {"input": "Capital of France?", "reference": "Paris"},
        {"input": "2+2?", "reference": "4"},
    ]

    p = Prompt(
        task="qa",
        instruction="Answer the question.",
        constraints=["Answer with only the final answer (no extra text)."],
        output_format="plain",
        style="concise",
        verification=False,
    )

    res = evaluator.score_prompt(p, dataset)
    assert res.quality_score == 1.0
    assert res.final_score == 1.0


def test_evaluator_summarization_rouge_l_runs():
    llm = MockLLM()
    metric = RougeLMetric()
    evaluator = Evaluator(llm=llm, metric=metric)

    dataset = [
        {
            "input": "OpenAI released a new model. It improves reasoning and speed. Users reported better results.",
            "reference": "OpenAI released a new model that improves reasoning and speed.",
        }
    ]

    p = Prompt(
        task="summarization",
        instruction="Summarize the text.",
        constraints=["Keep the summary under 3 sentences."],
        output_format="plain",
        style="concise",
        verification=False,
    )

    res = evaluator.score_prompt(p, dataset)
    assert 0.0 <= res.quality_score <= 1.0
    assert len(res.per_example) == 1
