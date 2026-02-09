from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from src.evaluation.llm_interface import LLMInterface, LLMConfig
from src.utils.hashing import hash_text
from src.prompt.representation import Prompt


@dataclass
class ExampleResult:
    input: str
    reference: str
    prediction: str
    metric_score: float
    metric_details: str


@dataclass
class EvalResult:
    prompt_text: str
    prompt_id: str
    quality_score: float
    cost: float
    final_score: float
    per_example: List[ExampleResult]


class Evaluator:
    """
    Evaluate a prompt on a dataset using a given LLM backend and metric.
    """

    def __init__(
        self,
        llm: LLMInterface,
        metric: Any,  # ExactMatchMetric or RougeLMetric (duck typing: .score(pred, ref)->MetricResult)
        lambda_cost: float = 0.0,
        llm_config: Optional[LLMConfig] = None,
    ):
        self.llm = llm
        self.metric = metric
        self.lambda_cost = lambda_cost
        self.llm_config = llm_config or LLMConfig()
        self._cache: Dict[str, str] = {}  # key -> prediction

    @staticmethod
    def _cost_proxy(prompt_text: str, user_input: str) -> float:
        # Simple deterministic proxy for Phase 1: word counts
        return float(len(prompt_text.split()) + len(user_input.split()))

    def score_prompt(self, prompt: Prompt, dataset: List[dict]) -> EvalResult:
        prompt_text = prompt.render()
        prompt_id = hash_text(prompt_text)

        per_example: List[ExampleResult] = []
        scores: List[float] = []
        total_cost = 0.0

        for ex in dataset:
            x = ex["input"]
            y = ex["reference"]

            cache_key = f"{prompt_id}:{hash_text(x)}"
            if cache_key in self._cache:
                pred = self._cache[cache_key]
            else:
                pred = self.llm.generate(prompt_text=prompt_text, user_input=x, config=self.llm_config)
                self._cache[cache_key] = pred

            mr = self.metric.score(pred, y)
            scores.append(mr.score)

            total_cost += self._cost_proxy(prompt_text, x)

            per_example.append(
                ExampleResult(
                    input=x,
                    reference=y,
                    prediction=pred,
                    metric_score=mr.score,
                    metric_details=mr.details,
                )
            )

        quality = sum(scores) / max(1, len(scores))
        final = quality - self.lambda_cost * total_cost

        return EvalResult(
            prompt_text=prompt_text,
            prompt_id=prompt_id,
            quality_score=quality,
            cost=total_cost,
            final_score=final,
            per_example=per_example,
        )
