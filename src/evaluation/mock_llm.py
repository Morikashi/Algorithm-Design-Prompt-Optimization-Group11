from __future__ import annotations
import re
from typing import Optional
from src.evaluation.llm_interface import LLMInterface, LLMConfig


class MockLLM(LLMInterface):
    """
    Deterministic fake LLM used for unit tests and fast debugging.
    It uses simple rules to return outputs for QA and Summarization-like inputs.
    """

    def generate(self, prompt_text: str, user_input: str, config: Optional[LLMConfig] = None) -> str:
        # Heuristic: detect QA vs summarization based on prompt tag or keywords
        lower_prompt = prompt_text.lower()
        lower_inp = user_input.lower()

        is_qa = ("[task: qa]" in lower_prompt) or ("question" in lower_inp and "summary" not in lower_prompt)
        is_sum = ("[task: summarization]" in lower_prompt) or ("summar" in lower_prompt)

        if is_qa:
            return self._answer_qa(lower_prompt, lower_inp)

        if is_sum:
            return self._summarize(lower_prompt, user_input)

        # default fallback
        return "I don't know."

    def _answer_qa(self, prompt_text_lower: str, user_input_lower: str) -> str:
        # A few deterministic rules
        if "capital of france" in user_input_lower:
            ans = "Paris"
        elif re.search(r"\b2\s*\+\s*2\b", user_input_lower):
            ans = "4"
        elif "largest planet" in user_input_lower:
            ans = "Jupiter"
        else:
            ans = "I don't know."

        # If prompt demands only final answer, keep it short
        if "only the final answer" in prompt_text_lower or "no extra text" in prompt_text_lower:
            return ans
        return f"The answer is {ans}."

    def _summarize(self, prompt_text_lower: str, user_input: str) -> str:
        # “Summary” = first N words, with N based on constraints
        max_words = 35
        if "under 3 sentences" in prompt_text_lower:
            max_words = 30
        if "very concise" in prompt_text_lower or "concise" in prompt_text_lower:
            max_words = 18

        words = user_input.strip().split()
        summary = " ".join(words[:max_words]).strip()

        # If JSON format requested, wrap it
        if "output format: json" in prompt_text_lower:
            return f'{{"summary": "{summary}"}}'
        return summary
