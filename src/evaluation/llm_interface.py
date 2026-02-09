from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LLMConfig:
    model: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout_s: float = 60.0


class LLMInterface(ABC):
    """
    Interface for any model backend (MockLLM, Ollama, etc.)
    """

    @abstractmethod
    def generate(self, prompt_text: str, user_input: str, config: Optional[LLMConfig] = None) -> str:
        """
        Given a rendered prompt and an input instance (question/article), return model output text.
        """
        raise NotImplementedError
