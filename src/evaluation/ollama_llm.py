from __future__ import annotations

import os
from typing import Optional

import requests

from src.evaluation.llm_interface import LLMInterface, LLMConfig


class OllamaLLM(LLMInterface):
    """
    Local LLM backend using Ollama HTTP API.
    Requires: ollama serve running locally (or reachable host).
    """

    def __init__(self, host: Optional[str] = None, model: Optional[str] = None):
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL") or "llama3.1:8b"

    def generate(self, prompt_text: str, user_input: str, config: Optional[LLMConfig] = None) -> str:
        cfg = config or LLMConfig(model=self.model)

        model = cfg.model or self.model
        temperature = cfg.temperature
        timeout_s = cfg.timeout_s

        # We compose a single prompt string. Prompt text already includes task tags and constraints.
        full_prompt = f"{prompt_text}\n\nINPUT:\n{user_input}\n\nOUTPUT:"

        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        # Optional max_tokens: Ollama uses num_predict
        if cfg.max_tokens is not None:
            payload["options"]["num_predict"] = int(cfg.max_tokens)

        url = f"{self.host}/api/generate"
        resp = requests.post(url, json=payload, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()

        # Ollama returns {"response": "...", ...}
        return (data.get("response") or "").strip()
