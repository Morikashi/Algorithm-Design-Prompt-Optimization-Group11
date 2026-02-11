from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import requests

from src.utils.hashing import hash_text


@dataclass(frozen=True)
class EmbeddingConfig:
    host: str = ""
    model: str = ""
    timeout_s: float = 60.0


class OllamaEmbeddings:
    """
    Minimal Ollama embeddings client.
    Uses POST /api/embed (Ollama).
    """

    def __init__(self, cfg: Optional[EmbeddingConfig] = None):
        cfg = cfg or EmbeddingConfig()
        host = (cfg.host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        model = (cfg.model or os.getenv("OLLAMA_EMBED_MODEL") or "granite-embedding:30m")
        self.host = host
        self.model = model
        self.timeout_s = cfg.timeout_s

        # In-memory cache: text-hash -> vector
        self._cache: Dict[str, List[float]] = {}

    def embed(self, text: str) -> List[float]:
        key = hash_text(text)
        if key in self._cache:
            return self._cache[key]

        url = f"{self.host}/api/embed"
        payload = {
            "model": self.model,
            "input": text,
        }

        resp = requests.post(url, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()

        # Ollama returns {"embeddings": [[...]]} for single input in some versions,
        # or {"embedding": [...]} in others. Handle both defensively.
        vec = None
        if isinstance(data.get("embedding"), list):
            vec = data["embedding"]
        elif isinstance(data.get("embeddings"), list) and data["embeddings"]:
            vec = data["embeddings"][0]

        if vec is None:
            raise ValueError(f"Unexpected embed response keys: {list(data.keys())}")

        self._cache[key] = vec
        return vec
