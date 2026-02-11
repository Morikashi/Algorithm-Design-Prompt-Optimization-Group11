from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Set, Callable, Optional

from src.prompt.representation import Prompt
from src.utils.hashing import hash_text
from src.analysis.vector_math import cosine_similarity
from src.analysis.embeddings import OllamaEmbeddings


@dataclass(frozen=True)
class HeuristicConfig:
    max_prompt_words: int = 250
    novelty_threshold: float = 0.92  # 1.0 means identical; lower is stricter
    enabled: bool = True

    # NEW:
    mode: str = "jaccard"  # "jaccard" or "embedding"
    embedding_cosine_threshold: float = 0.90  # used when mode="embedding"

def _jaccard_similarity(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union


def filter_neighbors(
    neighbors: Sequence[Prompt],
    evaluated_prompt_texts: Sequence[str],
    cfg: HeuristicConfig,
) -> List[Prompt]:
    """
    Heuristic constraints:
    - length cap (by words)
    - novelty cap (discard prompts too similar to any already evaluated prompt)
    """
    if not cfg.enabled:
        return list(neighbors)
    
    # If embedding mode requested, caller should use filter_neighbors_embedding(...)
    if getattr(cfg, "mode", "jaccard") == "embedding":
        # fall back to jaccard here, but recommended to call embedding function explicitly
        pass

    filtered: List[Prompt] = []
    seen_hashes: Set[str] = set()

    for p in neighbors:
        text = p.render()
        if len(text.split()) > cfg.max_prompt_words:
            continue

        h = hash_text(text)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        # novelty check
        too_similar = False
        for prev in evaluated_prompt_texts:
            if _jaccard_similarity(text, prev) >= cfg.novelty_threshold:
                too_similar = True
                break
        if too_similar:
            continue

        filtered.append(p)

    return filtered


def filter_neighbors_embedding(
    neighbors: Sequence[Prompt],
    evaluated_prompt_texts: Sequence[str],
    cfg: HeuristicConfig,
    embedder: OllamaEmbeddings,
) -> List[Prompt]:
    """
    Embedding-based novelty pruning: reject a neighbor if its cosine similarity
    to any previously evaluated prompt is >= embedding_cosine_threshold.
    """
    if not cfg.enabled:
        return list(neighbors)

    filtered: List[Prompt] = []
    seen_hashes: Set[str] = set()

    # Pre-embed evaluated prompts once (cache helps)
    eval_vecs = [embedder.embed(t) for t in evaluated_prompt_texts]

    for p in neighbors:
        text = p.render()

        if len(text.split()) > cfg.max_prompt_words:
            continue

        h = hash_text(text)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        v = embedder.embed(text)

        too_similar = False
        for ev in eval_vecs:
            if cosine_similarity(v, ev) >= cfg.embedding_cosine_threshold:
                too_similar = True
                break
        if too_similar:
            continue

        filtered.append(p)

    return filtered
