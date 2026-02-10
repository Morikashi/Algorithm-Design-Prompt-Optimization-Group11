from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Set, Callable

from src.prompt.representation import Prompt
from src.utils.hashing import hash_text


@dataclass(frozen=True)
class HeuristicConfig:
    max_prompt_words: int = 250
    novelty_threshold: float = 0.92  # 1.0 means identical; lower is stricter
    enabled: bool = True


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
