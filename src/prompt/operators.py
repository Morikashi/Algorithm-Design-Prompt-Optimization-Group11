#This file defines:
#constraint pools (QA vs summarization)
#operator functions
#neighbor generation with caps + dedup

from __future__ import annotations
from dataclasses import replace
from typing import List, Set
from src.prompt.representation import Prompt
from src.utils.hashing import hash_text


QA_CONSTRAINT_POOL = [
    "Answer with only the final answer (no extra text).",
    "If you are unsure, output: I don't know.",
    "Do not explain your reasoning.",
    "Be максимально concise (one token if possible).",
]

SUM_CONSTRAINT_POOL = [
    "Do not add information that is not in the source.",
    "Keep the summary under 3 sentences.",
    "Preserve key entities and numbers.",
    "Use neutral, factual tone.",
]

OUTPUT_FORMATS = ["plain", "json", "bullets"]
STYLES = ["concise", "formal", "step_by_step"]


def _pool_for_task(task: str) -> List[str]:
    if task == "qa":
        return QA_CONSTRAINT_POOL
    if task == "summarization":
        return SUM_CONSTRAINT_POOL
    return []


# --- Operators (each returns a new Prompt) ---

def add_constraint(p: Prompt, constraint: str) -> Prompt:
    if constraint in p.constraints:
        return p
    return replace(p, constraints=p.constraints + [constraint])


def remove_constraint(p: Prompt, constraint: str) -> Prompt:
    if constraint not in p.constraints:
        return p
    return replace(p, constraints=[c for c in p.constraints if c != constraint])


def swap_constraint(p: Prompt, remove_c: str, add_c: str) -> Prompt:
    p2 = remove_constraint(p, remove_c)
    return add_constraint(p2, add_c)


def set_output_format(p: Prompt, fmt: str) -> Prompt:
    if p.output_format == fmt:
        return p
    return replace(p, output_format=fmt)


def change_style(p: Prompt, style: str) -> Prompt:
    if p.style == style:
        return p
    return replace(p, style=style)


def enable_verification(p: Prompt) -> Prompt:
    if p.verification:
        return p
    return replace(p, verification=True)


def disable_verification(p: Prompt) -> Prompt:
    if not p.verification:
        return p
    return replace(p, verification=False)


def reorder_constraints(p: Prompt) -> Prompt:
    """Simple reordering operator: reverse constraint order."""
    if len(p.constraints) < 2:
        return p
    return replace(p, constraints=list(reversed(p.constraints)))


# --- Neighbor generation ---

def generate_neighbors(p: Prompt, max_neighbors: int = 25) -> List[Prompt]:
    """
    Generate a deduplicated list of neighbor prompts using rule-based operators.
    Caps neighbors to control branching factor.
    """
    neighbors: List[Prompt] = []
    seen: Set[str] = set()

    def push(q: Prompt):
        h = hash_text(q.render())
        if h in seen:
            return
        seen.add(h)
        neighbors.append(q)

    pool = _pool_for_task(p.task)

    # Add each possible constraint
    for c in pool:
        push(add_constraint(p, c))
        if len(neighbors) >= max_neighbors:
            return neighbors

    # Remove each existing constraint
    for c in list(p.constraints):
        push(remove_constraint(p, c))
        if len(neighbors) >= max_neighbors:
            return neighbors

    # Swap: remove one existing, add one from pool
    for existing in list(p.constraints):
        for c in pool:
            if c == existing:
                continue
            push(swap_constraint(p, existing, c))
            if len(neighbors) >= max_neighbors:
                return neighbors

    # Output format variations
    for fmt in OUTPUT_FORMATS:
        push(set_output_format(p, fmt))
        if len(neighbors) >= max_neighbors:
            return neighbors

    # Style variations
    for s in STYLES:
        push(change_style(p, s))
        if len(neighbors) >= max_neighbors:
            return neighbors

    # Verification toggle
    push(enable_verification(p))
    if len(neighbors) >= max_neighbors:
        return neighbors
    push(disable_verification(p))
    if len(neighbors) >= max_neighbors:
        return neighbors

    # Reorder constraints
    push(reorder_constraints(p))

    return neighbors[:max_neighbors]
