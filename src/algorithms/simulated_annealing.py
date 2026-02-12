from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Optional

from src.prompt.representation import Prompt
from src.prompt.operators import generate_neighbors
from src.evaluation.evaluator import Evaluator, EvalResult
from src.utils.trace import TraceLogger, TraceEvent


@dataclass
class SAConfig:
    max_steps: int = 200
    max_prompt_evals: int = 120
    max_neighbors: int = 25

    # Temperature schedule
    t_start: float = 1.0
    t_end: float = 0.05
    schedule: str = "exp"   # "exp" or "linear"

    # Randomness
    seed: int = 42


@dataclass
class SAResult:
    best_prompt: Prompt
    best_eval: EvalResult
    eval_count: int
    steps_taken: int
    trace_path: Optional[str]


def _temperature(step: int, total: int, t0: float, t1: float, schedule: str) -> float:
    if total <= 1:
        return t1
    progress = step / (total - 1)

    if schedule == "linear":
        return t0 + (t1 - t0) * progress

    # exp schedule (default)
    # t = t0 * (t1/t0) ^ progress
    if t0 <= 0 or t1 <= 0:
        return max(t1, 1e-6)
    return t0 * ((t1 / t0) ** progress)


def simulated_annealing(
    start: Prompt,
    evaluator: Evaluator,
    dataset: List[dict],
    config: SAConfig,
    trace_logger: Optional[TraceLogger] = None,
    run_id: str = "run",
    neighbor_fn: Optional[Callable[[Prompt, int], List[Prompt]]] = None,
) -> SAResult:
    """
    Simulated Annealing over prompt space:
    - At each step, sample a neighbor
    - Accept if improved, else accept with probability exp((new-old)/T)
    - Track global best
    Budgeted by max_prompt_evals.
    """
    rnd = random.Random(config.seed)
    neighbor_fn = neighbor_fn or (lambda p, m: generate_neighbors(p, max_neighbors=m))

    eval_idx = 0
    best_so_far = float("-inf")

    # Evaluate start
    current = start
    current_eval = evaluator.score_prompt(current, dataset)
    eval_idx += 1

    best_prompt = current
    best_eval = current_eval
    best_so_far = max(best_so_far, best_eval.final_score)

    if trace_logger:
        trace_logger.log(
            TraceEvent(
                run_id=run_id,
                algorithm="sa",
                step=0,
                eval_idx=eval_idx,
                prompt_id=current_eval.prompt_id,
                score=current_eval.final_score,
                quality_score=current_eval.quality_score,
                cost=current_eval.cost,
                best_so_far=best_so_far,
            )
        )

    steps_taken = 0

    for step in range(1, config.max_steps + 1):
        steps_taken = step
        if eval_idx >= config.max_prompt_evals:
            break

        # Temperature at this step
        T = _temperature(step - 1, config.max_steps, config.t_start, config.t_end, config.schedule)

        neighbors = neighbor_fn(current, config.max_neighbors)
        if not neighbors:
            break

        # Sample one neighbor (classic SA)
        candidate = rnd.choice(neighbors)
        cand_eval = evaluator.score_prompt(candidate, dataset)
        eval_idx += 1

        delta = cand_eval.final_score - current_eval.final_score

        accept = False
        if delta >= 0:
            accept = True
        else:
            # probability to accept worse move
            # exp(delta / T), T must be > 0
            if T <= 1e-12:
                accept = False
            else:
                p = math.exp(delta / T)
                accept = (rnd.random() < p)

        if accept:
            current = candidate
            current_eval = cand_eval

        if cand_eval.final_score > best_eval.final_score:
            best_prompt = candidate
            best_eval = cand_eval

        best_so_far = max(best_so_far, cand_eval.final_score)

        if trace_logger:
            trace_logger.log(
                TraceEvent(
                    run_id=run_id,
                    algorithm="sa",
                    step=step,
                    eval_idx=eval_idx,
                    prompt_id=cand_eval.prompt_id,
                    score=cand_eval.final_score,
                    quality_score=cand_eval.quality_score,
                    cost=cand_eval.cost,
                    best_so_far=best_so_far,
                )
            )

    return SAResult(
        best_prompt=best_prompt,
        best_eval=best_eval,
        eval_count=eval_idx,
        steps_taken=steps_taken,
        trace_path=(trace_logger.path if trace_logger else None),
    )
