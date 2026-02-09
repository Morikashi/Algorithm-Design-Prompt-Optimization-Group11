from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from src.prompt.representation import Prompt
from src.prompt.operators import generate_neighbors
from src.evaluation.evaluator import Evaluator, EvalResult
from src.utils.trace import TraceLogger, TraceEvent


@dataclass
class HillClimbConfig:
    max_steps: int = 30
    max_prompt_evals: int = 60          # budget in number of prompt evaluations
    max_neighbors: int = 25             # controls branching factor
    require_strict_improvement: bool = True


@dataclass
class HillClimbResult:
    best_prompt: Prompt
    best_eval: EvalResult
    eval_count: int
    steps_taken: int
    trace_path: Optional[str]


def hill_climb(
    start: Prompt,
    evaluator: Evaluator,
    dataset: List[dict],
    config: HillClimbConfig,
    trace_logger: Optional[TraceLogger] = None,
    run_id: str = "run",
    neighbor_fn: Optional[Callable[[Prompt, int], List[Prompt]]] = None,
) -> HillClimbResult:
    """
    Hill climbing in prompt space.

    At each step:
      - generate neighbors
      - evaluate all neighbors (within budget)
      - move to best improving neighbor
      - stop if no improvement (local optimum) or budget/steps exhausted
    """
    neighbor_fn = neighbor_fn or (lambda p, m: generate_neighbors(p, max_neighbors=m))

    eval_idx = 0
    best_so_far = float("-inf")

    # Evaluate start prompt
    start_eval = evaluator.score_prompt(start, dataset)
    eval_idx += 1
    best_prompt = start
    best_eval = start_eval
    best_so_far = max(best_so_far, best_eval.final_score)

    if trace_logger:
        trace_logger.log(
            TraceEvent(
                run_id=run_id,
                algorithm="hill_climb",
                step=0,
                eval_idx=eval_idx,
                prompt_id=best_eval.prompt_id,
                score=best_eval.final_score,
                quality_score=best_eval.quality_score,
                cost=best_eval.cost,
                best_so_far=best_so_far,
            )
        )

    current = start
    current_eval = start_eval

    steps_taken = 0

    # Main loop
    for step in range(1, config.max_steps + 1):
        steps_taken = step

        if eval_idx >= config.max_prompt_evals:
            break

        neighbors = neighbor_fn(current, config.max_neighbors)
        if not neighbors:
            break

        best_neighbor: Optional[Prompt] = None
        best_neighbor_eval: Optional[EvalResult] = None

        # Evaluate neighbors
        for nb in neighbors:
            if eval_idx >= config.max_prompt_evals:
                break

            nb_eval = evaluator.score_prompt(nb, dataset)
            eval_idx += 1

            best_so_far = max(best_so_far, nb_eval.final_score)
            if trace_logger:
                trace_logger.log(
                    TraceEvent(
                        run_id=run_id,
                        algorithm="hill_climb",
                        step=step,
                        eval_idx=eval_idx,
                        prompt_id=nb_eval.prompt_id,
                        score=nb_eval.final_score,
                        quality_score=nb_eval.quality_score,
                        cost=nb_eval.cost,
                        best_so_far=best_so_far,
                    )
                )

            if (best_neighbor_eval is None) or (nb_eval.final_score > best_neighbor_eval.final_score):
                best_neighbor = nb
                best_neighbor_eval = nb_eval

        # No neighbors were evaluated (budget hit)
        if best_neighbor is None or best_neighbor_eval is None:
            break

        # Decide whether to move
        improved = best_neighbor_eval.final_score > current_eval.final_score
        if config.require_strict_improvement:
            if not improved:
                break
        else:
            # allow sideways moves
            if best_neighbor_eval.final_score < current_eval.final_score:
                break

        # Move
        current = best_neighbor
        current_eval = best_neighbor_eval

        # Track global best
        if current_eval.final_score > best_eval.final_score:
            best_prompt = current
            best_eval = current_eval

    return HillClimbResult(
        best_prompt=best_prompt,
        best_eval=best_eval,
        eval_count=eval_idx,
        steps_taken=steps_taken,
        trace_path=(trace_logger.path if trace_logger else None),
    )
