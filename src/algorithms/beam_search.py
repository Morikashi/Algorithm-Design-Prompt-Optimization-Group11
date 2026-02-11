from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from src.prompt.representation import Prompt
from src.prompt.operators import generate_neighbors
from src.evaluation.evaluator import Evaluator, EvalResult
from src.utils.hashing import hash_text
from src.utils.trace import TraceLogger, TraceEvent
from src.algorithms.heuristics import HeuristicConfig, filter_neighbors

from src.analysis.embeddings import OllamaEmbeddings
from src.algorithms.heuristics import filter_neighbors_embedding

@dataclass
class BeamConfig:
    max_depth: int = 5
    beam_width: int = 5
    max_prompt_evals: int = 80
    max_neighbors: int = 25
    heuristic_cfg: HeuristicConfig = HeuristicConfig(enabled=False)


@dataclass
class BeamResult:
    best_prompt: Prompt
    best_eval: EvalResult
    eval_count: int
    trace_path: Optional[str]


def beam_search(
    start: Prompt,
    evaluator: Evaluator,
    dataset: List[dict],
    config: BeamConfig,
    trace_logger: Optional[TraceLogger] = None,
    run_id: str = "run",
    neighbor_fn: Optional[Callable[[Prompt, int], List[Prompt]]] = None,
) -> BeamResult:
    neighbor_fn = neighbor_fn or (lambda p, m: generate_neighbors(p, max_neighbors=m))

    eval_idx = 0
    best_so_far = float("-inf")

    # Evaluate start
    start_eval = evaluator.score_prompt(start, dataset)
    eval_idx += 1
    best_prompt = start
    best_eval = start_eval
    best_so_far = max(best_so_far, best_eval.final_score)

    evaluated_texts: List[str] = [start.render()]
    embedder = OllamaEmbeddings() if config.heuristic_cfg.enabled and config.heuristic_cfg.mode == "embedding" else None

    if trace_logger:
        trace_logger.log(
            TraceEvent(
                run_id=run_id,
                algorithm="beam",
                step=0,
                eval_idx=eval_idx,
                prompt_id=best_eval.prompt_id,
                score=best_eval.final_score,
                quality_score=best_eval.quality_score,
                cost=best_eval.cost,
                best_so_far=best_so_far,
            )
        )

    beam: List[Tuple[Prompt, EvalResult]] = [(start, start_eval)]

    for depth in range(1, config.max_depth + 1):
        if eval_idx >= config.max_prompt_evals:
            break

        candidates: List[Tuple[Prompt, EvalResult]] = []

        for p, _peval in beam:
            neighbors = neighbor_fn(p, config.max_neighbors)
            # neighbors = filter_neighbors(neighbors, evaluated_texts, config.heuristic_cfg)
            if config.heuristic_cfg.enabled and config.heuristic_cfg.mode == "embedding":
                assert embedder is not None
                neighbors = filter_neighbors_embedding(neighbors, evaluated_texts, config.heuristic_cfg, embedder)
            else:
                neighbors = filter_neighbors(neighbors, evaluated_texts, config.heuristic_cfg)

            for nb in neighbors:
                if eval_idx >= config.max_prompt_evals:
                    break

                nb_eval = evaluator.score_prompt(nb, dataset)
                eval_idx += 1
                evaluated_texts.append(nb.render())

                best_so_far = max(best_so_far, nb_eval.final_score)
                if trace_logger:
                    trace_logger.log(
                        TraceEvent(
                            run_id=run_id,
                            algorithm="beam",
                            step=depth,
                            eval_idx=eval_idx,
                            prompt_id=nb_eval.prompt_id,
                            score=nb_eval.final_score,
                            quality_score=nb_eval.quality_score,
                            cost=nb_eval.cost,
                            best_so_far=best_so_far,
                        )
                    )

                candidates.append((nb, nb_eval))

                if nb_eval.final_score > best_eval.final_score:
                    best_prompt = nb
                    best_eval = nb_eval

        # Keep top beam_width
        candidates.sort(key=lambda t: t[1].final_score, reverse=True)
        beam = candidates[: config.beam_width]

        if not beam:
            break

    return BeamResult(
        best_prompt=best_prompt,
        best_eval=best_eval,
        eval_count=eval_idx,
        trace_path=(trace_logger.path if trace_logger else None),
    )
