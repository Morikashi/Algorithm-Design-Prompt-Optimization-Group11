from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional, Set, Tuple

from src.prompt.representation import Prompt
from src.prompt.operators import generate_neighbors
from src.evaluation.evaluator import Evaluator, EvalResult
from src.utils.hashing import hash_text
from src.utils.trace import TraceLogger, TraceEvent
#from src.algorithms.heuristics import HeuristicConfig, filter_neighbors
from src.analysis.embeddings import OllamaEmbeddings
#from src.algorithms.heuristics import filter_neighbors_embedding


from src.algorithms.heuristics import HeuristicConfig, filter_neighbors, filter_neighbors_embedding


@dataclass
class BFSConfig:
    max_depth: int = 4
    max_prompt_evals: int = 80
    max_neighbors: int = 25
    heuristic_cfg: HeuristicConfig = HeuristicConfig(enabled=False)


@dataclass
class BFSResult:
    best_prompt: Prompt
    best_eval: EvalResult
    eval_count: int
    explored_nodes: int
    trace_path: Optional[str]


def bfs_search(
    start: Prompt,
    evaluator: Evaluator,
    dataset: List[dict],
    config: BFSConfig,
    trace_logger: Optional[TraceLogger] = None,
    run_id: str = "run",
    neighbor_fn: Optional[Callable[[Prompt, int], List[Prompt]]] = None,
) -> BFSResult:
    neighbor_fn = neighbor_fn or (lambda p, m: generate_neighbors(p, max_neighbors=m))

    eval_idx = 0
    best_so_far = float("-inf")

    start_eval = evaluator.score_prompt(start, dataset)
    eval_idx += 1
    best_prompt = start
    best_eval = start_eval
    best_so_far = max(best_so_far, best_eval.final_score)

    evaluated_texts: List[str] = [start.render()]
    embedder = OllamaEmbeddings() if config.heuristic_cfg.enabled and config.heuristic_cfg.mode == "embedding" else None # NEWLY ADDED


    # NEW:
    if trace_logger:
        trace_logger.log(
            TraceEvent(
                run_id=run_id,
                algorithm="bfs",
                step=0,
                eval_idx=eval_idx,
                prompt_id=best_eval.prompt_id,
                score=best_eval.final_score,
                quality_score=best_eval.quality_score,
                cost=best_eval.cost,
                best_so_far=best_so_far,
            )
        )

    q = deque([(start, 0)])
    visited: Set[str] = {hash_text(start.render())}
    explored_nodes = 0

    while q and eval_idx < config.max_prompt_evals:
        p, depth = q.popleft()
        explored_nodes += 1
        if depth >= config.max_depth:
            continue

        neighbors = neighbor_fn(p, config.max_neighbors)
        if config.heuristic_cfg.enabled and config.heuristic_cfg.mode == "embedding":
            assert embedder is not None
            neighbors = filter_neighbors_embedding(neighbors, evaluated_texts, config.heuristic_cfg, embedder)
        else:
            neighbors = filter_neighbors(neighbors, evaluated_texts, config.heuristic_cfg)


        for nb in neighbors:
            if eval_idx >= config.max_prompt_evals:
                break

            nb_hash = hash_text(nb.render())
            if nb_hash in visited:
                continue
            visited.add(nb_hash)

            nb_eval = evaluator.score_prompt(nb, dataset)
            eval_idx += 1

            evaluated_texts.append(nb.render())

            best_so_far = max(best_so_far, nb_eval.final_score)
            if trace_logger:
                trace_logger.log(
                    TraceEvent(
                        run_id=run_id,
                        algorithm="bfs",
                        step=depth + 1,
                        eval_idx=eval_idx,
                        prompt_id=nb_eval.prompt_id,
                        score=nb_eval.final_score,
                        quality_score=nb_eval.quality_score,
                        cost=nb_eval.cost,
                        best_so_far=best_so_far,
                    )
                )

            if nb_eval.final_score > best_eval.final_score:
                best_prompt = nb
                best_eval = nb_eval

            q.append((nb, depth + 1))

    return BFSResult(
        best_prompt=best_prompt,
        best_eval=best_eval,
        eval_count=eval_idx,
        explored_nodes=explored_nodes,
        trace_path=(trace_logger.path if trace_logger else None),
    )
