from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import List, Dict, Any

from src.evaluation.evaluator import EvalResult


def save_worst_examples(
    eval_result: EvalResult,
    out_dir: str,
    run_id: str,
    k: int = 5,
) -> str:
    """
    Save the k worst per-example results (lowest metric_score) to a JSON file.
    This is used for failure case analysis in Phase 2.
    """
    os.makedirs(out_dir, exist_ok=True)

    per = eval_result.per_example
    sorted_per = sorted(per, key=lambda r: r.metric_score)[:k]

    payload: Dict[str, Any] = {
        "run_id": run_id,
        "prompt_id": eval_result.prompt_id,
        "quality_score": eval_result.quality_score,
        "final_score": eval_result.final_score,
        "k": k,
        "worst_examples": [asdict(x) for x in sorted_per],
        "prompt_text": eval_result.prompt_text,
    }

    path = os.path.join(out_dir, f"worst_{run_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return path
