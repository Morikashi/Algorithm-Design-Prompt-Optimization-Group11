from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class TraceEvent:
    run_id: str
    algorithm: str
    step: int
    eval_idx: int
    prompt_id: str
    score: float
    quality_score: float
    cost: float
    best_so_far: float


class TraceLogger:
    """
    Lightweight CSV trace logger. Safe for Phase 1.
    Writes into results/ and avoids committing heavy outputs.
    """

    def __init__(self, out_dir: str = "results", filename: Optional[str] = None):
        os.makedirs(out_dir, exist_ok=True)
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trace_{ts}.csv"
        self.path = os.path.join(out_dir, filename)

        self._initialized = False

    def log(self, e: TraceEvent) -> None:
        header = [
            "run_id",
            "algorithm",
            "step",
            "eval_idx",
            "prompt_id",
            "score",
            "quality_score",
            "cost",
            "best_so_far",
        ]

        write_header = not self._initialized and not os.path.exists(self.path)

        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow(
                [
                    e.run_id,
                    e.algorithm,
                    e.step,
                    e.eval_idx,
                    e.prompt_id,
                    f"{e.score:.6f}",
                    f"{e.quality_score:.6f}",
                    f"{e.cost:.3f}",
                    f"{e.best_so_far:.6f}",
                ]
            )

        self._initialized = True
