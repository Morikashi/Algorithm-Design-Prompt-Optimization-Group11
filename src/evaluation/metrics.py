from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import List


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    # remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # normalize whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip()


@dataclass(frozen=True)
class MetricResult:
    score: float
    details: str = ""


class ExactMatchMetric:
    name = "exact_match"

    def score(self, prediction: str, reference: str) -> MetricResult:
        p = normalize_text(prediction)
        r = normalize_text(reference)
        return MetricResult(score=1.0 if p == r else 0.0, details=f"pred='{p}' ref='{r}'")


def lcs_length(a: List[str], b: List[str]) -> int:
    # DP LCS length, O(len(a)*len(b)) which is fine for small Phase-1 datasets
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


class RougeLMetric:
    """
    ROUGE-L F1 based on token-level LCS.
    Deterministic and lightweight for Phase 1.
    """
    name = "rouge_l"

    def score(self, prediction: str, reference: str) -> MetricResult:
        pred_toks = normalize_text(prediction).split()
        ref_toks = normalize_text(reference).split()

        if not pred_toks and not ref_toks:
            return MetricResult(score=1.0, details="both empty")
        if not pred_toks or not ref_toks:
            return MetricResult(score=0.0, details="one empty")

        lcs = lcs_length(pred_toks, ref_toks)
        prec = lcs / max(1, len(pred_toks))
        rec = lcs / max(1, len(ref_toks))
        if prec + rec == 0:
            f1 = 0.0
        else:
            f1 = 2 * prec * rec / (prec + rec)

        return MetricResult(score=float(f1), details=f"lcs={lcs} prec={prec:.3f} rec={rec:.3f}")
