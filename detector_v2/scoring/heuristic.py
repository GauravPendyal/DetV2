"""Deterministic per-claim scorer (Stage 0).

Fuses available signals by an optionally-weighted mean. With only the structural
signal wired in this is a pass-through; the weighting hook exists so additional
Stage-2+ signals can be blended without changing the pipeline. This is NOT a
learned fusion — that arrives in Stage 6.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..schemas import SignalResult
from .base import ClaimScorer


class HeuristicScorer(ClaimScorer):
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # Per-signal blend weights; unknown signals default to weight 1.0.
        self.weights = weights or {}

    def score_claim(self, signals: List[SignalResult]) -> Optional[float]:
        num = 0.0
        den = 0.0
        for s in signals:
            if not s.available or s.value is None:
                continue
            w = self.weights.get(s.name, 1.0)
            num += w * s.value
            den += w
        if den == 0.0:
            return None
        return num / den
