"""Per-claim risk scoring (signal fusion at the claim level).

Stage 0 uses a deterministic ``HeuristicScorer`` (mean of available signals).
Stage 1 adds a logistic-regression fusion, and Stage 6 compares LR vs a small
MLP. All implement the same ``ClaimScorer`` interface so the pipeline is
agnostic to which one is wired in.
"""
from __future__ import annotations

from .base import ClaimScorer
from .heuristic import HeuristicScorer

__all__ = ["ClaimScorer", "HeuristicScorer"]
