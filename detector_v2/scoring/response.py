"""Response-level scorers: turn a response's claims into one risk in [0, 1].

``AggregationResponseScorer`` (Stage 0) aggregates per-claim risks with a chosen
strategy. ``LogRegResponseScorer`` (Stage 1) featurizes the response and applies
a trained logistic-regression model. Both satisfy the same interface so the
pipeline is agnostic to which is wired in.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..aggregation import aggregate
from ..schemas import ClaimSignal


class ResponseScorer(ABC):
    name: str = "response_scorer"

    @abstractmethod
    def score(self, query: str, response: str, claims: List[ClaimSignal]) -> float:
        """Return response-level hallucination risk in [0, 1]."""
        raise NotImplementedError


class AggregationResponseScorer(ResponseScorer):
    """Stage-0 default: aggregate the per-claim risks (max / top-k / noisy-OR)."""

    name = "aggregation"

    def __init__(self, method: str = "topk_mean", k: int = 3):
        self.method = method
        self.k = k

    def score(self, query: str, response: str, claims: List[ClaimSignal]) -> float:
        risks = [c.claim_risk for c in claims if c.claim_risk is not None]
        return aggregate(risks, method=self.method, k=self.k)
