"""Aggregation strategies mapping a list of per-claim risks -> one response risk.

All take ``(values: list[float], k: int)`` and return a float in [0, 1].
``learned`` is a placeholder that falls back to top-k until Stage 7 trains one.
"""
from __future__ import annotations

from typing import Callable, Dict, List


def _clean(values: List[float]) -> List[float]:
    return [min(max(float(v), 0.0), 1.0) for v in values if v is not None]


def agg_max(values: List[float], k: int = 1) -> float:
    vals = _clean(values)
    return max(vals) if vals else 0.0


def agg_mean(values: List[float], k: int = 1) -> float:
    vals = _clean(values)
    return sum(vals) / len(vals) if vals else 0.0


def agg_topk_mean(values: List[float], k: int = 3) -> float:
    """Mean of the k riskiest claims — a middle ground between max and mean that
    resists both single-claim noise and dilution."""
    vals = sorted(_clean(values), reverse=True)
    if not vals:
        return 0.0
    top = vals[: max(1, k)]
    return sum(top) / len(top)


def agg_noisy_or(values: List[float], k: int = 1) -> float:
    """P(at least one claim hallucinated) under independence: 1 - prod(1 - p)."""
    vals = _clean(values)
    if not vals:
        return 0.0
    prod = 1.0
    for v in vals:
        prod *= (1.0 - v)
    return 1.0 - prod


def agg_learned(values: List[float], k: int = 3) -> float:
    # Stage 7 replaces this with a trained aggregator; until then, top-k.
    return agg_topk_mean(values, k)


AGGREGATORS: Dict[str, Callable[[List[float], int], float]] = {
    "max": agg_max,
    "mean": agg_mean,
    "topk_mean": agg_topk_mean,
    "noisy_or": agg_noisy_or,
    "learned": agg_learned,
}


def aggregate(values: List[float], method: str = "topk_mean", k: int = 3) -> float:
    fn = AGGREGATORS.get(method, agg_topk_mean)
    return fn(values, k)
