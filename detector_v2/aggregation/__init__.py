"""Response-level aggregation of per-claim risks.

Several strategies are provided so Stage 7 can pick the best by validation
rather than assuming the arithmetic mean is right (mean dilutes a single
fabricated clause among many true ones — usually the wrong choice for a gate).
"""
from __future__ import annotations

from .aggregators import AGGREGATORS, aggregate

__all__ = ["AGGREGATORS", "aggregate"]
