"""Signal extractors.

A *signal* observes one claim and returns a ``SignalResult`` — a normalized
risk contribution in [0, 1] plus raw features, or an honest
``available=False`` when its required inputs are missing. Signals never
fabricate a value to fill a gap; the scorer simply drops unavailable signals.

Stage 0 ships ``structural`` only. ``encoder`` (Stage 2), ``entailment``
(Stage 3), ``uncertainty`` (Stage 4) and ``self_consistency`` (Stage 5) are
added later behind the same ``Signal`` interface.
"""
from __future__ import annotations

from .base import Signal

__all__ = ["Signal"]
