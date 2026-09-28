"""Claim segmentation: split an LLM response into risk-units ("claims").

The Detector's claims are *triage units*, not the Verifier's authoritative
atomic facts. They only need to be granular enough that one fabricated clause
inside an otherwise-true paragraph is not diluted at the response level.
"""
from __future__ import annotations

from .base import Claim, ClaimSegmenter
from .rule_based import RuleBasedSegmenter

__all__ = ["Claim", "ClaimSegmenter", "RuleBasedSegmenter", "segment"]


def segment(text: str, **kwargs) -> "list[Claim]":
    """Convenience: segment with the default rule-based segmenter."""
    return RuleBasedSegmenter(**kwargs).segment(text)
