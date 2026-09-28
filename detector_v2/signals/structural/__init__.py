"""Structural / red-flag signal (Stage 0).

Cheap, always-available features over a claim — entities, numbers, dates,
superlatives, absolutes, false-premise cues, hedging vs over-assertion — mapped
to a heuristic risk contribution. This is the dependency-free floor signal.
"""
from __future__ import annotations

from .features import StructuralFeatures, extract_features
from .signal import StructuralSignal

__all__ = ["StructuralFeatures", "extract_features", "StructuralSignal"]
