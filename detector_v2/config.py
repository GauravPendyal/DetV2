"""Detector V2 configuration.

Thresholds mirror V1 (low=0.30, high=0.50) so V1-vs-V2 comparisons are made on
the same routing gate rather than a moved goalpost. Everything is overridable
via ``DETECTORV2_*`` environment variables so deployment can retune without a
code change.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class DetectorV2Config:
    """Runtime configuration for the V2 pipeline."""

    # Routing gate (same as V1 for comparability). risk <= low -> LOW/Accept.
    low_risk_threshold: float = field(default_factory=lambda: _env_float("DETECTORV2_LOW", 0.30))
    high_risk_threshold: float = field(default_factory=lambda: _env_float("DETECTORV2_HIGH", 0.50))

    # Claim segmentation guards.
    max_claims: int = field(default_factory=lambda: _env_int("DETECTORV2_MAX_CLAIMS", 64))
    min_claim_chars: int = field(default_factory=lambda: _env_int("DETECTORV2_MIN_CLAIM_CHARS", 8))

    # Response-level aggregation strategy (Stage 0 default; Stage 7 compares alternatives).
    aggregation: str = field(default_factory=lambda: os.getenv("DETECTORV2_AGG", "topk_mean"))
    aggregation_k: int = field(default_factory=lambda: _env_int("DETECTORV2_AGG_K", 3))

    # Which stage identity to advertise in model_source / diagnostics.
    stage: str = field(default_factory=lambda: os.getenv("DETECTORV2_STAGE", "stage0_heuristic"))

    def __post_init__(self) -> None:
        # Keep the gate coherent even if env vars are set inconsistently.
        if self.high_risk_threshold < self.low_risk_threshold:
            self.high_risk_threshold = self.low_risk_threshold

    @property
    def model_source(self) -> str:
        return f"detector_v2_{self.stage}"


DEFAULT_CONFIG = DetectorV2Config()
