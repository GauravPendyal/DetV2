"""Detector V2 data contracts.

The three enums below are **byte-for-byte identical** to
``orchestration/schemas.py`` so the adapter can hand a valid ``DetectorResult``
to the rest of HalluciGuard with no translation loss. We deliberately keep our
own copy (like V1 does) so this package stands alone and never imports the
monorepo at runtime.

V2 adds *claim-level* structures (``ClaimSignal``, ``ClaimRisk``) on top of the
canonical six fields. These are additive: the adapter still projects down to the
exact six-field contract, and claim detail rides along as extra keys that
existing consumers simply ignore.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class NextAction(str, Enum):
    ACCEPT = "Accept"
    VERIFY = "Verify"


class ExecutionStatus(str, Enum):
    COMPLETED = "completed"
    DEGRADED = "degraded"
    FAILED = "failed"


class SignalResult(BaseModel):
    """One signal's contribution for one claim.

    ``available`` is the honesty gate: a signal that needs data it did not get
    (no context, no logprobs) reports ``available=False`` with ``value=None`` and
    a reason, and is excluded from scoring — never defaulted to a fake number.
    """
    model_config = ConfigDict(extra="ignore")

    name: str
    available: bool = True
    value: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    features: Dict[str, float] = Field(default_factory=dict)
    unavailable_reason: Optional[str] = None


class ClaimSignal(BaseModel):
    """A segmented claim plus every signal computed over it and its fused risk."""
    model_config = ConfigDict(extra="ignore")

    claim_id: int
    text: str
    span: Optional[List[int]] = Field(default=None, description="[start, end] char offsets in the response.")
    signals: List[SignalResult] = Field(default_factory=list)
    claim_risk: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class DetectorV2Output(BaseModel):
    """Rich internal result: canonical fields + honest diagnostics + claims.

    ``hallucination_probability`` / ``confidence_score`` are Optional because a
    DEGRADED/FAILED run refuses to invent them. Until calibration (Stage 8) runs,
    ``hallucination_probability`` is an *uncalibrated* risk score and
    ``calibrated`` stays False so no consumer mistakes it for a probability.
    """
    model_config = ConfigDict(extra="ignore", use_enum_values=True)

    hallucination_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    risk_level: RiskLevel
    next_action: NextAction
    model_source: str
    status: ExecutionStatus = ExecutionStatus.COMPLETED

    # --- diagnostics / extension (not part of the canonical six) ---
    calibrated: bool = False
    claims: List[ClaimSignal] = Field(default_factory=list)
    query_id: Optional[str] = None
    domain: str = "general"
    input_char_length: int = Field(default=0, ge=0)
    latency_ms: Optional[float] = Field(default=None, ge=0.0)
    degraded_reason: Optional[str] = None
