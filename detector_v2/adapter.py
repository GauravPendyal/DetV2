"""Adapter to the canonical HalluciGuard ``DetectorResult`` contract.

``to_canonical_dict`` projects V2's rich output onto EXACTLY the six canonical
fields (byte-identical to V1's adapter behaviour), so existing consumers are
unaffected. ``to_extended_dict`` is the same six fields PLUS additive
claim-level detail — this is what the agent returns by default, and any consumer
reading only the six canonical keys keeps working.

Degraded/failed: the canonical contract cannot carry null numbers, so we emit
``confidence_score = 0.0`` (an honest "trust this at zero") together with
``status != completed`` and ``risk_level = HIGH`` / ``next_action = Verify``.
"""
from __future__ import annotations

from typing import Any, Dict

from .schemas import DetectorV2Output, ExecutionStatus

_CANONICAL_KEYS = (
    "hallucination_probability",
    "confidence_score",
    "risk_level",
    "next_action",
    "model_source",
    "status",
)


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _is_completed(output: DetectorV2Output) -> bool:
    return _enum_value(output.status) == ExecutionStatus.COMPLETED.value


def to_canonical_dict(output: DetectorV2Output) -> Dict[str, Any]:
    """Exactly the six canonical fields."""
    if _is_completed(output) and output.hallucination_probability is not None:
        prob = float(output.hallucination_probability)
        conf = float(output.confidence_score) if output.confidence_score is not None else 0.0
    else:
        # Fail-safe: no trustworthy numbers. 0.0 confidence flags exactly that.
        prob = 0.0
        conf = 0.0

    return {
        "hallucination_probability": prob,
        "confidence_score": conf,
        "risk_level": _enum_value(output.risk_level),
        "next_action": _enum_value(output.next_action),
        "model_source": output.model_source,
        "status": _enum_value(output.status),
    }


def to_extended_dict(output: DetectorV2Output) -> Dict[str, Any]:
    """Canonical six fields + additive V2 claim-level detail and diagnostics."""
    payload = to_canonical_dict(output)
    payload["calibrated"] = bool(output.calibrated)
    payload["claims"] = [
        {
            "claim_id": c.claim_id,
            "text": c.text,
            "span": c.span,
            "claim_risk": c.claim_risk,
            "signals": [
                {"name": s.name, "available": s.available, "value": s.value,
                 "unavailable_reason": s.unavailable_reason}
                for s in c.signals
            ],
        }
        for c in output.claims
    ]
    payload["diagnostics"] = {
        "query_id": output.query_id,
        "domain": output.domain,
        "input_char_length": output.input_char_length,
        "latency_ms": output.latency_ms,
        "degraded_reason": output.degraded_reason,
    }
    return payload


def to_canonical_model(output: DetectorV2Output):
    """Return a real ``orchestration.schemas.DetectorResult`` if importable,
    else the plain six-field dict — so V2 never hard-depends on the monorepo."""
    payload = to_canonical_dict(output)
    try:
        from orchestration.schemas import DetectorResult  # type: ignore
    except Exception:  # noqa: BLE001 - repo not importable in standalone use
        return payload
    return DetectorResult(**payload)
