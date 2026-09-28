"""HalluciGuard Detector V2.

A calibrated, claim-aware, multi-signal hallucination *triage gate* built from
scratch alongside (never replacing) the V1 DistilBERT detector.

Public entry point mirrors the V1 agent surface so V2 can drop into the
orchestration graph once it is proven to beat V1:

    from detector_v2 import DetectorAgent
    result = DetectorAgent().detect(user_query, llm_response, context=None)

``result`` is a dict that always carries the six canonical ``DetectorResult``
fields and additively includes claim-level detail under ``claims``.
"""
from __future__ import annotations

from .agent import DetectorAgent

__all__ = ["DetectorAgent"]
__version__ = "0.1.0"
