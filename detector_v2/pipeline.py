"""Detector V2 pipeline: claims -> signals -> per-claim scoring -> aggregation
-> routing gate. Stage 0 wiring (structural signal + heuristic scorer).

Failure policy (matches the brief): any missing model/signal or unexpected
error yields an HONEST degraded/failed result routed to Verify with null
probability — never a fabricated low-risk value.
"""
from __future__ import annotations

import time
from typing import List, Optional

from .aggregation import aggregate  # noqa: F401  (kept for BC; response scorers use it)
from .claims.base import Claim, ClaimSegmenter
from .claims.rule_based import RuleBasedSegmenter
from .config import DEFAULT_CONFIG, DetectorV2Config
from .schemas import (
    ClaimSignal,
    DetectorV2Output,
    ExecutionStatus,
    NextAction,
    RiskLevel,
    SignalResult,
)
from .scoring.base import ClaimScorer
from .scoring.heuristic import HeuristicScorer
from .scoring.response import AggregationResponseScorer, ResponseScorer
from .signals.base import Signal, SignalContext
from .signals.structural.signal import StructuralSignal


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


class DetectorV2Pipeline:
    def __init__(
        self,
        config: Optional[DetectorV2Config] = None,
        segmenter: Optional[ClaimSegmenter] = None,
        signals: Optional[List[Signal]] = None,
        scorer: Optional[ClaimScorer] = None,
        response_scorer: Optional[ResponseScorer] = None,
        calibrator=None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.segmenter = segmenter or RuleBasedSegmenter(
            min_claim_chars=self.config.min_claim_chars, max_claims=self.config.max_claims
        )
        # Stage 0: structural signal only. Later stages append to this list.
        self.signals = signals if signals is not None else [StructuralSignal()]
        self.scorer = scorer or HeuristicScorer()
        # Stage 0 default: aggregate per-claim risks. Stage 1+ can inject a
        # trained response scorer (e.g. logistic regression) here instead.
        self.response_scorer = response_scorer or AggregationResponseScorer(
            method=self.config.aggregation, k=self.config.aggregation_k
        )
        # Stage 7: optional leakage-safe calibrator. When set, the uncalibrated
        # response-risk is mapped to a calibrated P(hallucinated) and the output
        # is flagged calibrated=True; when None, behavior is unchanged.
        self.calibrator = calibrator

    def detect(
        self,
        user_query: str,
        llm_response: str,
        context: Optional[str] = None,
        domain: str = "general",
        query_id: Optional[str] = None,
    ) -> DetectorV2Output:
        start = time.perf_counter()
        char_len = len(llm_response or "")

        # Honest handling of empty input: nothing to assess -> route to Verify,
        # do NOT emit a fabricated low-risk number.
        if not llm_response or not llm_response.strip():
            return self._degraded(
                ExecutionStatus.DEGRADED, "empty_response", domain, query_id, char_len, start
            )

        try:
            return self._run(user_query, llm_response, context, domain, query_id, char_len, start)
        except Exception as exc:  # noqa: BLE001 - fail safe, never fabricate low risk
            return self._degraded(
                ExecutionStatus.FAILED, f"pipeline_error: {type(exc).__name__}: {exc}",
                domain, query_id, char_len, start,
            )

    # -- internals -------------------------------------------------------
    def _run(self, user_query, llm_response, context, domain, query_id, char_len, start) -> DetectorV2Output:
        claims = self.segmenter.segment(llm_response)
        if not claims:  # degenerate parse -> treat whole response as one claim
            claims = [Claim(claim_id=0, text=llm_response.strip(), span=[0, len(llm_response)])]

        ctx = SignalContext(
            user_query=user_query, context=context, response=llm_response, domain=domain
        )

        # Batched precompute hook: neural signals score all claims in one pass
        # here. A prepare() failure must not crash the run — the signal will then
        # report unavailable per claim and the pipeline degrades honestly.
        for sig in self.signals:
            try:
                sig.prepare(claims, ctx)
            except Exception:  # noqa: BLE001 - degrade, don't fabricate
                pass

        claim_signals: List[ClaimSignal] = []
        claim_risks: List[float] = []
        for claim in claims:
            results: List[SignalResult] = [sig.score(claim, ctx) for sig in self.signals]
            risk = self.scorer.score_claim(results)
            claim_signals.append(
                ClaimSignal(
                    claim_id=claim.claim_id, text=claim.text, span=claim.span,
                    signals=results, claim_risk=(None if risk is None else round(risk, 6)),
                )
            )
            if risk is not None:
                claim_risks.append(risk)

        # No signal produced a value for any claim -> honest degraded.
        if not claim_risks:
            return self._degraded(
                ExecutionStatus.DEGRADED, "no_signal_available", domain, query_id, char_len, start,
                claims=claim_signals,
            )

        response_risk = _clamp01(
            self.response_scorer.score(user_query, llm_response, claim_signals)
        )
        if self.calibrator is not None:
            # Calibrated probability + threshold-relative decisiveness.
            prob = _clamp01(self.calibrator.transform(response_risk))
            confidence = self.calibrator.decision_confidence(prob, self.config.low_risk_threshold)
            calibrated = True
        else:
            prob = response_risk
            confidence = max(prob, 1.0 - prob)  # decisiveness axis (uncalibrated)
            calibrated = False
        risk_level = self._classify(prob)
        latency_ms = (time.perf_counter() - start) * 1000.0

        return DetectorV2Output(
            hallucination_probability=round(prob, 6),
            confidence_score=round(confidence, 6),
            risk_level=risk_level,
            next_action=self._action(risk_level),
            model_source=self.config.model_source,
            status=ExecutionStatus.COMPLETED,
            calibrated=calibrated,
            claims=claim_signals,
            query_id=query_id,
            domain=domain,
            input_char_length=char_len,
            latency_ms=round(latency_ms, 3),
        )

    def _classify(self, risk: float) -> RiskLevel:
        if risk <= self.config.low_risk_threshold:
            return RiskLevel.LOW
        if risk >= self.config.high_risk_threshold:
            return RiskLevel.HIGH
        return RiskLevel.MEDIUM

    @staticmethod
    def _action(risk_level: RiskLevel) -> NextAction:
        # Safe default: only LOW is accepted; MEDIUM/HIGH route to the Verifier.
        return NextAction.ACCEPT if risk_level == RiskLevel.LOW else NextAction.VERIFY

    def _degraded(self, status, reason, domain, query_id, char_len, start, claims=None) -> DetectorV2Output:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return DetectorV2Output(
            hallucination_probability=None,
            confidence_score=None,
            risk_level=RiskLevel.HIGH,
            next_action=NextAction.VERIFY,
            model_source=self.config.model_source,
            status=status,
            calibrated=False,
            claims=claims or [],
            query_id=query_id,
            domain=domain,
            input_char_length=char_len,
            latency_ms=round(latency_ms, 3),
            degraded_reason=reason,
        )
