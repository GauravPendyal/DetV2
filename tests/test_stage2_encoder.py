import numpy as np

from detector_v2.claims.base import Claim
from detector_v2.pipeline import DetectorV2Pipeline
from detector_v2.schemas import ExecutionStatus, NextAction
from detector_v2.signals.base import SignalContext
from detector_v2.signals.encoder.signal import EncoderSignal


class _StubEncoder:
    """Deterministic stand-in for the DeBERTa encoder so the wiring is testable
    without torch or a trained model: risk grows with the digit count of a claim
    (numbers are the kind of checkable specific the real signal keys on)."""

    def __init__(self):
        self.calls = 0
        self.total_scored = 0

    def predict_proba(self, pairs, batch_size=16):
        self.calls += 1
        self.total_scored += len(pairs)
        out = [min(0.05 + 0.2 * sum(ch.isdigit() for ch in claim), 0.99) for _q, claim in pairs]
        return np.asarray(out, dtype=float)


def _claims(*texts):
    return [Claim(claim_id=i, text=t, span=[0, len(t)]) for i, t in enumerate(texts)]


def test_prepare_scores_all_claims_in_one_batch():
    sig = EncoderSignal(encoder=_StubEncoder())
    claims = _claims("No digits here.", "Built in 1889.", "It cost 4200 dollars in 1999.")
    ctx = SignalContext(user_query="q", response="...")
    sig.prepare(claims, ctx)
    assert sig._encoder.calls == 1  # one batched forward pass, not per-claim
    results = [sig.score(c, ctx) for c in claims]
    assert all(r.available and 0.0 <= r.value <= 1.0 for r in results)
    # more digits -> higher risk
    assert results[0].value < results[1].value < results[2].value
    assert "encoder_prob" in results[0].features


def test_score_without_prepare_falls_back_to_single():
    sig = EncoderSignal(encoder=_StubEncoder())
    ctx = SignalContext(user_query="q", response="...")
    r = sig.score(Claim(claim_id=0, text="Built in 1889.", span=[0, 14]), ctx)
    assert r.available and r.value > 0.0


def test_missing_model_dir_degrades_honestly():
    sig = EncoderSignal(model_dir="/definitely/not/a/real/model/dir")
    claims = _claims("Anything at all.")
    ctx = SignalContext(user_query="q", response="Anything at all.")
    sig.prepare(claims, ctx)
    r = sig.score(claims[0], ctx)
    assert r.available is False
    assert r.value is None
    assert "encoder_load_failed" in (r.unavailable_reason or "")


def test_pipeline_integrates_encoder_signal():
    pipeline = DetectorV2Pipeline(signals=[EncoderSignal(encoder=_StubEncoder())])
    out = pipeline.detect("q", "The tower was built in 1889 and cost 7000000 francs.")
    assert out.status == ExecutionStatus.COMPLETED
    assert 0.0 <= out.hallucination_probability <= 1.0
    assert any(s.name == "encoder" and s.available for c in out.claims for s in c.signals)


def test_pipeline_degrades_when_encoder_unavailable():
    pipeline = DetectorV2Pipeline(signals=[EncoderSignal(model_dir="/no/such/model")])
    out = pipeline.detect("q", "A plain sentence with no model behind it.")
    assert out.status == ExecutionStatus.DEGRADED
    assert out.next_action == NextAction.VERIFY
    assert out.hallucination_probability is None
    assert out.degraded_reason == "no_signal_available"
