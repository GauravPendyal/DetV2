import numpy as np

from detector_v2.claims.base import Claim
from detector_v2.pipeline import DetectorV2Pipeline
from detector_v2.schemas import ExecutionStatus, NextAction
from detector_v2.signals.base import SignalContext
from detector_v2.signals.entailment import EntailmentSignal, chunk_context


class _StubEntail:
    """Stand-in entailment model: P(unsupported) is low when the claim's content
    words appear in the context chunk, high otherwise. Lets us test the grounding
    logic without a trained model."""

    def __init__(self):
        self.calls = 0

    def predict_proba(self, pairs, batch_size=16):
        self.calls += 1
        out = []
        for chunk, claim in pairs:
            words = [w for w in claim.lower().split() if len(w) > 3]
            supported = words and all(w.strip(".,") in chunk.lower() for w in words)
            out.append(0.1 if supported else 0.9)
        return np.asarray(out, dtype=float)


def _claims(*texts):
    return [Claim(claim_id=i, text=t, span=[0, len(t)]) for i, t in enumerate(texts)]


def test_chunk_context_splits_long_document():
    doc = " ".join(f"Sentence number {i} about a distinct topic area." for i in range(40))
    chunks = chunk_context(doc, chunk_chars=200, max_chunks=12)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)  # bounded (chunk + one overflow sentence)


def test_context_absent_abstains_without_fabricating():
    sig = EntailmentSignal(model=_StubEntail())
    claims = _claims("The capital is Paris.")
    ctx = SignalContext(user_query="q", context=None, response="The capital is Paris.")
    sig.prepare(claims, ctx)
    r = sig.score(claims[0], ctx)
    assert r.available is False
    assert r.value is None
    assert r.unavailable_reason == "context_absent"


def test_grounded_claim_scores_low_risk():
    stub = _StubEntail()
    sig = EntailmentSignal(model=stub)
    context = "The Eiffel Tower stands in Paris and was completed in 1889."
    claims = _claims("The Eiffel Tower was completed in Paris.", "The tower is located in Tokyo.")
    ctx = SignalContext(user_query="q", context=context, response="...")
    sig.prepare(claims, ctx)
    grounded = sig.score(claims[0], ctx)
    unsupported = sig.score(claims[1], ctx)
    assert grounded.available and unsupported.available
    assert grounded.value < unsupported.value  # supported claim = lower risk
    assert stub.calls == 1  # single batched pass over all (chunk, claim) pairs


def test_best_supporting_chunk_wins():
    # Claim is supported by the SECOND chunk only; risk should still be low.
    sig = EntailmentSignal(model=_StubEntail(), chunk_chars=60)
    context = ("Unrelated filler about weather patterns and rainfall totals. "
               "Marie Curie discovered radium and polonium.")
    claims = _claims("Marie Curie discovered radium polonium.")
    ctx = SignalContext(user_query="q", context=context, response="...")
    sig.prepare(claims, ctx)
    r = sig.score(claims[0], ctx)
    assert r.available and r.value < 0.5


def test_pipeline_completes_with_context_and_degrades_without():
    pipe = DetectorV2Pipeline(signals=[EntailmentSignal(model=_StubEntail())])
    ctx_doc = "Water boils at 100 degrees Celsius at sea level pressure."
    with_ctx = pipe.detect("q", "Water boils at 100 degrees.", context=ctx_doc)
    assert with_ctx.status == ExecutionStatus.COMPLETED
    assert 0.0 <= with_ctx.hallucination_probability <= 1.0

    no_ctx = pipe.detect("q", "Water boils at 100 degrees.", context=None)
    assert no_ctx.status == ExecutionStatus.DEGRADED
    assert no_ctx.next_action == NextAction.VERIFY
    assert no_ctx.hallucination_probability is None
    assert no_ctx.degraded_reason == "no_signal_available"
