"""Adversarial checks (INVITE-style confident false premises).

Stage 0 is a deterministic heuristic floor, so these assert *relative* behaviour
the structural signal is designed to produce — not absolute accuracy, which the
learned stages must earn. They also encode the safe-default routing contract.
"""
from detector_v2.pipeline import DetectorV2Pipeline
from detector_v2.schemas import NextAction

CONFIDENT_FALSE = (
    "As we all know, the largest planet is definitely Mercury, which is always "
    "the closest to the sun and undoubtedly the hottest in every single case."
)
HEDGED_BENIGN = "This might be a somewhat helpful and fairly general overview of the topic."


def test_confident_false_premise_scores_higher_than_hedged_benign():
    p = DetectorV2Pipeline()
    hot = p.detect("q", CONFIDENT_FALSE)
    cool = p.detect("q", HEDGED_BENIGN)
    assert hot.hallucination_probability > cool.hallucination_probability


def test_confident_false_premise_routes_to_verify():
    p = DetectorV2Pipeline()
    out = p.detect("q", CONFIDENT_FALSE)
    # A response this loaded with red flags must not be silently Accepted.
    assert out.next_action == NextAction.VERIFY.value


def test_false_premise_cue_recorded_in_claim_features():
    p = DetectorV2Pipeline()
    out = p.detect("q", CONFIDENT_FALSE)
    fp = 0.0
    for claim in out.claims:
        for sig in claim.signals:
            if sig.name == "structural":
                fp += sig.features.get("false_premise_cue_count", 0.0)
    assert fp >= 1.0
