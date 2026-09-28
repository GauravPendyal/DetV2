from detector_v2.pipeline import DetectorV2Pipeline
from detector_v2.schemas import ExecutionStatus, NextAction, RiskLevel


def test_normal_response_completed_and_bounded():
    p = DetectorV2Pipeline()
    out = p.detect("What is the capital of France?", "Paris is the capital of France.")
    assert out.status == ExecutionStatus.COMPLETED.value
    assert 0.0 <= out.hallucination_probability <= 1.0
    assert 0.5 <= out.confidence_score <= 1.0
    assert out.calibrated is False  # Stage 0 emits uncalibrated risk
    assert len(out.claims) >= 1
    assert out.latency_ms is not None


def test_empty_response_is_degraded_and_routes_to_verify():
    p = DetectorV2Pipeline()
    out = p.detect("q", "   ")
    assert out.status == ExecutionStatus.DEGRADED.value
    assert out.risk_level == RiskLevel.HIGH.value
    assert out.next_action == NextAction.VERIFY.value
    # Never fabricate a low-risk number on a non-assessment.
    assert out.hallucination_probability is None
    assert out.confidence_score is None
    assert out.degraded_reason == "empty_response"


def test_risk_level_matches_thresholds():
    p = DetectorV2Pipeline()
    out = p.detect("q", "The sky is blue today and the weather feels calm.")
    p_ = out.hallucination_probability
    if p_ <= p.config.low_risk_threshold:
        assert out.risk_level == RiskLevel.LOW.value
        assert out.next_action == NextAction.ACCEPT.value
    elif p_ >= p.config.high_risk_threshold:
        assert out.risk_level == RiskLevel.HIGH.value
    else:
        assert out.risk_level == RiskLevel.MEDIUM.value


def test_pipeline_never_raises_on_odd_input():
    p = DetectorV2Pipeline()
    for weird in ["​​", "123456", "!!!", "a" * 5000]:
        out = p.detect("q", weird)
        assert out.status in {s.value for s in ExecutionStatus}
