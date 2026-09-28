from detector_v2.config import DetectorV2Config
from detector_v2.pipeline import DetectorV2Pipeline


def _long_response(n=200):
    return " ".join(
        f"In the year {1900 + (i % 120)} the committee number {i} reported a distinct finding."
        for i in range(n)
    )


def test_long_response_completes_and_caps_claims():
    cfg = DetectorV2Config(max_claims=64)
    p = DetectorV2Pipeline(config=cfg)
    out = p.detect("Summarize the report.", _long_response(200))
    assert out.status == "completed"
    assert len(out.claims) <= 64
    assert out.latency_ms is not None


def test_long_context_argument_does_not_break():
    p = DetectorV2Pipeline()
    long_ctx = "Background document. " * 2000
    out = p.detect("q", "The report concludes with three key points.", context=long_ctx)
    assert out.status == "completed"
    assert 0.0 <= out.hallucination_probability <= 1.0


def test_latency_recorded_and_finite():
    p = DetectorV2Pipeline()
    out = p.detect("q", _long_response(50))
    assert out.latency_ms >= 0.0
