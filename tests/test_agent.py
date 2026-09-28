from detector_v2 import DetectorAgent


def test_detect_returns_canonical_plus_claims():
    agent = DetectorAgent()
    r = agent.detect("What is the capital of France?", "Paris is the capital of France.")
    for key in ("hallucination_probability", "confidence_score", "risk_level",
                "next_action", "model_source", "status"):
        assert key in r
    assert r["model_source"].startswith("detector_v2_")
    assert isinstance(r["claims"], list)


def test_detect_canonical_is_strict_six_fields():
    agent = DetectorAgent()
    r = agent.detect_canonical("q", "Some response text about a topic.")
    assert set(r.keys()) == {
        "hallucination_probability", "confidence_score", "risk_level",
        "next_action", "model_source", "status",
    }


def test_pipeline_is_built_once_and_reused():
    agent = DetectorAgent()
    first = agent.pipeline
    agent.detect("q", "one.")
    agent.detect("q", "two.")
    # Same pipeline object across calls -> no per-inference reconstruction.
    assert agent.pipeline is first
