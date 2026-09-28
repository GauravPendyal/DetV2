"""Contract tests: V2 must stay byte-compatible with the canonical DetectorResult."""
from detector_v2.adapter import to_canonical_dict, to_canonical_model, to_extended_dict
from detector_v2.pipeline import DetectorV2Pipeline

CANONICAL_KEYS = {
    "hallucination_probability", "confidence_score", "risk_level",
    "next_action", "model_source", "status",
}
RISK_VALUES = {"LOW", "MEDIUM", "HIGH"}
ACTION_VALUES = {"Accept", "Verify"}
STATUS_VALUES = {"completed", "degraded", "failed"}


def _out(resp="Paris is the capital of France."):
    return DetectorV2Pipeline().detect("q", resp)


def test_canonical_dict_has_exactly_six_keys():
    d = to_canonical_dict(_out())
    assert set(d.keys()) == CANONICAL_KEYS


def test_canonical_enum_values_are_strings_in_contract_domain():
    d = to_canonical_dict(_out())
    assert d["risk_level"] in RISK_VALUES
    assert d["next_action"] in ACTION_VALUES
    assert d["status"] in STATUS_VALUES
    assert isinstance(d["hallucination_probability"], float)
    assert isinstance(d["confidence_score"], float)
    assert 0.0 <= d["hallucination_probability"] <= 1.0
    assert 0.0 <= d["confidence_score"] <= 1.0


def test_degraded_emits_zero_confidence_not_fabricated_probability():
    d = to_canonical_dict(_out("   "))  # empty -> degraded
    assert d["status"] == "degraded"
    assert d["confidence_score"] == 0.0
    assert d["hallucination_probability"] == 0.0
    assert d["risk_level"] == "HIGH"
    assert d["next_action"] == "Verify"


def test_extended_dict_is_additive_superset():
    d = to_extended_dict(_out())
    assert CANONICAL_KEYS.issubset(set(d.keys()))
    assert "claims" in d and isinstance(d["claims"], list)
    assert "diagnostics" in d
    # Additive keys must not disturb the canonical six.
    assert set(to_canonical_dict(_out()).keys()) == CANONICAL_KEYS


def test_canonical_model_falls_back_to_dict_when_orchestration_absent():
    m = to_canonical_model(_out())
    # orchestration package is not importable here -> plain dict with six keys.
    assert isinstance(m, dict)
    assert set(m.keys()) == CANONICAL_KEYS
