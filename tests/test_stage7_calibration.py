"""Stage 7 tests: calibrator behavior + calibrated pipeline routing/contract.

Uses lightweight stub signals/calibrator so the suite stays fast and
deterministic (no encoder load); the real calibrator is unit-tested directly.
"""
from __future__ import annotations

import numpy as np

from detector_v2.adapter import to_canonical_dict
from detector_v2.calibration.calibrator import ProbabilityCalibrator
from detector_v2.config import DetectorV2Config
from detector_v2.pipeline import DetectorV2Pipeline
from detector_v2.schemas import ExecutionStatus, NextAction, RiskLevel, SignalResult
from detector_v2.signals.base import Signal

CANONICAL_KEYS = {
    "hallucination_probability", "confidence_score", "risk_level",
    "next_action", "model_source", "status",
}


# -- stubs ---------------------------------------------------------------
class _EncoderLike(Signal):
    name = "encoder"

    def __init__(self, value: float):
        self._v = value

    def score(self, claim, ctx):
        return SignalResult(name=self.name, available=True, value=self._v)


class _EntailmentLike(Signal):  # optional: abstains honestly when no context
    name = "entailment"

    def score(self, claim, ctx):
        if not ctx.context:
            return self.unavailable("no_context")
        return SignalResult(name=self.name, available=True, value=0.5)


class _DeadSignal(Signal):
    name = "encoder"

    def score(self, claim, ctx):
        return self.unavailable("model_missing")


class _StubCalibrator:
    """Fixed mapping so routing logic is deterministic; confidence uses the same
    threshold-relative rule as the real calibrator (distinct from probability)."""

    def __init__(self, mapped: float, threshold: float):
        self.mapped = mapped
        self.threshold = threshold
        self.method = "stub"
        self.metadata = {}

    def transform(self, risk):
        return self.mapped

    def decision_confidence(self, p, threshold=None):
        t = self.threshold if threshold is None else threshold
        c = (p - t) / max(1.0 - t, 1e-9) if p >= t else (t - p) / max(t, 1e-9)
        return min(1.0, max(0.0, c))


def _pipeline(signals, mapped, threshold):
    cal = _StubCalibrator(mapped, threshold)
    cfg = DetectorV2Config(stage="stage7_calibrated", low_risk_threshold=threshold,
                           high_risk_threshold=max(threshold, 0.5))
    return DetectorV2Pipeline(config=cfg, signals=signals, calibrator=cal)


# -- calibrated pipeline: routing paths ----------------------------------
def test_pass_path_low_prob_accepts_and_is_calibrated():
    out = _pipeline([_EncoderLike(0.8)], mapped=0.05, threshold=0.40).detect("q", "A factual sentence.")
    assert out.status == ExecutionStatus.COMPLETED.value
    assert out.calibrated is True
    assert out.hallucination_probability == 0.05
    assert out.risk_level == RiskLevel.LOW.value
    assert out.next_action == NextAction.ACCEPT.value


def test_verify_path_high_prob_routes_to_verify():
    out = _pipeline([_EncoderLike(0.2)], mapped=0.90, threshold=0.40).detect("q", "A factual sentence.")
    assert out.status == ExecutionStatus.COMPLETED.value
    assert out.calibrated is True
    assert out.next_action == NextAction.VERIFY.value
    assert out.risk_level == RiskLevel.HIGH.value


def test_threshold_moves_the_accept_verify_boundary():
    # Same calibrated prob (0.45); only the threshold differs -> decision flips.
    strict = _pipeline([_EncoderLike(0.5)], mapped=0.45, threshold=0.40).detect("q", "Some sentence here.")
    lax = _pipeline([_EncoderLike(0.5)], mapped=0.45, threshold=0.60).detect("q", "Some sentence here.")
    assert strict.next_action == NextAction.VERIFY.value
    assert lax.next_action == NextAction.ACCEPT.value


def test_missing_context_keeps_encoder_core_and_completes():
    # Entailment abstains without context; encoder still carries -> not degraded.
    out = _pipeline([_EncoderLike(0.3), _EntailmentLike()], mapped=0.1, threshold=0.4).detect(
        "q", "A grounded-looking sentence.", context=None)
    assert out.status == ExecutionStatus.COMPLETED.value
    ent = [s for c in out.claims for s in c.signals if s.name == "entailment"]
    assert ent and all(s.available is False for s in ent)  # honest abstention, not fabricated


def test_degraded_model_routes_to_verify_without_fabricating():
    out = _pipeline([_DeadSignal()], mapped=0.1, threshold=0.4).detect("q", "A sentence.")
    assert out.status == ExecutionStatus.DEGRADED.value
    assert out.next_action == NextAction.VERIFY.value
    assert out.risk_level == RiskLevel.HIGH.value
    assert out.hallucination_probability is None
    assert out.calibrated is False
    assert out.degraded_reason == "no_signal_available"


# -- calibrated pipeline: contract + probability range -------------------
def test_calibrated_probability_in_range_and_canonical_contract():
    out = _pipeline([_EncoderLike(0.4)], mapped=0.73, threshold=0.4).detect("q", "A sentence to score.")
    assert out.calibrated is True
    assert 0.0 <= out.hallucination_probability <= 1.0
    assert 0.0 <= out.confidence_score <= 1.0
    d = to_canonical_dict(out)
    assert set(d.keys()) == CANONICAL_KEYS


def test_confidence_is_not_a_copy_of_probability():
    # prob=0.9 far above threshold 0.4 -> decisive: confidence != probability.
    out = _pipeline([_EncoderLike(0.5)], mapped=0.90, threshold=0.40).detect("q", "A sentence to score.")
    assert abs(out.confidence_score - out.hallucination_probability) > 1e-6


# -- real calibrator unit tests ------------------------------------------
def test_isotonic_fit_is_monotonic_and_bounded():
    rng = np.random.RandomState(0)
    raw = np.linspace(0, 1, 200)
    y = (raw + rng.normal(0, 0.15, size=raw.shape) > 0.5).astype(int)
    cal = ProbabilityCalibrator.fit(raw, y, method="isotonic", threshold=0.4)
    p = cal.transform_many(np.linspace(0, 1, 50))
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert np.all(np.diff(p) >= -1e-9)  # non-decreasing


def test_sigmoid_fit_outputs_probabilities():
    rng = np.random.RandomState(1)
    raw = np.linspace(0, 1, 200)
    y = (raw + rng.normal(0, 0.1, size=raw.shape) > 0.5).astype(int)
    cal = ProbabilityCalibrator.fit(raw, y, method="sigmoid", threshold=0.5)
    p = cal.transform_many(raw)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_decision_confidence_zero_at_threshold_one_at_extremes():
    cal = ProbabilityCalibrator.fit([0.0, 1.0], [0, 1], method="isotonic", threshold=0.4)
    assert abs(cal.decision_confidence(0.4)) < 1e-9        # on the fence
    assert abs(cal.decision_confidence(1.0) - 1.0) < 1e-9  # decisive Verify
    assert abs(cal.decision_confidence(0.0) - 1.0) < 1e-9  # decisive Accept
    mid = cal.decision_confidence(0.7)
    assert 0.0 < mid < 1.0


def test_calibrator_save_load_roundtrip(tmp_path):
    raw = np.linspace(0, 1, 100)
    y = (raw > 0.5).astype(int)
    cal = ProbabilityCalibrator.fit(raw, y, method="isotonic", threshold=0.37,
                                    metadata={"core_signal": "stage2_encoder"})
    path = str(tmp_path / "calibrator.joblib")
    cal.save(path)
    loaded = ProbabilityCalibrator.load(path)
    assert loaded.method == "isotonic"
    assert loaded.threshold == 0.37
    assert loaded.metadata["core_signal"] == "stage2_encoder"
    probe = [0.1, 0.5, 0.9]
    assert np.allclose(loaded.transform_many(probe), cal.transform_many(probe))
