"""Assemble the Stage-7 calibrated agent: encoder core (+ optional entailment)
with a fitted calibrator and the tuned routing threshold wired into the gate.

Kept out of the tools so the eval script, the smoke test, and the unit tests all
build the exact same object.
"""
from __future__ import annotations

from typing import Optional

from ..agent import DetectorAgent
from ..config import DetectorV2Config
from ..pipeline import DetectorV2Pipeline
from .calibrator import ProbabilityCalibrator


def build_stage7_agent(
    encoder_model_dir: str,
    calibrator: ProbabilityCalibrator,
    *,
    entailment_model_dir: Optional[str] = None,
    stage: str = "stage7_calibrated",
) -> DetectorAgent:
    """Encoder is the always-on core signal; entailment is optional (added only
    if a model dir is given). The calibrator's tuned threshold becomes the
    Accept/Verify boundary (``low_risk_threshold``)."""
    from ..signals.encoder import EncoderSignal

    t = float(calibrator.threshold)
    config = DetectorV2Config(
        stage=stage, low_risk_threshold=t, high_risk_threshold=max(t, 0.50)
    )
    signals = [EncoderSignal(model_dir=encoder_model_dir)]
    if entailment_model_dir:
        from ..signals.entailment import EntailmentSignal

        signals.append(EntailmentSignal(model_dir=entailment_model_dir))
    pipeline = DetectorV2Pipeline(config=config, signals=signals, calibrator=calibrator)
    return DetectorAgent(pipeline=pipeline)
