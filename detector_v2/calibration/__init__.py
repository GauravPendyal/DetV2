"""Probability calibration (Stage 7).

Provides a leakage-safe 1-D calibrator (isotonic / Platt) over the selected core
signal's response-risk, plus the tuned routing threshold. A pipeline given a
fitted calibrator emits a calibrated P(hallucinated) and sets
``DetectorV2Output.calibrated=True``; without one, the score stays an
uncalibrated risk and ``calibrated`` stays False.
"""
from .calibrator import ProbabilityCalibrator  # noqa: F401

__all__ = ["ProbabilityCalibrator"]
