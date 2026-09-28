"""Stage 7 probability calibrator + operating threshold.

Wraps the selected core detector's *uncalibrated* response-risk and maps it to a
calibrated P(hallucinated) via a leakage-safe 1-D fit, and carries the tuned
routing threshold. The calibrator is fit on a held-out split the core signal
never trained on (HaluEval ``summarization``) — NEVER on the 769-row local eval
set (that stays the final, untouched test set).

``confidence_score`` is deliberately NOT a copy of the probability. It is the
decision-distance from the operating threshold, normalized per side to [0, 1]:
high when the calibrated probability sits far from the Accept/Verify boundary
(a decisive call), low when it sits on the fence. So a calibrated P near 0.0 or
1.0 is *confident*, and a P near the threshold is *not* — which a raw copy of P
could never express.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else float(v)


class ProbabilityCalibrator:
    """1-D leakage-safe calibrator over a core signal's response-risk.

    ``method`` is 'isotonic' (monotonic, non-parametric reshaping — good when the
    raw scores are mis-scaled/saturated) or 'sigmoid' (Platt scaling).
    """

    def __init__(self, method: str, model, threshold: float, metadata: Optional[Dict] = None):
        self.method = method
        self._model = model
        self.threshold = float(threshold)
        self.metadata = metadata or {}

    # -- fit ------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        raw_scores: Sequence[float],
        labels: Sequence[int],
        method: str = "isotonic",
        threshold: float = 0.5,
        metadata: Optional[Dict] = None,
    ) -> "ProbabilityCalibrator":
        X = np.asarray(raw_scores, dtype=float).reshape(-1)
        y = np.asarray(labels, dtype=int).reshape(-1)
        if method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(X, y)
        elif method == "sigmoid":
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(C=1e6, solver="lbfgs")
            model.fit(X.reshape(-1, 1), y)
        else:  # pragma: no cover - guarded by callers
            raise ValueError(f"unknown calibration method: {method}")
        return cls(method, model, threshold, metadata)

    # -- transform ------------------------------------------------------
    def transform_many(self, risks: Sequence[float]) -> np.ndarray:
        X = np.asarray(risks, dtype=float).reshape(-1)
        if self.method == "isotonic":
            p = self._model.predict(X)
        else:
            p = self._model.predict_proba(X.reshape(-1, 1))[:, 1]
        return np.clip(p, 0.0, 1.0)

    def transform(self, risk: float) -> float:
        return _clamp01(float(self.transform_many([risk])[0]))

    # -- confidence (NOT a copy of the probability) ---------------------
    def decision_confidence(self, p: float, threshold: Optional[float] = None) -> float:
        """Normalized distance of the calibrated P from the routing boundary.

        Returns 0 exactly at the threshold (a coin-flip decision) and rises to 1
        at the extremes of whichever side P falls on. This is an orthogonal
        *decisiveness* axis, independent of the risk magnitude itself.
        """
        t = self.threshold if threshold is None else float(threshold)
        if p >= t:
            c = (p - t) / max(1.0 - t, 1e-9)
        else:
            c = (t - p) / max(t, 1e-9)
        return _clamp01(c)

    # -- persistence ----------------------------------------------------
    def save(self, path: str) -> None:
        import joblib

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(
            {"method": self.method, "model": self._model,
             "threshold": self.threshold, "metadata": self.metadata},
            path,
        )
        with open(os.path.splitext(path)[0] + ".meta.json", "w", encoding="utf-8") as fh:
            json.dump({"method": self.method, "threshold": self.threshold,
                       "metadata": self.metadata}, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "ProbabilityCalibrator":
        import joblib

        blob = joblib.load(path)
        return cls(blob["method"], blob["model"], blob["threshold"], blob.get("metadata"))
