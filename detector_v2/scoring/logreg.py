"""Stage-1 logistic-regression response model + scorer.

A low-capacity, fully inspectable learned fusion over the Stage-0 structural
features. Its purpose is to replace the hand-set heuristic weights with weights
*learned* from data, and to establish the train/eval harness — not to be the
final detector (the DeBERTa encoder in Stage 2 is the real learned signal).

Standardization + L2 logistic regression is deliberately simple so the
V0-vs-Stage1 improvement is attributable to *learning the weights*, nothing else.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..features.response import feature_names, featurize_claims
from ..schemas import ClaimSignal
from .response import ResponseScorer


class LogRegResponseModel:
    def __init__(self, C: float = 1.0, class_weight: Optional[str] = "balanced"):
        self.feature_names = feature_names()
        self.pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=C, class_weight=class_weight, max_iter=2000)),
        ])
        self.metadata: dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogRegResponseModel":
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(hallucinated) for each row."""
        return self.pipeline.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "feature_names": self.feature_names,
                     "metadata": self.metadata}, path)
        with open(os.path.splitext(path)[0] + ".meta.json", "w", encoding="utf-8") as fh:
            json.dump({"feature_names": self.feature_names, "metadata": self.metadata}, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "LogRegResponseModel":
        blob = joblib.load(path)
        obj = cls()
        obj.pipeline = blob["pipeline"]
        obj.feature_names = blob["feature_names"]
        obj.metadata = blob.get("metadata", {})
        return obj


class LogRegResponseScorer(ResponseScorer):
    name = "logreg"

    def __init__(self, model: LogRegResponseModel):
        self.model = model

    def score(self, query: str, response: str, claims: List[ClaimSignal]) -> float:
        x = featurize_claims(claims).reshape(1, -1)
        return float(self.model.predict_proba(x)[0])
