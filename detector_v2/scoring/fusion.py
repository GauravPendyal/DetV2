"""Stage-6 learned fusion: a logistic regression over the Stage-1/2/3 signal
blocks, plus a response scorer that wires it into the pipeline.

``FusionResponseModel`` mirrors ``LogRegResponseModel`` (StandardScaler ->
LogisticRegression, C=1.0, class_weight="balanced") but additionally carries the
signal ``subset`` it was fit for and the ``impute`` constants for missing blocks,
so a saved model is fully self-describing.

``FusionResponseScorer`` holds the fusion model and the Stage-1 LR model: it
computes the Stage-1 prob from the response's structural features, reads the
encoder/entailment values off the scored ``ClaimSignal``s, builds the SAME
feature vector used in training, and returns the fused risk. This is Stage 6's
only integration point — nothing is wired into live orchestration here.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..features.fusion import (
    build_vector,
    components_from_claimsignals,
    fusion_feature_names,
)
from ..features.response import featurize_claims
from ..schemas import ClaimSignal
from .logreg import LogRegResponseModel
from .response import ResponseScorer


class FusionResponseModel:
    def __init__(self, subset: Sequence[str], C: float = 1.0,
                 class_weight: Optional[str] = "balanced"):
        self.subset = list(subset)
        self.feature_names = fusion_feature_names(self.subset)
        self.pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=C, class_weight=class_weight, max_iter=2000)),
        ])
        self.impute: Dict[str, float] = {}
        self.metadata: dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FusionResponseModel":
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(hallucinated) for each row."""
        return self.pipeline.predict_proba(X)[:, 1]

    def weights(self) -> Dict[str, float]:
        """Learned coefficients keyed by feature name (post-standardization)."""
        lr: LogisticRegression = self.pipeline.named_steps["lr"]
        coef = lr.coef_.ravel().tolist()
        out = {name: float(w) for name, w in zip(self.feature_names, coef)}
        out["__intercept__"] = float(lr.intercept_[0])
        return out

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({"subset": self.subset, "pipeline": self.pipeline,
                     "feature_names": self.feature_names, "impute": self.impute,
                     "metadata": self.metadata}, path)
        with open(os.path.splitext(path)[0] + ".meta.json", "w", encoding="utf-8") as fh:
            json.dump({"subset": self.subset, "feature_names": self.feature_names,
                       "impute": self.impute, "metadata": self.metadata}, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "FusionResponseModel":
        blob = joblib.load(path)
        obj = cls(blob["subset"])
        obj.pipeline = blob["pipeline"]
        obj.feature_names = blob["feature_names"]
        obj.impute = blob.get("impute", {})
        obj.metadata = blob.get("metadata", {})
        return obj


class FusionResponseScorer(ResponseScorer):
    name = "fusion"

    def __init__(self, model: FusionResponseModel, stage1_model: LogRegResponseModel,
                 agg_k: int = 3):
        self.model = model
        self.stage1 = stage1_model
        self.k = agg_k

    def score(self, query: str, response: str, claims: List[ClaimSignal]) -> float:
        s1_prob = float(self.stage1.predict_proba(featurize_claims(claims).reshape(1, -1))[0])
        comp = components_from_claimsignals(claims, s1_prob, k=self.k)
        x = build_vector(self.model.subset, comp, self.model.impute).reshape(1, -1)
        return float(self.model.predict_proba(x)[0])
