import os
import tempfile

import numpy as np

from detector_v2.agent import DetectorAgent
from detector_v2.features.response import feature_names
from detector_v2.pipeline import DetectorV2Pipeline
from detector_v2.scoring.logreg import LogRegResponseModel, LogRegResponseScorer


def _synthetic(n=200, seed=0):
    """Feature 2 (struct_max) linearly separates the classes."""
    rng = np.random.RandomState(seed)
    d = len(feature_names())
    X = rng.rand(n, d)
    y = (X[:, 3] > 0.5).astype(int)  # struct_max
    return X, y


def test_train_predict_proba_in_range():
    X, y = _synthetic()
    model = LogRegResponseModel().fit(X, y)
    p = model.predict_proba(X)
    assert p.shape == (len(y),)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_learns_separable_signal():
    X, y = _synthetic()
    model = LogRegResponseModel().fit(X, y)
    pred = (model.predict_proba(X) >= 0.5).astype(int)
    acc = (pred == y).mean()
    assert acc > 0.9


def test_save_load_roundtrip(tmp_path):
    X, y = _synthetic()
    model = LogRegResponseModel().fit(X, y)
    path = os.path.join(tmp_path, "m.joblib")
    model.save(path)
    loaded = LogRegResponseModel.load(path)
    assert np.allclose(model.predict_proba(X), loaded.predict_proba(X))
    assert os.path.exists(os.path.splitext(path)[0] + ".meta.json")


def test_scorer_integrates_with_pipeline(tmp_path):
    X, y = _synthetic()
    model = LogRegResponseModel().fit(X, y)
    pipeline = DetectorV2Pipeline(response_scorer=LogRegResponseScorer(model))
    agent = DetectorAgent(pipeline=pipeline)
    r = agent.detect("q", "The company definitely earned exactly 4.2 billion dollars in 1999.")
    assert r["status"] == "completed"
    assert 0.0 <= r["hallucination_probability"] <= 1.0
    assert r["model_source"].startswith("detector_v2_")
