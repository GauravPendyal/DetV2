import os

import numpy as np

from detector_v2.features.fusion import (
    CONFIGS,
    SignalComponents,
    _agg3,
    build_training_matrix,
    build_vector,
    components_from_claimsignals,
    components_from_probs,
    compute_impute,
    fusion_feature_names,
    subset_available,
)
from detector_v2.schemas import ClaimSignal, SignalResult
from detector_v2.scoring.fusion import FusionResponseModel, FusionResponseScorer
# --- feature layout -------------------------------------------------------
def test_feature_names_per_config():
    assert fusion_feature_names(CONFIGS["A"]) == ["s1_prob"]
    assert fusion_feature_names(CONFIGS["B"]) == ["s2_mean", "s2_max", "s2_topk"]
    assert fusion_feature_names(CONFIGS["C"]) == ["s3_mean", "s3_max", "s3_topk", "s3_avail"]
    assert fusion_feature_names(CONFIGS["G"]) == [
        "s1_prob", "s2_mean", "s2_max", "s2_topk", "s3_mean", "s3_max", "s3_topk", "s3_avail",
    ]
    # canonical order is enforced regardless of subset spelling order
    assert fusion_feature_names(["entailment", "structural"]) == [
        "s1_prob", "s3_mean", "s3_max", "s3_topk", "s3_avail",
    ]


def test_agg3_mean_max_topk():
    mean, mx, topk = _agg3([0.1, 0.9, 0.5], k=2)
    assert abs(mean - 0.5) < 1e-9
    assert mx == 0.9
    assert abs(topk - 0.7) < 1e-9  # mean of top-2 = (0.9 + 0.5)/2


# --- imputation + availability flag --------------------------------------
def _impute():
    return {"s1_prob": 0.4, "s2_mean": 0.3, "s2_max": 0.6, "s2_topk": 0.5,
            "s3_mean": 0.35, "s3_max": 0.7, "s3_topk": 0.55}


def test_entailment_present_sets_flag_one():
    comp = components_from_probs(0.8, [0.2, 0.4], [0.1, 0.9], k=3)
    vec = build_vector(CONFIGS["G"], comp, _impute())
    assert vec[-1] == 1.0  # s3_avail present


def test_entailment_absent_imputes_with_flag_zero():
    comp = components_from_probs(0.8, [0.2, 0.4], None, k=3)  # no context -> no entailment
    assert comp.s3_available is False
    vec = build_vector(CONFIGS["G"], comp, _impute())
    # entailment block = imputed means, flag 0
    assert list(vec[-4:]) == [0.35, 0.7, 0.55, 0.0]
    # structural + encoder untouched
    assert vec[0] == 0.8


def test_subset_available_degrades_when_all_blocks_missing():
    comp = components_from_probs(None, None, None)  # nothing available
    assert subset_available(CONFIGS["C"], comp) is False
    assert subset_available(CONFIGS["G"], comp) is False
    comp2 = components_from_probs(0.5, [0.3], None)
    assert subset_available(CONFIGS["C"], comp2) is False  # entailment-only, no entailment
    assert subset_available(CONFIGS["G"], comp2) is True   # structural/encoder carry it


# --- missingness augmentation --------------------------------------------
def test_augmentation_produces_both_regimes():
    comps = [components_from_probs(0.5, [0.4], [0.6]) for _ in range(5)]
    labels = [1, 0, 1, 0, 1]
    impute = compute_impute(comps)
    X, y = build_training_matrix(CONFIGS["G"], comps, labels, impute)
    assert X.shape[0] == 2 * len(comps)  # present + masked duplicate each
    assert y.shape[0] == 2 * len(comps)
    flags = X[:, -1]  # s3_avail column
    assert set(flags.tolist()) == {0.0, 1.0}  # both regimes present


def test_augmentation_skipped_for_single_entailment_and_non_entailment():
    comps = [components_from_probs(0.5, [0.4], [0.6]) for _ in range(4)]
    labels = [1, 0, 1, 0]
    impute = compute_impute(comps)
    Xc, _ = build_training_matrix(CONFIGS["C"], comps, labels, impute)  # entailment only
    Xd, _ = build_training_matrix(CONFIGS["D"], comps, labels, impute)  # structural+encoder
    assert Xc.shape[0] == len(comps)
    assert Xd.shape[0] == len(comps)


# --- model roundtrip ------------------------------------------------------
def _fit_toy(subset):
    rng = np.random.RandomState(0)
    comps, labels = [], []
    for _ in range(120):
        y = rng.randint(2)
        s1 = 0.7 + 0.2 * rng.rand() if y else 0.2 + 0.2 * rng.rand()
        comps.append(components_from_probs(s1, [rng.rand()], [rng.rand()]))
        labels.append(y)
    impute = compute_impute(comps)
    X, y = build_training_matrix(subset, comps, labels, impute)
    m = FusionResponseModel(subset)
    m.impute = impute
    m.fit(X, y)
    return m, comps, labels, impute


def test_model_fit_predict_range_and_weights():
    m, comps, labels, impute = _fit_toy(CONFIGS["G"])
    X, _ = build_training_matrix(CONFIGS["G"], comps, labels, impute)
    p = m.predict_proba(X)
    assert p.min() >= 0.0 and p.max() <= 1.0
    w = m.weights()
    assert "s1_prob" in w and "__intercept__" in w


def test_model_save_load_roundtrip(tmp_path):
    m, comps, labels, impute = _fit_toy(CONFIGS["G"])
    path = os.path.join(tmp_path, "G.joblib")
    m.save(path)
    loaded = FusionResponseModel.load(path)
    X, _ = build_training_matrix(CONFIGS["G"], comps, labels, impute)
    assert np.allclose(m.predict_proba(X), loaded.predict_proba(X))
    assert loaded.subset == list(CONFIGS["G"])
    assert loaded.impute == impute
    assert os.path.exists(os.path.splitext(path)[0] + ".meta.json")


# --- scorer parity: vector from ClaimSignals == vector from components ----
class _StubStage1:
    """Stand-in Stage-1 model returning a fixed prob, so we test the plumbing."""

    def __init__(self, prob):
        self.prob = prob

    def predict_proba(self, X):
        return np.asarray([self.prob])


def _claimsignal(cid, enc, ent):
    signals = [SignalResult(name="structural", available=True, value=0.5, features={})]
    if enc is not None:
        signals.append(SignalResult(name="encoder", available=True, value=enc,
                                    features={"encoder_prob": enc}))
    if ent is not None:
        signals.append(SignalResult(name="entailment", available=True, value=ent,
                                    features={"entailment_unsupported": ent}))
    return ClaimSignal(claim_id=cid, text=f"claim {cid}", span=[0, 5],
                       signals=signals, claim_risk=0.5)


def test_scorer_matches_component_vector():
    m, comps, labels, impute = _fit_toy(CONFIGS["G"])
    s1_prob = 0.65
    claims = [_claimsignal(0, 0.4, 0.6), _claimsignal(1, 0.2, 0.8)]

    # Reference: build components directly from the same raw probs.
    comp_ref = components_from_probs(s1_prob, [0.4, 0.2], [0.6, 0.8])
    x_ref = build_vector(CONFIGS["G"], comp_ref, impute).reshape(1, -1)
    expected = float(m.predict_proba(x_ref)[0])

    scorer = FusionResponseScorer(m, _StubStage1(s1_prob))
    got = scorer.score("q", "resp", claims)
    assert abs(got - expected) < 1e-9


def test_scorer_imputes_entailment_when_absent():
    m, comps, labels, impute = _fit_toy(CONFIGS["G"])
    s1_prob = 0.65
    claims = [_claimsignal(0, 0.4, None), _claimsignal(1, 0.2, None)]  # no entailment
    comp_ref = components_from_probs(s1_prob, [0.4, 0.2], None)
    x_ref = build_vector(CONFIGS["G"], comp_ref, impute).reshape(1, -1)
    expected = float(m.predict_proba(x_ref)[0])
    scorer = FusionResponseScorer(m, _StubStage1(s1_prob))
    assert abs(scorer.score("q", "resp", claims) - expected) < 1e-9
