import math

from detector_v2.evaluation.metrics import (
    classification_metrics,
    expected_calibration_error,
    latency_summary,
    recall_at_routing_budget,
    routing_curve,
    score_metrics,
)


def test_classification_metrics_known_values():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 1]
    m = classification_metrics(y_true, y_pred)
    assert m["tp"] == 1 and m["fn"] == 1 and m["fp"] == 1 and m["tn"] == 1
    assert abs(m["precision"] - 0.5) < 1e-9
    assert abs(m["recall"] - 0.5) < 1e-9


def test_score_metrics_perfect_separation():
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9]
    m = score_metrics(y, s)
    assert abs(m["auroc"] - 1.0) < 1e-9
    assert m["scored_n"] == 4


def test_score_metrics_handles_none_scores():
    y = [1, 0, 1]
    s = [0.9, None, 0.8]  # one degraded row
    m = score_metrics(y, s)
    assert m["scored_n"] == 2


def test_routing_curve_endpoints():
    y = [1, 0, 1, 0]
    s = [0.9, 0.1, 0.6, 0.2]
    curve = routing_curve(y, s, thresholds=[0.0, 1.01])
    assert curve[0]["frac_routed"] == 1.0        # threshold 0 routes everything
    assert curve[0]["hallucination_recall"] == 1.0
    assert curve[-1]["frac_routed"] == 0.0       # threshold >1 routes nothing


def test_degraded_always_routed_in_curve():
    y = [1, 0]
    s = [None, 0.1]  # degraded positive must count as routed even at high threshold
    curve = routing_curve(y, s, thresholds=[0.99])
    assert curve[0]["hallucination_recall"] == 1.0


def test_recall_at_budget():
    y = [1, 1, 0, 0]
    s = [0.9, 0.4, 0.3, 0.1]
    curve = routing_curve(y, s)
    r = recall_at_routing_budget(curve, 0.5)
    assert r["frac_routed"] <= 0.5


def test_ece_bounds():
    y = [0, 1, 0, 1]
    s = [0.0, 1.0, 0.0, 1.0]  # perfectly calibrated extremes
    assert expected_calibration_error(y, s) < 1e-9


def test_latency_summary():
    ls = latency_summary([1.0, 2.0, 3.0, 4.0])
    assert ls["n"] == 4
    assert ls["mean_ms"] == 2.5
