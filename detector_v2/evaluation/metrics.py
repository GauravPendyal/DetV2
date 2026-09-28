"""Evaluation metrics for the detector.

Reports every metric the brief requires: precision / recall / F1, AUROC, AUPRC,
TP/FP/TN/FN, ECE, Brier, latency, and the HalluciGuard-specific
*hallucination-recall vs %-routed-to-Verifier* curve.

Score-based metrics (AUROC/AUPRC/Brier/ECE) are computed only over rows that
produced a numeric score (status=completed); degraded rows are counted
separately and always treated as routed-to-Verify in the routing/confusion view.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

try:  # sklearn is a Stage-0 dependency, but degrade gracefully if absent.
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    _SK = True
except Exception:  # noqa: BLE001
    _SK = False


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    c = confusion(y_true, y_pred)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, **c}


def expected_calibration_error(y_true: Sequence[int], scores: Sequence[float], n_bins: int = 10) -> float:
    if not scores:
        return float("nan")
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(scores, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def score_metrics(y_true: Sequence[int], scores: Sequence[Optional[float]]) -> Dict[str, float]:
    """AUROC/AUPRC/Brier/ECE over rows with a numeric score only."""
    pairs = [(t, s) for t, s in zip(y_true, scores) if s is not None]
    out: Dict[str, float] = {"scored_n": len(pairs)}
    if not pairs:
        return {**out, "auroc": float("nan"), "auprc": float("nan"),
                "brier": float("nan"), "ece": float("nan")}
    yt = [t for t, _ in pairs]
    ss = [float(s) for _, s in pairs]
    single_class = len(set(yt)) < 2
    if _SK and not single_class:
        out["auroc"] = float(roc_auc_score(yt, ss))
        out["auprc"] = float(average_precision_score(yt, ss))
        out["brier"] = float(brier_score_loss(yt, ss))
    else:
        out["auroc"] = float("nan") if single_class else _auroc_fallback(yt, ss)
        out["auprc"] = float("nan")
        out["brier"] = float(np.mean((np.asarray(ss) - np.asarray(yt, dtype=float)) ** 2))
    out["ece"] = expected_calibration_error(yt, ss)
    return out


def _auroc_fallback(y_true: Sequence[int], scores: Sequence[float]) -> float:
    # Rank-based (Mann-Whitney) AUROC when sklearn is unavailable.
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    y = np.asarray(y_true)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def routing_curve(
    y_true: Sequence[int],
    scores: Sequence[Optional[float]],
    thresholds: Optional[Sequence[float]] = None,
) -> List[Dict[str, float]]:
    """Hallucination recall vs fraction of responses routed to the Verifier.

    A response is routed when it is degraded (score None -> always routed) or its
    score >= threshold. This is the core cost/recall trade-off for the gate.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.linspace(0.0, 1.0, 21)]
    y = np.asarray(y_true)
    n = len(y)
    n_pos = int(y.sum()) or 1
    curve: List[Dict[str, float]] = []
    for thr in thresholds:
        routed = np.array([(s is None) or (float(s) >= thr) for s in scores])
        frac_routed = float(routed.sum() / n) if n else 0.0
        recall = float((routed & (y == 1)).sum() / n_pos)
        curve.append({"threshold": float(thr), "frac_routed": frac_routed, "hallucination_recall": recall})
    return curve


def recall_at_routing_budget(curve: List[Dict[str, float]], budget: float) -> Dict[str, float]:
    """Best hallucination recall achievable while routing <= ``budget`` of traffic."""
    feasible = [pt for pt in curve if pt["frac_routed"] <= budget]
    if not feasible:
        return {"budget": budget, "hallucination_recall": 0.0, "frac_routed": 0.0, "threshold": 1.0}
    best = max(feasible, key=lambda pt: pt["hallucination_recall"])
    return {"budget": budget, **best}


def latency_summary(latencies_ms: Sequence[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {"n": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    arr = np.asarray(latencies_ms, dtype=float)
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
    }
