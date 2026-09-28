"""Run a detector over the local eval sets and assemble a metrics report.

Reports **per dataset separately** (never a single combined accuracy that hides
cross-dataset weakness), plus a combined block and a macro-average across
datasets. Two prediction views:
  * ``gate``          — the actual routing decision (next_action == Verify -> 1),
                        which is what HalluciGuard experiences.
  * ``threshold@0.50``— score >= 0.50 over completed rows, for apples-to-apples
                        comparison with V1's eval_report.json.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ..datasets.local_eval import EvalRow, LOCAL_DATASETS, load_local_dataset
from .metrics import (
    classification_metrics,
    latency_summary,
    recall_at_routing_budget,
    routing_curve,
    score_metrics,
)

DetectFn = Callable[[str, str, Optional[str]], Any]  # (query, response, context) -> output


def _score_and_pred(output) -> Dict[str, Any]:
    """Pull (label-space) score, routing decision, latency, degraded flag from a
    DetectorV2Output (or any object exposing the same attributes)."""
    status = getattr(output, "status", "completed")
    status = status.value if hasattr(status, "value") else status
    prob = getattr(output, "hallucination_probability", None)
    action = getattr(output, "next_action", None)
    action = action.value if hasattr(action, "value") else action
    return {
        "score": None if (status != "completed" or prob is None) else float(prob),
        "routed": 1 if action == "Verify" else 0,
        "latency_ms": getattr(output, "latency_ms", None),
        "degraded": status != "completed",
    }


def evaluate_detector(detect_output: DetectFn, rows: List[EvalRow]) -> Dict[str, Any]:
    y_true: List[int] = []
    scores: List[Optional[float]] = []
    routed: List[int] = []
    latencies: List[float] = []
    degraded = 0

    for r in rows:
        out = detect_output(r.user_prompt, r.llm_response, r.context)
        info = _score_and_pred(out)
        y_true.append(r.label)
        scores.append(info["score"])
        routed.append(info["routed"])
        if info["latency_ms"] is not None:
            latencies.append(info["latency_ms"])
        degraded += int(info["degraded"])

    # threshold@0.50 view over completed rows only.
    thr_true = [t for t, s in zip(y_true, scores) if s is not None]
    thr_pred = [1 if s >= 0.50 else 0 for s in scores if s is not None]

    curve = routing_curve(y_true, scores)
    return {
        "n": len(rows),
        "positives": sum(y_true),
        "negatives": len(y_true) - sum(y_true),
        "degraded": degraded,
        "gate": classification_metrics(y_true, routed),
        "threshold@0.50": classification_metrics(thr_true, thr_pred) if thr_true else {},
        "score_metrics": score_metrics(y_true, scores),
        "latency": latency_summary(latencies),
        "routing_budgets": {
            f"<= {int(b*100)}%": recall_at_routing_budget(curve, b) for b in (0.3, 0.5, 0.7)
        },
        "routing_curve": curve,
    }


def evaluate_all(detect_output: DetectFn, datasets: Optional[List[str]] = None) -> Dict[str, Any]:
    names = datasets or list(LOCAL_DATASETS)
    per_dataset: Dict[str, Any] = {}
    all_rows: List[EvalRow] = []
    for name in names:
        rows = load_local_dataset(name)
        per_dataset[name] = evaluate_detector(detect_output, rows)
        all_rows.extend(rows)

    combined = evaluate_detector(detect_output, all_rows)
    # Macro-average of gate-F1/recall across datasets (generalization headline).
    macro = {
        "gate_f1": sum(per_dataset[n]["gate"]["f1"] for n in names) / len(names),
        "gate_recall": sum(per_dataset[n]["gate"]["recall"] for n in names) / len(names),
    }
    return {"per_dataset": per_dataset, "combined": combined, "macro_avg": macro}
