"""Stage 7 — evaluate the calibrated encoder detector on the held-out local 769,
compare against V1 (read-only), and run the required smoke test.

The 769-row local set is used ONLY here as the final test (never for fit/threshold).
Emits reports/stage7_eval.json.

Usage (from DetV2, venv active):
    python tools/eval_stage7.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detector_v2.calibration.build import build_stage7_agent  # noqa: E402
from detector_v2.calibration.calibrator import ProbabilityCalibrator  # noqa: E402
from detector_v2.datasets.local_eval import LOCAL_DATASETS, load_local_dataset  # noqa: E402
from detector_v2.evaluation.metrics import (  # noqa: E402
    classification_metrics,
    latency_summary,
    score_metrics,
)
from detector_v2.schemas import ExecutionStatus, NextAction  # noqa: E402


def _eval_dataset(agent, rows) -> Dict:
    y: List[int] = []
    scores: List[Optional[float]] = []
    routed: List[int] = []
    lat: List[float] = []
    degraded = 0
    for r in rows:
        out = agent.detect_output(r.user_prompt, r.llm_response, context=r.context)
        y.append(int(r.label))
        scores.append(out.hallucination_probability)
        routed.append(1 if out.next_action == NextAction.VERIFY.value else 0)
        lat.append(out.latency_ms or 0.0)
        if out.status != ExecutionStatus.COMPLETED.value:
            degraded += 1
    n_pos = sum(y)
    caught = sum(1 for yy, rr in zip(y, routed) if yy == 1 and rr == 1)
    return {
        "n": len(y), "positives": n_pos, "negatives": len(y) - n_pos, "degraded": degraded,
        "gate": classification_metrics(y, routed),
        "hallucination_recall": round(caught / n_pos, 4) if n_pos else None,
        "routing_fraction": round(sum(routed) / len(y), 4) if y else None,
        "score_metrics": score_metrics(y, scores),
        "latency": latency_summary(lat),
    }, y, scores, routed, lat


def _load_v1() -> Dict:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                        "Detector_agent", "eval_report.json"))
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return {"path": path, "report": json.load(fh)}
    except Exception as exc:  # noqa: BLE001
        return {"path": path, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder-model", default=os.path.join("models", "stage2_encoder"))
    ap.add_argument("--calibrator", default=os.path.join("models", "stage7_calibration", "calibrator.joblib"))
    ap.add_argument("--report", default=os.path.join("reports", "stage7_eval.json"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    cal = ProbabilityCalibrator.load(os.path.abspath(args.calibrator))
    agent = build_stage7_agent(os.path.abspath(args.encoder_model), cal)
    print(f"stage7 agent: core=encoder method={cal.method} threshold={cal.threshold} "
          f"calibrated=True (entailment optional, not wired in default path)")

    per_dataset: Dict[str, Dict] = {}
    all_y: List[int] = []
    all_scores: List[Optional[float]] = []
    all_routed: List[int] = []
    all_lat: List[float] = []
    for name in LOCAL_DATASETS:
        rows = load_local_dataset(name)
        res, y, scores, routed, lat = _eval_dataset(agent, rows)
        per_dataset[name] = res
        all_y += y; all_scores += scores; all_routed += routed; all_lat += lat
        print(f"  {name:<22} n={res['n']:<4} recall={res['hallucination_recall']} "
              f"routing={res['routing_fraction']} auroc={res['score_metrics'].get('auroc', float('nan')):.3f} "
              f"p95={res['latency']['p95_ms']:.0f}ms")
    n_pos = sum(all_y)
    caught = sum(1 for yy, rr in zip(all_y, all_routed) if yy == 1 and rr == 1)
    combined = {
        "n": len(all_y), "positives": n_pos, "negatives": len(all_y) - n_pos,
        "gate": classification_metrics(all_y, all_routed),
        "hallucination_recall": round(caught / n_pos, 4) if n_pos else None,
        "routing_fraction": round(sum(all_routed) / len(all_y), 4) if all_y else None,
        "score_metrics": score_metrics(all_y, all_scores),
        "latency": latency_summary(all_lat),
    }

    # Smoke test (clearly hallucinated -> expect Verify).
    smoke = agent.detect_canonical(
        "Who painted the Mona Lisa?",
        "The Mona Lisa was painted by Pablo Picasso in 1750.",
    )
    smoke_out = agent.detect_output(
        "Who painted the Mona Lisa?",
        "The Mona Lisa was painted by Pablo Picasso in 1750.",
    )

    v1 = _load_v1()
    report = {
        "core_signal": "stage2_encoder", "calibration_method": cal.method,
        "threshold": cal.threshold, "calibrator_metadata": cal.metadata,
        "per_dataset": per_dataset, "combined_local": combined,
        "smoke_test": {
            "query": "Who painted the Mona Lisa?",
            "response": "The Mona Lisa was painted by Pablo Picasso in 1750.",
            "canonical": smoke,
            "calibrated": smoke_out.calibrated,
            "confidence_score": smoke_out.confidence_score,
        },
        "v1_reference": v1,
        "eval_seconds": round(time.perf_counter() - t0, 1),
    }
    rep_out = os.path.abspath(args.report)
    os.makedirs(os.path.dirname(rep_out), exist_ok=True)
    with open(rep_out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"\nCOMBINED (local 769)  recall={combined['hallucination_recall']}  "
          f"routing={combined['routing_fraction']}  gateF1={combined['gate']['f1']:.3f}  "
          f"AUROC={combined['score_metrics'].get('auroc', float('nan')):.3f}")
    print(f"\nSMOKE: prob={smoke['hallucination_probability']} conf={smoke_out.confidence_score} "
          f"risk={smoke['risk_level']} action={smoke['next_action']} calibrated={smoke_out.calibrated}")
    if "report" in v1:
        vr = v1["report"]
        print(f"\nV1 reference read from {v1['path']}")
    else:
        print(f"\nV1 reference unavailable: {v1.get('error')}")
    print(f"\nreport -> {rep_out}  ({report['eval_seconds']}s)")


if __name__ == "__main__":
    main()
