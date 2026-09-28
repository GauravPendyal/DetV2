"""Stage 7 — fit the leakage-safe calibrator + tune the routing threshold for the
selected CORE signal (Stage-2 encoder).

Why the encoder is the core (evidence, held-out local sets, from the stage evals):
  * combined AUROC 0.643 (vs structural 0.568) — best single signal;
  * in-family halluciguard AUROC 0.993 (vs 0.966); always-on & context-free.
Fusion (Stage 6) did not beat it cross-dataset, so no new model is trained here.

Calibration data is leakage-safe: HaluEval ``summarization`` is the group held out
from Stage 2, so the encoder's scores on it are out-of-sample; the 769-row local
eval set is NEVER used here (it stays the final test set). The summarization pool
is split 50/50 (stratified by label): one half fits the calibrator, the other
half tunes the threshold and measures post-calibration ECE/Brier (never fit on).

Usage (from DetV2, venv active):
    python tools/train_stage7_calibration.py --max-pairs 500 --target-recall 0.90
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detector_v2.calibration.calibrator import ProbabilityCalibrator  # noqa: E402
from detector_v2.config import DetectorV2Config  # noqa: E402
from detector_v2.datasets.public_train import load_halueval  # noqa: E402
from detector_v2.evaluation.metrics import score_metrics  # noqa: E402
from detector_v2.pipeline import DetectorV2Pipeline  # noqa: E402


def _stratified_halves(labels: List[int], seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.RandomState(seed)
    fit_idx: List[int] = []
    sel_idx: List[int] = []
    for lab in (0, 1):
        idxs = [i for i, y in enumerate(labels) if y == lab]
        rng.shuffle(idxs)
        half = len(idxs) // 2
        fit_idx += idxs[:half]
        sel_idx += idxs[half:]
    rng.shuffle(fit_idx)
    rng.shuffle(sel_idx)
    return fit_idx, sel_idx


def _sweep(p: np.ndarray, y: np.ndarray, thresholds) -> List[dict]:
    """Accept iff p <= t; a hallucination is caught iff routed (p > t)."""
    pos = y == 1
    n_pos = int(pos.sum())
    out = []
    for t in thresholds:
        routed = p > t
        recall = float((routed & pos).sum() / n_pos) if n_pos else float("nan")
        out.append({
            "threshold": round(float(t), 4),
            "hallucination_recall": round(recall, 4),
            "routing_fraction": round(float(routed.mean()), 4),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder-model", default=os.path.join("models", "stage2_encoder"))
    ap.add_argument("--max-pairs", type=int, default=500,
                    help="HaluEval summarization source rows (yields up to 2x)")
    ap.add_argument("--target-recall", type=float, default=0.90,
                    help="pick the largest threshold still reaching this hallucination recall "
                         "(highest recall floor at the lowest routing cost)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out-dir", default=os.path.join("models", "stage7_calibration"))
    ap.add_argument("--report", default=os.path.join("reports", "stage7_calibration.json"))
    args = ap.parse_args()

    from detector_v2.signals.encoder import EncoderSignal

    t0 = time.perf_counter()
    enc_dir = os.path.abspath(args.encoder_model)
    # Encoder-only pipeline, NO calibrator -> hallucination_probability is the raw risk.
    config = DetectorV2Config(stage="stage7_calibration_fit")
    pipe = DetectorV2Pipeline(config=config, signals=[EncoderSignal(model_dir=enc_dir)])

    rows = load_halueval(configs=["summarization"], max_pairs_per_config=args.max_pairs, seed=args.seed)
    print(f"calibration pool: halueval_summarization={len(rows)} (leakage-safe: held out from Stage 2)")

    raw: List[float] = []
    y: List[int] = []
    tc = time.perf_counter()
    for i, r in enumerate(rows):
        out = pipe.detect(r.query or "", r.response or "", context=None)
        if out.hallucination_probability is None:  # degenerate row -> skip (not fabricated)
            continue
        raw.append(float(out.hallucination_probability))
        y.append(int(r.label))
        if (i + 1) % 200 == 0:
            print(f"  scored {i + 1}/{len(rows)} ({time.perf_counter() - tc:.0f}s)")
    raw_arr = np.asarray(raw, dtype=float)
    y_arr = np.asarray(y, dtype=int)
    print(f"encoder scored {len(raw)} rows; pos_rate {y_arr.mean():.3f} "
          f"({time.perf_counter() - tc:.0f}s)")

    fit_i, sel_i = _stratified_halves(y, args.seed)
    r_fit, y_fit = raw_arr[fit_i], y_arr[fit_i]
    r_sel, y_sel = raw_arr[sel_i], y_arr[sel_i]

    # Raw (pre-calibration) metrics on the selection half.
    sm_raw = score_metrics(y_sel.tolist(), r_sel.tolist())

    # Fit both methods on the fit half; pick lower ECE on the selection half.
    candidates = {}
    for method in ("isotonic", "sigmoid"):
        cal = ProbabilityCalibrator.fit(r_fit, y_fit, method=method, threshold=0.5)
        p_sel = cal.transform_many(r_sel)
        sm = score_metrics(y_sel.tolist(), p_sel.tolist())
        candidates[method] = {"calibrator": cal, "p_sel": p_sel, "metrics": sm}
        print(f"  {method:<9} ECE {sm_raw['ece']:.4f} -> {sm['ece']:.4f}  "
              f"Brier {sm_raw['brier']:.4f} -> {sm['brier']:.4f}  AUROC {sm['auroc']:.4f}")
    best_method = min(candidates, key=lambda m: candidates[m]["metrics"]["ece"])
    chosen = candidates[best_method]
    p_sel = chosen["p_sel"]

    # Threshold sweep on the calibrated selection half; pick smallest t reaching target recall.
    grid = np.round(np.arange(0.05, 0.96, 0.05), 4)
    table = _sweep(p_sel, y_sel, grid)
    reaching = [row for row in table if row["hallucination_recall"] >= args.target_recall]
    if reaching:
        # Largest threshold that still meets the recall floor -> minimal routing cost.
        picked = max(reaching, key=lambda r: r["threshold"])
    else:  # fall back to the max-recall threshold (recall-favoring gate)
        picked = max(table, key=lambda r: r["hallucination_recall"])
    threshold = picked["threshold"]

    metadata = {
        "core_signal": "stage2_encoder",
        "core_model": args.encoder_model,
        "selection_evidence": {
            "combined_auroc": {"stage1_structural": 0.568, "stage2_encoder": 0.643},
            "halluciguard_auroc": {"stage1_structural": 0.966, "stage2_encoder": 0.993},
            "note": "encoder = best single-signal AUROC + always-on/context-free; "
                    "Stage-6 fusion did not beat it cross-dataset",
        },
        "calibration_source": "halueval_summarization (held out from Stage 2)",
        "never_used": ["769-row local eval set", "RAGTruth"],
        "n_scored": len(raw), "n_fit": len(fit_i), "n_sel": len(sel_i),
        "method_compared": {m: candidates[m]["metrics"] for m in candidates},
        "chosen_method": best_method,
        "ece_before": sm_raw["ece"], "ece_after": chosen["metrics"]["ece"],
        "brier_before": sm_raw["brier"], "brier_after": chosen["metrics"]["brier"],
        "target_recall": args.target_recall,
        "threshold_selected": threshold,
        "threshold_selected_at": picked,
        "confidence_definition": "decision-distance from threshold, per-side normalized "
                                 "to [0,1]; NOT a copy of the probability",
        "seed": args.seed,
    }
    cal = ProbabilityCalibrator(chosen["calibrator"].method, chosen["calibrator"]._model,
                                threshold, metadata)
    out_path = os.path.join(os.path.abspath(args.out_dir), "calibrator.joblib")
    cal.save(out_path)

    report = {"provenance": {k: metadata[k] for k in (
                  "core_signal", "core_model", "calibration_source", "never_used",
                  "n_scored", "n_fit", "n_sel", "seed")},
              "selection_evidence": metadata["selection_evidence"],
              "calibration": {"chosen_method": best_method,
                              "ece_before": sm_raw["ece"], "ece_after": chosen["metrics"]["ece"],
                              "brier_before": sm_raw["brier"], "brier_after": chosen["metrics"]["brier"],
                              "methods_compared": {m: candidates[m]["metrics"] for m in candidates}},
              "threshold": {"target_recall": args.target_recall, "selected": threshold,
                            "selected_row": picked, "sweep": table},
              "train_seconds": round(time.perf_counter() - t0, 1)}
    rep_out = os.path.abspath(args.report)
    os.makedirs(os.path.dirname(rep_out), exist_ok=True)
    with open(rep_out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\n=== Stage-7 threshold sweep (calibrated, on held-out selection half) ===")
    print(f"{'thr':>6}{'hall_recall':>14}{'routing_frac':>14}")
    for row in table:
        mark = "  <=" if row["threshold"] == threshold else ""
        print(f"{row['threshold']:>6}{row['hallucination_recall']:>14.3f}"
              f"{row['routing_fraction']:>14.3f}{mark}")
    print(f"\ncore=stage2_encoder  method={best_method}  threshold={threshold}  "
          f"ECE {sm_raw['ece']:.4f}->{chosen['metrics']['ece']:.4f}")
    print(f"calibrator -> {out_path}")
    print(f"report     -> {rep_out}  ({report['train_seconds']}s)")


if __name__ == "__main__":
    main()
