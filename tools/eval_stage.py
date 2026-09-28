"""Stage evaluation runner.

Usage (from the DetV2 dir, with the venv active):
    python tools/eval_stage.py --stage stage0_heuristic

Builds one DetectorAgent (models loaded once), measures cold-start vs warm
latency, evaluates per-dataset on the local held-out sets, and writes
reports/<stage>_eval.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detector_v2.agent import DetectorAgent  # noqa: E402
from detector_v2.config import DetectorV2Config  # noqa: E402
from detector_v2.evaluation.runner import evaluate_all  # noqa: E402
from detector_v2.pipeline import DetectorV2Pipeline  # noqa: E402


def _build_agent(stage: str, response_model: str | None, encoder_model: str | None) -> DetectorAgent:
    config = DetectorV2Config(stage=stage)
    if encoder_model:
        # Stage 2: encoder is the (only) per-claim signal; per-claim probs are
        # aggregated into a response risk by the default aggregation scorer.
        from detector_v2.signals.encoder import EncoderSignal
        pipeline = DetectorV2Pipeline(config=config, signals=[EncoderSignal(model_dir=encoder_model)])
        return DetectorAgent(pipeline=pipeline)
    if response_model:
        from detector_v2.scoring.logreg import LogRegResponseModel, LogRegResponseScorer
        model = LogRegResponseModel.load(response_model)
        pipeline = DetectorV2Pipeline(config=config, response_scorer=LogRegResponseScorer(model))
        return DetectorAgent(pipeline=pipeline)
    return DetectorAgent(config=config)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="stage0_heuristic")
    ap.add_argument("--response-model", default=None, help="path to a trained response model (.joblib)")
    ap.add_argument("--encoder-model", default=None, help="path to a trained encoder model dir (Stage 2)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.environ["DETECTORV2_STAGE"] = args.stage

    # Cold start = construct agent + first inference (models load lazily here).
    t0 = time.perf_counter()
    agent = _build_agent(args.stage, args.response_model, args.encoder_model)
    _ = agent.detect_output("warmup query", "This is a warmup sentence for cold start timing.")
    cold_ms = (time.perf_counter() - t0) * 1000.0

    report = evaluate_all(agent.detect_output)
    report["meta"] = {"stage": args.stage, "cold_start_ms": round(cold_ms, 3)}

    out = args.out or os.path.join(os.path.dirname(__file__), "..", "reports", f"{args.stage}_eval.json")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    _print_summary(report, args.stage)
    print(f"\nFull report -> {out}")


def _print_summary(report: dict, stage: str) -> None:
    print(f"\n=== Detector V2 [{stage}] — per-dataset (gate view) ===")
    print(f"{'dataset':<24} {'n':>5} {'recall':>8} {'prec':>8} {'f1':>8} {'auroc':>8} {'p95ms':>8} {'degr':>5}")
    for name, d in report["per_dataset"].items():
        g = d["gate"]
        sm = d["score_metrics"]
        auroc = sm.get("auroc", float("nan"))
        print(f"{name:<24} {d['n']:>5} {g['recall']:>8.3f} {g['precision']:>8.3f} "
              f"{g['f1']:>8.3f} {auroc:>8.3f} {d['latency']['p95_ms']:>8.3f} {d['degraded']:>5}")
    c = report["combined"]["gate"]
    m = report["macro_avg"]
    print(f"{'COMBINED':<24} {report['combined']['n']:>5} {c['recall']:>8.3f} "
          f"{c['precision']:>8.3f} {c['f1']:>8.3f}")
    print(f"MACRO-AVG gate F1={m['gate_f1']:.3f}  recall={m['gate_recall']:.3f}")


if __name__ == "__main__":
    main()
