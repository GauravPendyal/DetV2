"""Stage-3 grounded-entailment evaluation.

Two things the brief asks for:
  1. The head-to-head that motivates this stage — Stage 2 (context-free encoder)
     vs Stage 3 (context-aware entailment) on **public RAGTruth test**, which has
     the source document the local ragtruth set lacks. Same rows, same aggregation.
  2. Proof of honest degradation — Stage 3 on the context-free local 769 sets must
     abstain (no fabricated evidence), routing every row to Verify.

RAGTruth is load-for-eval only (never trained on). Writes reports/stage3_grounded_eval.json.
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
from detector_v2.datasets.local_eval import EvalRow  # noqa: E402
from detector_v2.datasets.ragtruth import load_ragtruth  # noqa: E402
from detector_v2.evaluation.runner import evaluate_all, evaluate_detector  # noqa: E402
from detector_v2.pipeline import DetectorV2Pipeline  # noqa: E402


def _encoder_agent(model_dir: str) -> DetectorAgent:
    from detector_v2.signals.encoder import EncoderSignal
    cfg = DetectorV2Config(stage="stage2_encoder")
    return DetectorAgent(pipeline=DetectorV2Pipeline(config=cfg, signals=[EncoderSignal(model_dir=model_dir)]))


def _entailment_agent(model_dir: str) -> DetectorAgent:
    from detector_v2.signals.entailment import EntailmentSignal
    cfg = DetectorV2Config(stage="stage3_entailment")
    return DetectorAgent(pipeline=DetectorV2Pipeline(config=cfg, signals=[EntailmentSignal(model_dir=model_dir)]))


def _rows_from_ragtruth(max_rows, seed, task_types):
    rt = load_ragtruth("test", max_rows=max_rows, seed=seed, task_types=task_types)
    return [EvalRow("ragtruth_public", r.query, r.response, r.label, context=r.context) for r in rt]


def _line(tag, d):
    g = d["gate"]; sm = d["score_metrics"]; n = d["n"]
    routed = (g["tp"] + g["fp"]) / n if n else 0.0
    return (f"{tag:<22} {n:>4} {sm['auroc']:>7.3f} {sm['auprc']:>7.3f} "
            f"{g['recall']:>7.3f} {g['precision']:>7.3f} {g['f1']:>7.3f} "
            f"{sm['ece']:>6.3f} {sm['brier']:>6.3f} {routed:>7.2f} {d['degraded']:>5} "
            f"{d['latency']['p95_ms']:>8.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder-model", default=os.path.join("models", "stage2_encoder"))
    ap.add_argument("--entailment-model", default=os.path.join("models", "stage3_entailment"))
    ap.add_argument("--max-rows", type=int, default=600)
    ap.add_argument("--task-types", default="Summary,QA,Data2txt")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=os.path.join("reports", "stage3_grounded_eval.json"))
    args = ap.parse_args()

    task_types = [t.strip() for t in args.task_types.split(",") if t.strip()]
    rows = _rows_from_ragtruth(args.max_rows, args.seed, task_types)
    pos = sum(r.label for r in rows)
    print(f"RAGTruth test sample: n={len(rows)} pos={pos} ({pos/len(rows):.2f}) tasks={task_types}")

    enc = _encoder_agent(os.path.abspath(args.encoder_model))
    ent = _entailment_agent(os.path.abspath(args.entailment_model))

    print("\n=== Stage 2 vs Stage 3 on public RAGTruth (grounded) ===")
    hdr = f"{'signal':<22} {'n':>4} {'auroc':>7} {'auprc':>7} {'recall':>7} {'prec':>7} {'f1':>7} {'ece':>6} {'brier':>6} {'routed':>7} {'degr':>5} {'p95ms':>8}"
    print(hdr)
    t0 = time.perf_counter()
    enc_res = evaluate_detector(enc.detect_output, rows)
    print(_line("stage2_encoder", enc_res))
    ent_res = evaluate_detector(ent.detect_output, rows)
    print(_line("stage3_entailment", ent_res))
    print(f"(eval wall time {time.perf_counter()-t0:.1f}s)")

    # Honest-degradation check: Stage 3 on the context-free local sets.
    print("\n=== Stage 3 on context-free local sets (must abstain, not fabricate) ===")
    local = evaluate_all(ent.detect_output)
    for name, d in local["per_dataset"].items():
        print(f"  {name:<22} n={d['n']:>4} degraded={d['degraded']:>4} "
              f"routed_all={'yes' if d['gate']['tp']+d['gate']['fp']==d['n'] else 'no'} "
              f"score_rows={sum(1 for _ in d['routing_curve']) and (d['n']-d['degraded'])}")

    report = {
        "ragtruth_grounded": {
            "n": len(rows), "positives": pos, "task_types": task_types,
            "stage2_encoder": enc_res, "stage3_entailment": ent_res,
        },
        "stage3_on_local_context_free": {
            name: {"n": d["n"], "degraded": d["degraded"], "gate": d["gate"]}
            for name, d in local["per_dataset"].items()
        },
    }
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nFull report -> {out}")


if __name__ == "__main__":
    main()
