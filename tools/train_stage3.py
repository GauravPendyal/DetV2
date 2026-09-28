"""Train the Stage-3 grounded-entailment classifier (MiniCheck-STYLE, our own head).

Data: HaluEval WITH its context field — qa/dialogue use ``knowledge``, summarization
uses ``document``. We train on qa + dialogue and hold out summarization entirely as a
grounded cross-group test. Input is ``(context, claim)``; target is the weak claim
label (1 = the claim's response was hallucinated -> treat claim as unsupported). The
public RAGTruth set is NEVER trained on — it is a fully external grounded test
(tools/eval_stage3_grounded.py).

Difference from "just download MiniCheck": the backbone is a plain deberta-v3 LM; the
entailment head is fit here, by us, on HaluEval grounding pairs.

Usage (from DetV2, venv active):
    python tools/train_stage3.py --max-pairs 600 --epochs 2 --max-len 256
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

from detector_v2.aggregation import aggregate  # noqa: E402
from detector_v2.claims.rule_based import RuleBasedSegmenter  # noqa: E402
from detector_v2.datasets.public_train import load_halueval  # noqa: E402
from detector_v2.evaluation.metrics import classification_metrics, score_metrics  # noqa: E402

_CONTEXT_TRAIN_GROUPS = ["qa", "dialogue"]  # summarization held out (document context)


def _segment_rows(rows, segmenter):
    pairs: List[Tuple[str, str]] = []
    claim_labels: List[int] = []
    resp_groups: List[List[int]] = []
    resp_labels: List[int] = []
    for r in rows:
        ctx = r.context or ""
        claims = segmenter.segment(r.response)
        if not claims:
            claims = [type("C", (), {"text": r.response.strip()})()]
        start = len(pairs)
        for c in claims:
            pairs.append((ctx, c.text))
            claim_labels.append(r.label)
        resp_groups.append(list(range(start, len(pairs))))
        resp_labels.append(r.label)
    return pairs, np.asarray(claim_labels, int), resp_groups, np.asarray(resp_labels, int)


def _response_scores(claim_probs, resp_groups, method, k):
    out = np.zeros(len(resp_groups))
    for i, idxs in enumerate(resp_groups):
        vals = [float(claim_probs[j]) for j in idxs]
        out[i] = aggregate(vals, method=method, k=k) if vals else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=600, help="HaluEval source rows per config")
    ap.add_argument("--max-claims", type=int, default=8000)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--model-name", default="microsoft/deberta-v3-xsmall")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--agg", default="topk_mean")
    ap.add_argument("--agg-k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "models", "stage3_entailment"))
    args = ap.parse_args()

    from detector_v2.signals.encoder.train import train_encoder

    t0 = time.perf_counter()
    seg = RuleBasedSegmenter()
    all_rows = load_halueval(max_pairs_per_config=args.max_pairs, seed=args.seed)
    train_pool = [r for r in all_rows if r.group in _CONTEXT_TRAIN_GROUPS]
    holdout = [r for r in all_rows if r.group == "summarization"]

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(train_pool))
    n_val = int(len(train_pool) * args.val_frac)
    val_rows = [train_pool[i] for i in perm[:n_val]]
    fit_rows = [train_pool[i] for i in perm[n_val:]]
    print(f"context-train groups {sorted(set(_CONTEXT_TRAIN_GROUPS))}: fit {len(fit_rows)} / "
          f"val {len(val_rows)}; holdout 'summarization' {len(holdout)}")

    tr_pairs, tr_y, _, _ = _segment_rows(fit_rows, seg)
    va_pairs, va_y, va_groups, va_ry = _segment_rows(val_rows, seg)
    ho_pairs, ho_y, ho_groups, ho_ry = _segment_rows(holdout, seg)

    if len(tr_pairs) > args.max_claims:
        keep = rng.permutation(len(tr_pairs))[:args.max_claims]
        tr_pairs = [tr_pairs[i] for i in keep]
        tr_y = tr_y[keep]
    print(f"training (context,claim) pairs: {len(tr_pairs)} (pos {tr_y.mean():.3f}); "
          f"val {len(va_pairs)}; holdout {len(ho_pairs)}")

    encoder = train_encoder(
        tr_pairs, tr_y.tolist(), model_name=args.model_name, epochs=args.epochs,
        lr=args.lr, batch_size=args.batch_size, max_len=args.max_len, seed=args.seed,
    )

    def report(name, pairs, y, groups, ry):
        cp = encoder.predict_proba(pairs, batch_size=max(args.batch_size, 32))
        c_cm = classification_metrics(y.tolist(), (cp >= 0.5).astype(int).tolist())
        c_sm = score_metrics(y.tolist(), cp.tolist())
        rp = _response_scores(cp, groups, args.agg, args.agg_k)
        r_cm = classification_metrics(ry.tolist(), (rp >= 0.5).astype(int).tolist())
        r_sm = score_metrics(ry.tolist(), rp.tolist())
        print(f"  [{name}] claim: f1={c_cm['f1']:.3f} auroc={c_sm['auroc']:.3f}  |  "
              f"response: f1={r_cm['f1']:.3f} recall={r_cm['recall']:.3f} prec={r_cm['precision']:.3f} "
              f"auroc={r_sm['auroc']:.3f} ece={r_sm['ece']:.3f} brier={r_sm['brier']:.3f}")
        return {"claim": {"classification@0.50": c_cm, "score_metrics": c_sm},
                "response": {"classification@0.50": r_cm, "score_metrics": r_sm}}

    print("Stage-3 grounded entailment (single-pass truncated-context eval):")
    metrics = {
        "val": report("val", va_pairs, va_y, va_groups, va_ry),
        "holdout_group:summarization": report("holdout:summarization", ho_pairs, ho_y, ho_groups, ho_ry),
    }

    encoder.metadata = {
        "trained_on": "HaluEval (context-grounded)", "backbone": args.model_name,
        "signal": "grounded_entailment", "train_groups": _CONTEXT_TRAIN_GROUPS,
        "holdout_group": "summarization", "input_format": "(context, claim)",
        "weak_labels": "claim inherits response label (unsupported=1)",
        "n_train_pairs": len(tr_pairs), "n_val_pairs": len(va_pairs), "n_holdout_pairs": len(ho_pairs),
        "max_pairs_per_config": args.max_pairs, "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "max_len": args.max_len, "aggregation": args.agg, "aggregation_k": args.agg_k,
        "seed": args.seed, "metrics": metrics, "train_seconds": round(time.perf_counter() - t0, 1),
        "model_name": args.model_name,
    }
    out = os.path.abspath(args.out)
    encoder.save(out)
    print(f"saved entailment model -> {out}  ({encoder.metadata['train_seconds']}s)")

    rep_out = os.path.abspath(os.path.join(os.path.dirname(out), "..", "reports", "stage3_train.json"))
    os.makedirs(os.path.dirname(rep_out), exist_ok=True)
    with open(rep_out, "w", encoding="utf-8") as fh:
        json.dump(encoder.metadata, fh, indent=2)
    print(f"train report -> {rep_out}")


if __name__ == "__main__":
    main()
