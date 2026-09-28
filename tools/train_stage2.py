"""Train the Stage-2 DeBERTa-v3 claim-level encoder on HaluEval.

Cross-dataset-generalization protocol (mirrors Stage 1):
  * ``--holdout-group`` (default: summarization) is removed from training and
    used as a cross-group test.
  * Remaining groups are split train/val.
  * The local HalluciGuard sets stay untouched (evaluated separately by
    tools/eval_stage.py --stage stage2_encoder --encoder-model ...).

Weak claim labels: HaluEval labels are response-level, so each claim segmented
from a response inherits that response's label (1 = hallucinated). This is noisy
by construction (a hallucinated response may contain true claims) and is reported
honestly. We evaluate at BOTH levels: per-claim (the training target) and, more
importantly, per-response (per-claim probs aggregated the way the pipeline does).

Usage (from DetV2, venv active):
    python tools/train_stage2.py --holdout-group summarization --max-pairs 700 \
        --model-name microsoft/deberta-v3-xsmall --epochs 2
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


def _segment_rows(rows, segmenter):
    """Expand each response into (query, claim_text) pairs with the row's weak
    label, and remember which response each claim belongs to for response-level
    eval."""
    pairs: List[Tuple[str, str]] = []
    claim_labels: List[int] = []
    resp_groups: List[List[int]] = []   # per response: indices into `pairs`
    resp_labels: List[int] = []
    for r in rows:
        claims = segmenter.segment(r.response)
        if not claims:
            claims = [type("C", (), {"text": r.response.strip()})()]
        start = len(pairs)
        for c in claims:
            pairs.append((r.query, c.text))
            claim_labels.append(r.label)
        resp_groups.append(list(range(start, len(pairs))))
        resp_labels.append(r.label)
    return pairs, np.asarray(claim_labels, dtype=int), resp_groups, np.asarray(resp_labels, dtype=int)


def _response_scores(claim_probs, resp_groups, method, k) -> np.ndarray:
    out = np.zeros(len(resp_groups), dtype=float)
    for i, idxs in enumerate(resp_groups):
        vals = [float(claim_probs[j]) for j in idxs]
        out[i] = aggregate(vals, method=method, k=k) if vals else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-group", default="summarization")
    ap.add_argument("--max-pairs", type=int, default=700, help="HaluEval source rows per config")
    ap.add_argument("--max-claims", type=int, default=9000, help="cap on training claims (tractability)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--model-name", default="microsoft/deberta-v3-xsmall")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--agg", default="topk_mean")
    ap.add_argument("--agg-k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "models", "stage2_encoder"))
    args = ap.parse_args()

    from detector_v2.signals.encoder.train import train_encoder

    t0 = time.perf_counter()
    seg = RuleBasedSegmenter()
    all_rows = load_halueval(max_pairs_per_config=args.max_pairs, seed=args.seed)
    train_pool = [r for r in all_rows if r.group != args.holdout_group]
    holdout = [r for r in all_rows if r.group == args.holdout_group]

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(train_pool))
    n_val = int(len(train_pool) * args.val_frac)
    val_rows = [train_pool[i] for i in perm[:n_val]]
    fit_rows = [train_pool[i] for i in perm[n_val:]]
    print(f"loaded {len(all_rows)} responses; fit {len(fit_rows)} / val {len(val_rows)} "
          f"(groups {sorted({r.group for r in fit_rows})}); holdout '{args.holdout_group}' {len(holdout)}")

    tr_pairs, tr_y, tr_groups, tr_ry = _segment_rows(fit_rows, seg)
    va_pairs, va_y, va_groups, va_ry = _segment_rows(val_rows, seg)
    ho_pairs, ho_y, ho_groups, ho_ry = _segment_rows(holdout, seg)

    if len(tr_pairs) > args.max_claims:  # keep CPU training tractable
        keep = rng.permutation(len(tr_pairs))[:args.max_claims]
        keep_set = set(keep.tolist())
        # rebuild fit set restricted to kept claims (drops response grouping for train,
        # which we don't need — response-level eval is reported on val/holdout).
        tr_pairs = [tr_pairs[i] for i in keep]
        tr_y = tr_y[keep]
    print(f"training claims: {len(tr_pairs)} (pos rate {tr_y.mean():.3f}); "
          f"val claims {len(va_pairs)}; holdout claims {len(ho_pairs)}")

    encoder = train_encoder(
        tr_pairs, tr_y.tolist(),
        model_name=args.model_name, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, max_len=args.max_len, seed=args.seed,
    )

    def report(name, pairs, y, groups, ry):
        cp = encoder.predict_proba(pairs, batch_size=max(args.batch_size, 32))
        claim_pred = (cp >= 0.5).astype(int)
        c_cm = classification_metrics(y.tolist(), claim_pred.tolist())
        c_sm = score_metrics(y.tolist(), cp.tolist())
        rp = _response_scores(cp, groups, args.agg, args.agg_k)
        r_pred = (rp >= 0.5).astype(int)
        r_cm = classification_metrics(ry.tolist(), r_pred.tolist())
        r_sm = score_metrics(ry.tolist(), rp.tolist())
        print(f"  [{name}] claim: f1={c_cm['f1']:.3f} auroc={c_sm['auroc']:.3f}  |  "
              f"response({args.agg}): f1={r_cm['f1']:.3f} recall={r_cm['recall']:.3f} "
              f"prec={r_cm['precision']:.3f} auroc={r_sm['auroc']:.3f} "
              f"ece={r_sm['ece']:.3f} brier={r_sm['brier']:.3f}")
        return {"claim": {"classification@0.50": c_cm, "score_metrics": c_sm},
                "response": {"classification@0.50": r_cm, "score_metrics": r_sm}}

    print("Stage-2 DeBERTa-v3 encoder:")
    metrics = {
        "val": report("val", va_pairs, va_y, va_groups, va_ry),
        f"holdout_group:{args.holdout_group}": report(f"holdout:{args.holdout_group}", ho_pairs, ho_y, ho_groups, ho_ry),
    }

    encoder.metadata = {
        "trained_on": "HaluEval", "backbone": args.model_name,
        "train_groups": sorted({r.group for r in fit_rows}),
        "holdout_group": args.holdout_group,
        "weak_labels": "claim inherits response label",
        "n_train_claims": len(tr_pairs), "n_val_claims": len(va_pairs),
        "n_holdout_claims": len(ho_pairs),
        "max_pairs_per_config": args.max_pairs, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr, "max_len": args.max_len,
        "aggregation": args.agg, "aggregation_k": args.agg_k, "seed": args.seed,
        "metrics": metrics,
        "train_seconds": round(time.perf_counter() - t0, 1),
        "model_name": args.model_name, "max_len": args.max_len,
    }
    out = os.path.abspath(args.out)
    encoder.save(out)
    print(f"saved encoder -> {out}  ({encoder.metadata['train_seconds']}s)")

    rep_out = os.path.abspath(os.path.join(os.path.dirname(out), "..", "reports", "stage2_train.json"))
    os.makedirs(os.path.dirname(rep_out), exist_ok=True)
    with open(rep_out, "w", encoding="utf-8") as fh:
        json.dump(encoder.metadata, fh, indent=2)
    print(f"train report -> {rep_out}")


if __name__ == "__main__":
    main()
