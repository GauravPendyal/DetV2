"""Train the Stage-1 logistic-regression response model on HaluEval.

Dataset-wise generalization protocol:
  * ``--holdout-group`` (default: summarization) is removed from training
    entirely and used as a cross-group test.
  * The remaining groups are split train/val (stratified).
  * The local HalluciGuard sets are untouched here (evaluated separately by
    tools/eval_stage.py --stage stage1_logreg).

Usage (from DetV2, venv active):
    python tools/train_stage1.py --holdout-group summarization --max-pairs 1500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detector_v2.datasets.public_train import load_halueval  # noqa: E402
from detector_v2.evaluation.metrics import classification_metrics, score_metrics  # noqa: E402
from detector_v2.features.response import featurize_text, feature_names  # noqa: E402
from detector_v2.scoring.logreg import LogRegResponseModel  # noqa: E402


def _xy(rows):
    X = np.vstack([featurize_text(r.query, r.response) for r in rows])
    y = np.asarray([r.label for r in rows], dtype=int)
    return X, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-group", default="summarization")
    ap.add_argument("--max-pairs", type=int, default=1500)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "models", "stage1_logreg.joblib"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    all_rows = load_halueval(max_pairs_per_config=args.max_pairs, seed=args.seed)
    train_pool = [r for r in all_rows if r.group != args.holdout_group]
    holdout = [r for r in all_rows if r.group == args.holdout_group]
    print(f"loaded {len(all_rows)} examples; train-pool {len(train_pool)} "
          f"(groups {sorted({r.group for r in train_pool})}); holdout '{args.holdout_group}' {len(holdout)}")

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(train_pool))
    n_val = int(len(train_pool) * args.val_frac)
    val_rows = [train_pool[i] for i in perm[:n_val]]
    fit_rows = [train_pool[i] for i in perm[n_val:]]

    Xtr, ytr = _xy(fit_rows)
    Xval, yval = _xy(val_rows)
    Xho, yho = _xy(holdout)

    model = LogRegResponseModel().fit(Xtr, ytr)

    def report(name, X, y):
        p = model.predict_proba(X)
        pred = (p >= 0.5).astype(int)
        cm = classification_metrics(y.tolist(), pred.tolist())
        sm = score_metrics(y.tolist(), p.tolist())
        print(f"  [{name}] n={len(y)} f1={cm['f1']:.3f} recall={cm['recall']:.3f} "
              f"prec={cm['precision']:.3f} auroc={sm['auroc']:.3f} auprc={sm['auprc']:.3f} "
              f"ece={sm['ece']:.3f} brier={sm['brier']:.3f}")
        return {"classification@0.50": cm, "score_metrics": sm}

    print("Stage-1 logistic regression:")
    metrics = {
        "train": report("train", Xtr, ytr),
        "val": report("val", Xval, yval),
        f"holdout_group:{args.holdout_group}": report(f"holdout:{args.holdout_group}", Xho, yho),
    }

    coefs = model.pipeline.named_steps["lr"].coef_[0]
    weights = sorted(zip(feature_names(), coefs.tolist()), key=lambda kv: abs(kv[1]), reverse=True)
    model.metadata = {
        "trained_on": "HaluEval",
        "train_groups": sorted({r.group for r in fit_rows}),
        "holdout_group": args.holdout_group,
        "n_train": len(fit_rows), "n_val": len(val_rows), "n_holdout": len(holdout),
        "max_pairs_per_config": args.max_pairs, "seed": args.seed,
        "metrics": metrics,
        "top_weights": weights[:10],
        "train_seconds": round(time.perf_counter() - t0, 2),
    }
    out = os.path.abspath(args.out)
    model.save(out)
    print(f"\ntop weights: {[(n, round(w,3)) for n,w in weights[:6]]}")
    print(f"saved -> {out}")

    rep_out = os.path.join(os.path.dirname(out), "..", "reports", "stage1_train.json")
    rep_out = os.path.abspath(rep_out)
    os.makedirs(os.path.dirname(rep_out), exist_ok=True)
    with open(rep_out, "w", encoding="utf-8") as fh:
        json.dump(model.metadata, fh, indent=2)
    print(f"train report -> {rep_out}")


if __name__ == "__main__":
    main()
