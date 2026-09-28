"""Train the Stage-6 learned fusion (logistic regression) over the Stage-1/2/3
signals — all 7 ablations A-G.

Meta-training data (leakage-clean, OUT-OF-SAMPLE for every base signal):
  1. HaluEval ``summarization`` — the group held out from Stages 1, 2 AND 3.
  2. A fresh external grounded set never seen by any stage (default tals/vitaminc,
     fallback pietrolesci/nli_fever): evidence -> context, claim -> response.
RAGTruth and the local 769 rows are NEVER trained on (held-out test).

We run the three frozen base signals ONCE per row to cache SignalComponents,
split train/val 80/20 (stratified by source x label), compute impute constants,
then fit one LogisticRegression per ablation subset — augmenting entailment-
containing subsets with entailment-masked duplicates so the fusion learns the
Stage-3-absent fallback. Base model artifacts are untouched.

Usage (from DetV2, venv active):
    python tools/train_stage6_fusion.py --max-pairs 600 --external-max 1200
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

from detector_v2.claims.base import Claim  # noqa: E402
from detector_v2.claims.rule_based import RuleBasedSegmenter  # noqa: E402
from detector_v2.datasets.external_grounded import load_external_grounded  # noqa: E402
from detector_v2.datasets.public_train import load_halueval  # noqa: E402
from detector_v2.evaluation.metrics import classification_metrics, score_metrics  # noqa: E402
from detector_v2.features.fusion import (  # noqa: E402
    CONFIGS,
    SignalComponents,
    build_training_matrix,
    build_vector,
    components_from_probs,
    compute_impute,
)
from detector_v2.features.response import featurize_text  # noqa: E402
from detector_v2.scoring.fusion import FusionResponseModel  # noqa: E402
from detector_v2.scoring.logreg import LogRegResponseModel  # noqa: E402
from detector_v2.signals.base import SignalContext  # noqa: E402


def _row_components(row, seg, enc_sig, ent_sig, stage1, k) -> SignalComponents:
    """Run the three frozen base signals once over one response -> components."""
    resp = row.response or ""
    claims = seg.segment(resp)
    if not claims:
        claims = [Claim(claim_id=0, text=resp.strip(), span=[0, len(resp)])]
    ctx = SignalContext(user_query=row.query or "", context=row.context, response=resp)
    enc_sig.prepare(claims, ctx)
    ent_sig.prepare(claims, ctx)
    s2: List[float] = []
    s3: List[float] = []
    for c in claims:
        er = enc_sig.score(c, ctx)
        if er.available and er.value is not None:
            s2.append(float(er.value))
        tr = ent_sig.score(c, ctx)
        if tr.available and tr.value is not None:
            s3.append(float(tr.value))
    s1 = float(stage1.predict_proba(featurize_text(row.query or "", resp).reshape(1, -1))[0])
    return components_from_probs(s1, s2 or None, s3 or None, k)


def _stratified_split(groups: List[str], labels: List[int], val_frac: float, seed: int):
    """Indices split train/val, stratified by (source_group, label)."""
    rng = np.random.RandomState(seed)
    strata: dict = {}
    for i, (g, y) in enumerate(zip(groups, labels)):
        strata.setdefault((g, y), []).append(i)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for idxs in strata.values():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_val = int(round(len(idxs) * val_frac))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=600,
                    help="HaluEval summarization source rows (yields up to 2x: right+hallucinated)")
    ap.add_argument("--external-dataset", default="tals/vitaminc")
    ap.add_argument("--external-max", type=int, default=1200)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--agg-k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--stage1-model", default=os.path.join("models", "stage1_logreg.joblib"))
    ap.add_argument("--encoder-model", default=os.path.join("models", "stage2_encoder"))
    ap.add_argument("--entailment-model", default=os.path.join("models", "stage3_entailment"))
    ap.add_argument("--out-dir", default=os.path.join("models", "stage6_fusion"))
    ap.add_argument("--report", default=os.path.join("reports", "stage6_fusion_train.json"))
    args = ap.parse_args()

    from detector_v2.signals.encoder import EncoderSignal
    from detector_v2.signals.entailment import EntailmentSignal

    t0 = time.perf_counter()
    seg = RuleBasedSegmenter()
    stage1 = LogRegResponseModel.load(os.path.abspath(args.stage1_model))
    enc_sig = EncoderSignal(model_dir=os.path.abspath(args.encoder_model))
    ent_sig = EntailmentSignal(model_dir=os.path.abspath(args.entailment_model))

    # 1) Meta-training pool: summarization (held out from all base models) + external.
    summ = load_halueval(configs=["summarization"], max_pairs_per_config=args.max_pairs, seed=args.seed)
    for r in summ:
        r.group = "halueval_summarization"
    ext = load_external_grounded(args.external_dataset, max_rows=args.external_max, seed=args.seed)
    pool = summ + ext
    print(f"meta-training pool: summarization={len(summ)} + external '{args.external_dataset}'={len(ext)} "
          f"= {len(pool)} rows")

    # 2) Run the three frozen base signals once per row -> cached components.
    comps: List[SignalComponents] = []
    labels: List[int] = []
    groups: List[str] = []
    tc = time.perf_counter()
    for i, row in enumerate(pool):
        comps.append(_row_components(row, seg, enc_sig, ent_sig, stage1, args.agg_k))
        labels.append(int(row.label))
        groups.append(row.group)
        if (i + 1) % 200 == 0:
            print(f"  cached {i + 1}/{len(pool)} components ({time.perf_counter() - tc:.0f}s)")
    n_s3 = sum(1 for c in comps if c.s3_available)
    print(f"components cached: {len(comps)} (entailment available on {n_s3}); "
          f"pos_rate {np.mean(labels):.3f} ({time.perf_counter() - tc:.0f}s)")

    # 3) Stratified train/val split; impute constants from TRAIN only.
    tr_idx, va_idx = _stratified_split(groups, labels, args.val_frac, args.seed)
    tr_comps = [comps[i] for i in tr_idx]
    tr_y = [labels[i] for i in tr_idx]
    va_comps = [comps[i] for i in va_idx]
    va_y = [labels[i] for i in va_idx]
    impute = compute_impute(tr_comps)
    print(f"split: train={len(tr_idx)} val={len(va_idx)}; impute={ {k: round(v, 4) for k, v in impute.items()} }")

    # 4) Fit all 7 ablation LRs on the SAME cached components.
    os.makedirs(os.path.abspath(args.out_dir), exist_ok=True)
    provenance = {
        "meta_training_sources": {
            "halueval_summarization": {"n": len(summ), "note": "held out from Stages 1/2/3"},
            "external_grounded": {"dataset": args.external_dataset, "n": len(ext),
                                  "note": "never seen by any stage"},
        },
        "never_trained_on": ["RAGTruth (public test)", "local Datasets/*.jsonl (769)"],
        "n_pool": len(pool), "n_train": len(tr_idx), "n_val": len(va_idx),
        "entailment_available_in_pool": int(n_s3),
        "pool_pos_rate": float(np.mean(labels)), "agg_k": args.agg_k, "seed": args.seed,
        "impute": impute,
        "base_models": {"stage1": args.stage1_model, "stage2": args.encoder_model,
                        "stage3": args.entailment_model},
    }
    configs_out = {}
    manifest = {"configs": {}, "provenance": provenance}
    print("\n=== Stage-6 fusion ablations (val metrics) ===")
    print(f"{'cfg':<4}{'subset':<34}{'auroc':>7}{'auprc':>7}{'f1':>7}{'ece':>7}{'brier':>7}")
    for key, subset in CONFIGS.items():
        Xtr, ytr = build_training_matrix(subset, tr_comps, tr_y, impute)
        model = FusionResponseModel(subset)
        model.impute = impute
        model.fit(Xtr, ytr)

        # Validation: NO augmentation (real operating distribution: entailment present).
        Xva = np.asarray([build_vector(subset, c, impute) for c in va_comps], dtype=float)
        pva = model.predict_proba(Xva)
        cm = classification_metrics(va_y, (pva >= 0.5).astype(int).tolist())
        sm = score_metrics(va_y, pva.tolist())
        weights = model.weights()

        model.metadata = {
            "stage": "stage6_fusion", "config": key, "subset": subset,
            "feature_names": model.feature_names, "weights": weights,
            "val_metrics": {"classification@0.50": cm, "score_metrics": sm},
            "n_train_rows_after_aug": int(Xtr.shape[0]), "provenance": provenance,
        }
        path = os.path.join(os.path.abspath(args.out_dir), f"{key}.joblib")
        model.save(path)

        configs_out[key] = {"subset": subset, "feature_names": model.feature_names,
                            "weights": weights,
                            "val_metrics": {"classification@0.50": cm, "score_metrics": sm},
                            "n_train_rows_after_aug": int(Xtr.shape[0])}
        manifest["configs"][key] = {"subset": subset, "file": f"{key}.joblib",
                                    "feature_names": model.feature_names}
        print(f"{key:<4}{'+'.join(subset):<34}{sm['auroc']:>7.3f}{sm['auprc']:>7.3f}"
              f"{cm['f1']:>7.3f}{sm['ece']:>7.3f}{sm['brier']:>7.3f}")

    with open(os.path.join(os.path.abspath(args.out_dir), "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    report = {"provenance": provenance, "configs": configs_out,
              "train_seconds": round(time.perf_counter() - t0, 1)}
    rep_out = os.path.abspath(args.report)
    os.makedirs(os.path.dirname(rep_out), exist_ok=True)
    with open(rep_out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nsaved 7 fusion models -> {os.path.abspath(args.out_dir)}")
    print(f"train report -> {rep_out}  ({report['train_seconds']}s)")


if __name__ == "__main__":
    main()
