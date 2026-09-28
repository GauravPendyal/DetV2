"""Cached single-pass evaluation of all 7 Stage-6 fusion ablations (A-G).

We run the three frozen base signals ONCE per eval row (timing each), cache the
resulting ``SignalComponents``, then apply all 7 fusion LRs in memory — so the
expensive neural passes happen once, not 7x. Per-config latency is the sum of the
measured times of only the signals that config actually uses.

Held-out TEST sets (never trained on):
  * local 769  — halluciguard_dataset / ragtruth / truthfulqa (context ABSENT ->
    entailment abstains honestly; config C degrades and routes to Verify).
  * public RAGTruth — real source context present (entailment active).

V1 numbers are read (read-only) from ../Detector_agent/eval_report.json for the
side-by-side; V1 is NOT re-run. Emits reports/stage6_fusion_eval.json.

Usage (from DetV2, venv active):
    python tools/eval_stage6_fusion.py --ragtruth-max 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detector_v2.claims.base import Claim  # noqa: E402
from detector_v2.claims.rule_based import RuleBasedSegmenter  # noqa: E402
from detector_v2.datasets.local_eval import LOCAL_DATASETS, load_local_dataset  # noqa: E402
from detector_v2.datasets.ragtruth import load_ragtruth  # noqa: E402
from detector_v2.evaluation.metrics import (  # noqa: E402
    classification_metrics,
    latency_summary,
    recall_at_routing_budget,
    routing_curve,
    score_metrics,
)
from detector_v2.features.fusion import (  # noqa: E402
    CONFIGS,
    SignalComponents,
    build_vector,
    components_from_probs,
    subset_available,
)
from detector_v2.features.response import featurize_text  # noqa: E402
from detector_v2.scoring.fusion import FusionResponseModel  # noqa: E402
from detector_v2.scoring.logreg import LogRegResponseModel  # noqa: E402
from detector_v2.signals.base import SignalContext  # noqa: E402

# Gate thresholds (mirror DetectorV2Config): Accept iff risk <= low, else Verify.
LOW = 0.30

# Block -> which timed signal feeds it (for per-config latency accounting).
_BLOCK_TIME_KEY = {"structural": "s1_ms", "encoder": "s2_ms", "entailment": "s3_ms"}


def _row_eval(query, response, context, seg, enc_sig, ent_sig, stage1, k
              ) -> Tuple[SignalComponents, Dict[str, float]]:
    """Run the three base signals once over one row; return (components, per-signal ms)."""
    resp = response or ""
    claims = seg.segment(resp)
    if not claims:
        claims = [Claim(claim_id=0, text=resp.strip(), span=[0, len(resp)])]
    ctx = SignalContext(user_query=query or "", context=context, response=resp)

    t = time.perf_counter()
    s1 = float(stage1.predict_proba(featurize_text(query or "", resp).reshape(1, -1))[0])
    s1_ms = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    enc_sig.prepare(claims, ctx)
    s2: List[float] = []
    for c in claims:
        er = enc_sig.score(c, ctx)
        if er.available and er.value is not None:
            s2.append(float(er.value))
    s2_ms = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    ent_sig.prepare(claims, ctx)
    s3: List[float] = []
    for c in claims:
        tr = ent_sig.score(c, ctx)
        if tr.available and tr.value is not None:
            s3.append(float(tr.value))
    s3_ms = (time.perf_counter() - t) * 1000.0

    comp = components_from_probs(s1, s2 or None, s3 or None, k)
    return comp, {"s1_ms": s1_ms, "s2_ms": s2_ms, "s3_ms": s3_ms}


def _assemble(y_true, scores, latencies) -> Dict:
    """Replicate runner.evaluate_detector's report from precomputed arrays.
    gate view: routed to Verify iff degraded (score None) or score > LOW.
    threshold@0.50 view: over scored rows only."""
    routed = [1 if (s is None or float(s) > LOW) else 0 for s in scores]
    degraded = sum(1 for s in scores if s is None)
    thr_true = [t for t, s in zip(y_true, scores) if s is not None]
    thr_pred = [1 if float(s) >= 0.50 else 0 for s in scores if s is not None]
    curve = routing_curve(y_true, scores)
    return {
        "n": len(y_true),
        "positives": int(sum(y_true)),
        "negatives": int(len(y_true) - sum(y_true)),
        "degraded": int(degraded),
        "gate": classification_metrics(y_true, routed),
        "threshold@0.50": classification_metrics(thr_true, thr_pred) if thr_true else {},
        "score_metrics": score_metrics(y_true, scores),
        "latency": latency_summary(latencies),
        "routing_budgets": {
            f"<= {int(b*100)}%": recall_at_routing_budget(curve, b) for b in (0.3, 0.5, 0.7)
        },
        "routing_curve": curve,
    }


def _load_v1(detv2_root: str) -> Optional[Dict]:
    """Best-effort read of V1's eval_report.json (sibling of DetV2, read-only)."""
    path = os.path.abspath(os.path.join(detv2_root, "..", "Detector_agent", "eval_report.json"))
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return {"path": path, "report": json.load(fh)}
    except Exception as exc:  # noqa: BLE001
        return {"path": path, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=os.path.join("models", "stage6_fusion"))
    ap.add_argument("--ragtruth-max", type=int, default=300)
    ap.add_argument("--agg-k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--stage1-model", default=os.path.join("models", "stage1_logreg.joblib"))
    ap.add_argument("--encoder-model", default=os.path.join("models", "stage2_encoder"))
    ap.add_argument("--entailment-model", default=os.path.join("models", "stage3_entailment"))
    ap.add_argument("--report", default=os.path.join("reports", "stage6_fusion_eval.json"))
    args = ap.parse_args()

    from detector_v2.signals.encoder import EncoderSignal
    from detector_v2.signals.entailment import EntailmentSignal

    detv2_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    t0 = time.perf_counter()
    seg = RuleBasedSegmenter()
    stage1 = LogRegResponseModel.load(os.path.abspath(args.stage1_model))
    enc_sig = EncoderSignal(model_dir=os.path.abspath(args.encoder_model))
    ent_sig = EntailmentSignal(model_dir=os.path.abspath(args.entailment_model))

    # Load the 7 fusion models (each self-describes its subset + impute).
    models: Dict[str, FusionResponseModel] = {}
    for key in CONFIGS:
        models[key] = FusionResponseModel.load(
            os.path.join(os.path.abspath(args.models_dir), f"{key}.joblib"))

    # Assemble the held-out TEST datasets: 3 local (context absent) + public RAGTruth.
    datasets: List[Tuple[str, bool, List[Tuple[str, str, Optional[str], int]]]] = []
    for name in LOCAL_DATASETS:
        rows = load_local_dataset(name)
        datasets.append((name, False,
                         [(r.user_prompt, r.llm_response, r.context, r.label) for r in rows]))
    rt = load_ragtruth(split="test", max_rows=args.ragtruth_max, seed=args.seed)
    datasets.append(("ragtruth_public", True,
                     [(r.query, r.context, r.response, r.label) for r in rt]))
    print("eval datasets: " + ", ".join(f"{n}={len(rows)}" for n, _, rows in datasets))

    # 1) Cache components + per-signal timing ONCE per row, per dataset.
    cached: Dict[str, Dict] = {}
    for name, ctx_present, rows in datasets:
        comps: List[SignalComponents] = []
        timings: List[Dict[str, float]] = []
        y: List[int] = []
        tc = time.perf_counter()
        for q, resp, context, label in rows:
            comp, tm = _row_eval(q, resp, context, seg, enc_sig, ent_sig, stage1, args.agg_k)
            comps.append(comp)
            timings.append(tm)
            y.append(int(label))
        n_s3 = sum(1 for c in comps if c.s3_available)
        cached[name] = {"comps": comps, "timings": timings, "y": y,
                        "context_present": ctx_present, "n_entailment": n_s3}
        print(f"  {name}: cached {len(comps)} rows, entailment available on {n_s3} "
              f"({time.perf_counter() - tc:.0f}s)")

    # 2) Apply all 7 configs in memory across the cached components.
    v1 = _load_v1(detv2_root)
    configs_out: Dict[str, Dict] = {}
    local_names = list(LOCAL_DATASETS)
    for key, subset in CONFIGS.items():
        model = models[key]
        impute = model.impute
        keys = [_BLOCK_TIME_KEY[b] for b in subset]
        per_dataset: Dict[str, Dict] = {}
        local_y: List[int] = []
        local_scores: List[Optional[float]] = []
        local_lat: List[float] = []
        for name, ctx_present, _rows in datasets:
            c = cached[name]
            comps, timings, y = c["comps"], c["timings"], c["y"]
            # Batched predict over available rows; degraded rows stay None.
            avail_idx = [i for i, comp in enumerate(comps) if subset_available(subset, comp)]
            scores: List[Optional[float]] = [None] * len(comps)
            if avail_idx:
                X = np.asarray([build_vector(subset, comps[i], impute) for i in avail_idx],
                               dtype=float)
                p = model.predict_proba(X)
                for i, pi in zip(avail_idx, p):
                    scores[i] = float(pi)
            lat = [sum(tm[kk] for kk in keys) for tm in timings]
            per_dataset[name] = _assemble(y, scores, lat)
            if name in local_names:
                local_y += y
                local_scores += scores
                local_lat += lat
        combined_local = _assemble(local_y, local_scores, local_lat)
        macro_local = {
            "gate_f1": sum(per_dataset[n]["gate"]["f1"] for n in local_names) / len(local_names),
            "gate_recall": sum(per_dataset[n]["gate"]["recall"] for n in local_names) / len(local_names),
        }
        configs_out[key] = {
            "subset": subset, "feature_names": model.feature_names,
            "weights": model.metadata.get("weights", {}),
            "per_dataset": per_dataset,
            "combined_local": combined_local,
            "macro_local": macro_local,
        }

    # 3) Compact console table: per-config AUROC per dataset.
    ds_order = local_names + ["ragtruth_public"]
    print("\n=== Stage-6 fusion eval — AUROC per dataset ===")
    print(f"{'cfg':<4}{'subset':<26}" + "".join(f"{n[:14]:>15}" for n in ds_order))
    for key, subset in CONFIGS.items():
        cells = []
        for n in ds_order:
            sm = configs_out[key]["per_dataset"][n]["score_metrics"]
            cells.append(f"{sm.get('auroc', float('nan')):>15.3f}")
        print(f"{key:<4}{'+'.join(subset):<26}" + "".join(cells))

    provenance = {
        "test_sets": {
            "local_769": {"datasets": local_names, "context": "absent (entailment abstains)"},
            "ragtruth_public": {"dataset": "wandb/RAGTruth-processed", "n": len(rt),
                                "context": "present"},
        },
        "gate_threshold_low": LOW, "agg_k": args.agg_k, "seed": args.seed,
        "base_models": {"stage1": args.stage1_model, "stage2": args.encoder_model,
                        "stage3": args.entailment_model},
        "entailment_available": {n: cached[n]["n_entailment"] for n in cached},
    }
    report = {"provenance": provenance, "v1_reference": v1, "configs": configs_out,
              "eval_seconds": round(time.perf_counter() - t0, 1)}
    rep_out = os.path.abspath(args.report)
    os.makedirs(os.path.dirname(rep_out), exist_ok=True)
    with open(rep_out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\neval report -> {rep_out}  ({report['eval_seconds']}s)")


if __name__ == "__main__":
    main()
