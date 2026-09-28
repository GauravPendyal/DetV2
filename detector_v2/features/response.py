"""Response feature engineering.

Turns a response (its per-claim structural features) into a fixed-length vector
for the Stage-1 logistic-regression model. The SAME vector is produced two ways
so training and inference never drift:
  * ``featurize_text``   — from raw (query, response); used in training.
  * ``featurize_claims`` — from already-scored ``ClaimSignal``s; used at
                            inference so nothing is recomputed.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..claims.rule_based import RuleBasedSegmenter
from ..schemas import ClaimSignal
from ..signals.base import SignalContext
from ..signals.structural.signal import StructuralSignal

_DENSITY_KEYS = [
    "d_specificity", "d_superlative_density", "d_absolute_density",
    "d_overassert_density", "d_false_premise_density", "d_hedge_density",
]
_COUNT_KEYS = [
    "num_count", "date_count", "proper_noun_count", "superlative_count",
    "absolute_count", "overassertion_count", "hedge_count", "false_premise_cue_count",
]


def feature_names() -> List[str]:
    names = ["n_claims", "total_tokens", "struct_mean", "struct_max"]
    for k in _DENSITY_KEYS:
        names += [f"{k}_mean", f"{k}_max"]
    names += [f"{k}_per_tok" for k in _COUNT_KEYS]
    return names


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _build_vector(per_claim: List[Tuple[float, Dict[str, float]]]) -> np.ndarray:
    if not per_claim:
        return np.zeros(len(feature_names()), dtype=float)
    values = [v for v, _ in per_claim]
    feats = [f for _, f in per_claim]
    total_tokens = sum(f.get("n_tokens", 0.0) for f in feats) or 1.0

    vec: List[float] = [float(len(per_claim)), float(total_tokens), _mean(values), max(values)]
    for k in _DENSITY_KEYS:
        xs = [f.get(k, 0.0) for f in feats]
        vec += [_mean(xs), max(xs)]
    for k in _COUNT_KEYS:
        vec.append(sum(f.get(k, 0.0) for f in feats) / total_tokens)
    return np.asarray(vec, dtype=float)


# Shared segmenter/signal for the training-time featurizer (built once).
_SEG = RuleBasedSegmenter()
_SIG = StructuralSignal()


def featurize_text(query: str, response: str) -> np.ndarray:
    claims = _SEG.segment(response or "")
    ctx = SignalContext(user_query=query or "", response=response or "")
    per_claim = [(r.value or 0.0, r.features) for r in (_SIG.score(c, ctx) for c in claims)]
    return _build_vector(per_claim)


def featurize_claims(claims: List[ClaimSignal]) -> np.ndarray:
    per_claim: List[Tuple[float, Dict[str, float]]] = []
    for c in claims:
        struct = next((s for s in c.signals if s.name == "structural"), None)
        if struct is None:
            continue
        per_claim.append((struct.value or 0.0, struct.features))
    return _build_vector(per_claim)
