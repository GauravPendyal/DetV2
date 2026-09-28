"""Stage-6 fusion feature engineering (train == inference).

Each base signal contributes a small block; the fusion logistic regression maps
the concatenated blocks to P(hallucinated). The SAME vector must be produced two
ways so meta-training and inference never drift (mirroring the Stage-1
``featurize_text``/``featurize_claims`` parity):
  * training  — from cached ``SignalComponents`` computed once per row.
  * inference — from a response's already-scored ``ClaimSignal``s.

Blocks (canonical order):
  * ``structural``  : ``[s1_prob]``                              (Stage-1 LR prob)
  * ``encoder``     : ``[s2_mean, s2_max, s2_topk]``             (Stage-2 per-claim probs)
  * ``entailment``  : ``[s3_mean, s3_max, s3_topk, s3_avail]``   (Stage-3 + availability flag)

Missing Stage 3 (context absent) is imputed with train-time constants and
``s3_avail=0``; the fusion LR learns to fall back on the other blocks (the train
script duplicates entailment-present rows with entailment masked). A subset with
*no* available block degrades honestly upstream — nothing is fabricated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..aggregation import aggregate
from ..schemas import ClaimSignal

CANON_ORDER = ["structural", "encoder", "entailment"]

# Ablation configs A-G (requirement 7).
CONFIGS: Dict[str, List[str]] = {
    "A": ["structural"],
    "B": ["encoder"],
    "C": ["entailment"],
    "D": ["structural", "encoder"],
    "E": ["encoder", "entailment"],
    "F": ["structural", "entailment"],
    "G": ["structural", "encoder", "entailment"],
}

_BLOCK_FEATURES: Dict[str, List[str]] = {
    "structural": ["s1_prob"],
    "encoder": ["s2_mean", "s2_max", "s2_topk"],
    "entailment": ["s3_mean", "s3_max", "s3_topk", "s3_avail"],
}


def fusion_feature_names(subset: Sequence[str]) -> List[str]:
    """Feature names for a subset, always in canonical block order."""
    names: List[str] = []
    for block in CANON_ORDER:
        if block in subset:
            names += _BLOCK_FEATURES[block]
    return names


def _agg3(values: List[float], k: int = 3):
    """(mean, max, top-k mean) of per-claim probs, reusing the shared aggregators."""
    return (
        aggregate(values, "mean", k),
        aggregate(values, "max", k),
        aggregate(values, "topk_mean", k),
    )


@dataclass
class SignalComponents:
    """Per-response base-signal outputs. A ``None`` aggregate means that signal
    was unavailable for this response (e.g. entailment with no context)."""

    s1_prob: Optional[float] = None
    s2_mean: Optional[float] = None
    s2_max: Optional[float] = None
    s2_topk: Optional[float] = None
    s3_mean: Optional[float] = None
    s3_max: Optional[float] = None
    s3_topk: Optional[float] = None

    @property
    def s1_available(self) -> bool:
        return self.s1_prob is not None

    @property
    def s2_available(self) -> bool:
        return self.s2_mean is not None

    @property
    def s3_available(self) -> bool:
        return self.s3_mean is not None

    def masked_entailment(self) -> "SignalComponents":
        """Copy with the entailment block marked unavailable (missingness aug)."""
        return SignalComponents(
            s1_prob=self.s1_prob,
            s2_mean=self.s2_mean, s2_max=self.s2_max, s2_topk=self.s2_topk,
            s3_mean=None, s3_max=None, s3_topk=None,
        )


def components_from_probs(
    s1_prob: Optional[float],
    s2_probs: Optional[List[float]],
    s3_probs: Optional[List[float]],
    k: int = 3,
) -> SignalComponents:
    """Aggregate raw per-claim probability lists into a ``SignalComponents``."""
    comp = SignalComponents(s1_prob=(None if s1_prob is None else float(s1_prob)))
    if s2_probs:
        comp.s2_mean, comp.s2_max, comp.s2_topk = _agg3(list(s2_probs), k)
    if s3_probs:
        comp.s3_mean, comp.s3_max, comp.s3_topk = _agg3(list(s3_probs), k)
    return comp


def components_from_claimsignals(
    claims: List[ClaimSignal], s1_prob: Optional[float], k: int = 3
) -> SignalComponents:
    """Read encoder/entailment per-claim values off scored ``ClaimSignal``s; the
    Stage-1 prob is supplied by the caller (it applies the Stage-1 LR model)."""
    s2 = [s.value for c in claims for s in c.signals
          if s.name == "encoder" and s.available and s.value is not None]
    s3 = [s.value for c in claims for s in c.signals
          if s.name == "entailment" and s.available and s.value is not None]
    return components_from_probs(s1_prob, s2 or None, s3 or None, k)


def compute_impute(comps: Sequence[SignalComponents]) -> Dict[str, float]:
    """Mean of each feature over rows where it is available — the fallback used
    when a block is missing at inference. Computed on TRAIN rows only."""
    def _mean(vals: List[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    return {
        "s1_prob": _mean([c.s1_prob for c in comps if c.s1_available]),
        "s2_mean": _mean([c.s2_mean for c in comps if c.s2_available]),
        "s2_max": _mean([c.s2_max for c in comps if c.s2_available]),
        "s2_topk": _mean([c.s2_topk for c in comps if c.s2_available]),
        "s3_mean": _mean([c.s3_mean for c in comps if c.s3_available]),
        "s3_max": _mean([c.s3_max for c in comps if c.s3_available]),
        "s3_topk": _mean([c.s3_topk for c in comps if c.s3_available]),
    }


def subset_available(subset: Sequence[str], comp: SignalComponents) -> bool:
    """A subset can be scored iff at least one of its blocks is available.
    (Otherwise the pipeline degrades honestly and routes to Verify.)"""
    avail = {"structural": comp.s1_available,
             "encoder": comp.s2_available,
             "entailment": comp.s3_available}
    return any(avail[b] for b in subset)


def build_training_matrix(
    subset: Sequence[str],
    comps: Sequence[SignalComponents],
    labels: Sequence[int],
    impute: Dict[str, float],
    augment_missing_entailment: bool = True,
):
    """Stack per-row feature vectors for meta-training. When entailment is one of
    several blocks, each entailment-present row is duplicated with entailment
    MASKED (imputed + flag 0, same label) so the LR learns the Stage-3-absent
    fallback regime — the training sources all have context, so without this the
    ``s3_avail`` flag would be constant and unlearnable. Single-signal entailment
    (config C) is never augmented: it degrades honestly when context is absent."""
    X: List[np.ndarray] = []
    y: List[int] = []
    for comp, lab in zip(comps, labels):
        X.append(build_vector(subset, comp, impute))
        y.append(int(lab))
    if augment_missing_entailment and "entailment" in subset and len(subset) > 1:
        for comp, lab in zip(comps, labels):
            if comp.s3_available:
                X.append(build_vector(subset, comp.masked_entailment(), impute))
                y.append(int(lab))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def build_vector(
    subset: Sequence[str], comp: SignalComponents, impute: Dict[str, float]
) -> np.ndarray:
    """Feature vector for a subset. Missing blocks are mean-imputed; entailment
    additionally carries an availability flag (1 present / 0 imputed)."""
    vec: List[float] = []
    for block in CANON_ORDER:
        if block not in subset:
            continue
        if block == "structural":
            vec.append(comp.s1_prob if comp.s1_available else impute["s1_prob"])
        elif block == "encoder":
            if comp.s2_available:
                vec += [comp.s2_mean, comp.s2_max, comp.s2_topk]
            else:
                vec += [impute["s2_mean"], impute["s2_max"], impute["s2_topk"]]
        elif block == "entailment":
            if comp.s3_available:
                vec += [comp.s3_mean, comp.s3_max, comp.s3_topk, 1.0]
            else:
                vec += [impute["s3_mean"], impute["s3_max"], impute["s3_topk"], 0.0]
    return np.asarray(vec, dtype=float)
