"""Fresh EXTERNAL grounded set for fusion meta-training (never seen by any stage).

The Stage-6 fusion meta-learner must be trained on base-signal predictions that
are *out-of-sample* for every base model (else stacking leaks). HaluEval
``summarization`` is one such source (held out from Stages 1/2/3); this module
adds a second, more diverse one: a public fact-verification set that no Detector
V2 stage has ever touched.

Primary: ``tals/vitaminc`` — ``evidence`` (short passage) is the grounding
context, ``claim`` is the response, ``label`` maps SUPPORTS -> 0 (grounded) /
REFUTES -> 1 (unsupported); NOT ENOUGH INFO is dropped so the label is clean and
binary. Fallback: ``pietrolesci/nli_fever`` (premise/hypothesis/label) if
VitaminC fails to load. Evidence is short, so the Stage-3 entailment path over
these rows stays cheap.

Leakage discipline: this set is used for the FUSION layer only. It is not
RAGTruth and not the local 769 rows (both remain held-out test), and no base
signal was trained on it, so its base-signal features are genuinely
out-of-sample.
"""
from __future__ import annotations

import random
from typing import List, Optional

from datasets import load_dataset

from .public_train import TrainRow  # reuse the (query, response, context, label, group) shape

# (dataset_id, split, context_field, claim_field, label_field)
_SPECS = {
    "tals/vitaminc": ("train", "evidence", "claim", "label"),
    "pietrolesci/nli_fever": ("train", "premise", "hypothesis", "label"),
}

# Normalize heterogeneous label spellings -> {0 grounded/entailed, 1 unsupported/refuted, None drop}.
_LABEL_MAP = {
    "supports": 0, "support": 0, "entailment": 0, "entailed": 0, "true": 0, "0": 0,
    "refutes": 1, "refute": 1, "contradiction": 1, "contradict": 1, "false": 1, "2": 1,
    "not enough info": None, "nei": None, "neutral": None, "1": None,
}


def _norm_label(raw) -> Optional[int]:
    """VitaminC uses SUPPORTS/REFUTES/NOT ENOUGH INFO strings; NLI-FEVER uses
    entailment/neutral/contradiction (or 0/1/2). Map both, drop the middle."""
    return _LABEL_MAP.get(str(raw).strip().lower())


def load_external_grounded(
    dataset_id: str = "tals/vitaminc",
    max_rows: int = 1500,
    seed: int = 13,
    stratify: bool = True,
    fallback: Optional[str] = "pietrolesci/nli_fever",
) -> List[TrainRow]:
    """Load a capped, label-stratified sample of the external grounded set as
    ``TrainRow``s (query="", context=evidence, response=claim, label, group)."""
    try:
        rows = _load(dataset_id)
        group = dataset_id.split("/")[-1]
    except Exception as exc:  # noqa: BLE001 - fall back to the alternate grounded set
        if not fallback or fallback == dataset_id:
            raise
        print(f"external_grounded: {dataset_id} failed ({type(exc).__name__}: {exc}); "
              f"falling back to {fallback}")
        rows = _load(fallback)
        group = fallback.split("/")[-1]

    for r in rows:
        r.group = group

    rng = random.Random(seed)
    if max_rows is None or max_rows >= len(rows):
        rng.shuffle(rows)
        return rows

    if not stratify:
        rng.shuffle(rows)
        return rows[:max_rows]

    # Proportional stratified sample over label so the pos/neg balance is preserved.
    strata: dict = {}
    for r in rows:
        strata.setdefault(r.label, []).append(r)
    for v in strata.values():
        rng.shuffle(v)
    out: List[TrainRow] = []
    total = len(rows)
    for group_rows in strata.values():
        take = max(1, round(max_rows * len(group_rows) / total))
        out.extend(group_rows[:take])
    rng.shuffle(out)
    return out[:max_rows]


def _load(dataset_id: str) -> List[TrainRow]:
    if dataset_id not in _SPECS:
        raise KeyError(f"unknown external grounded dataset {dataset_id!r}; "
                       f"choose from {list(_SPECS)}")
    split, ctx_field, claim_field, label_field = _SPECS[dataset_id]
    ds = load_dataset(dataset_id, split=split)
    rows: List[TrainRow] = []
    for r in ds:
        label = _norm_label(r.get(label_field))
        if label is None:
            continue
        context = str(r.get(ctx_field, "")).strip()
        response = str(r.get(claim_field, "")).strip()
        if not context or not response:
            continue
        rows.append(TrainRow(query="", response=response, context=context,
                             label=label, group=dataset_id.split("/")[-1]))
    if not rows:
        raise ValueError(f"{dataset_id}: no usable rows after label/field filtering")
    return rows
