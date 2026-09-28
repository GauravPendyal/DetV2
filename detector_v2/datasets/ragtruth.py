"""Public RAGTruth loader (`wandb/RAGTruth-processed`).

RAGTruth is the fully-EXTERNAL grounded test set for Stage 3: every row carries a
real source ``context`` (the document/passages the response was meant to be
grounded in), so it is the honest place to ask whether grounding context helps —
the local ``halluciguard_ragtruth.jsonl`` had its source document stripped.

Leakage discipline: RAGTruth is NEVER used to train any Detector V2 stage. It is
load-for-eval only. Response-level label = any annotated hallucination span
(``evident_conflict`` or ``baseless_info``) present -> 1, else 0.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

from datasets import load_dataset

_RAGTRUTH = "wandb/RAGTruth-processed"


@dataclass
class RagTruthRow:
    query: str
    context: str          # the source document / passages (grounding evidence)
    response: str
    label: int            # 1 = contains hallucination, 0 = grounded
    task_type: str        # Summary / QA / Data2txt


def _label(row: dict) -> int:
    h = row.get("hallucination_labels_processed") or {}
    return 1 if (int(h.get("evident_conflict", 0)) + int(h.get("baseless_info", 0))) > 0 else 0


def load_ragtruth(
    split: str = "test",
    max_rows: Optional[int] = None,
    seed: int = 13,
    task_types: Optional[List[str]] = None,
    stratify: bool = True,
) -> List[RagTruthRow]:
    """Load RAGTruth rows with source context. ``max_rows`` samples (stratified by
    (task_type, label) when ``stratify``) so CPU eval stays tractable and balanced."""
    ds = load_dataset(_RAGTRUTH, split=split)
    rows: List[RagTruthRow] = []
    for r in ds:
        ctx = str(r.get("context", "")).strip()
        resp = str(r.get("output", "")).strip()
        if not ctx or not resp:
            continue
        tt = str(r.get("task_type", ""))
        if task_types and tt not in task_types:
            continue
        rows.append(RagTruthRow(str(r.get("query", "")), ctx, resp, _label(r), tt))

    if max_rows is None or max_rows >= len(rows):
        random.Random(seed).shuffle(rows)
        return rows

    rng = random.Random(seed)
    if not stratify:
        rng.shuffle(rows)
        return rows[:max_rows]

    # Proportional stratified sample over (task_type, label) strata.
    strata: dict = {}
    for r in rows:
        strata.setdefault((r.task_type, r.label), []).append(r)
    for v in strata.values():
        rng.shuffle(v)
    out: List[RagTruthRow] = []
    total = len(rows)
    for key, group in strata.items():
        take = max(1, round(max_rows * len(group) / total))
        out.extend(group[:take])
    rng.shuffle(out)
    return out[:max_rows]
