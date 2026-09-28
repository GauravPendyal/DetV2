"""Public training data (HaluEval) for the learned stages.

HaluEval (`pminervini/HaluEval`) ships paired right/hallucinated answers, so each
source row yields one supported (label 0) and one hallucinated (label 1) example
— naturally balanced. We use three configs as **groups** for dataset-wise
splitting: qa, dialogue, summarization.

Generalization discipline:
  * The local `Datasets/*.jsonl` are NEVER used for training — only held-out eval.
  * One HaluEval group can be held out entirely (see tools/train_stage1.py) to
    measure cross-group transfer before we ever touch the local sets.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

from datasets import load_dataset

_HALUEVAL = "pminervini/HaluEval"

# (config, query_field, context_field, right_field, hallucinated_field)
_CONFIGS = {
    "qa": ("question", "knowledge", "right_answer", "hallucinated_answer"),
    "dialogue": ("dialogue_history", "knowledge", "right_response", "hallucinated_response"),
    "summarization": (None, "document", "right_summary", "hallucinated_summary"),
}


@dataclass
class TrainRow:
    query: str
    response: str
    context: Optional[str]
    label: int          # 1 = hallucinated, 0 = supported
    group: str          # HaluEval config name


def load_halueval(
    configs: Optional[List[str]] = None,
    max_pairs_per_config: int = 1500,
    seed: int = 13,
) -> List[TrainRow]:
    configs = configs or list(_CONFIGS)
    rng = random.Random(seed)
    rows: List[TrainRow] = []
    for cfg in configs:
        qf, cf, rf, hf = _CONFIGS[cfg]
        ds = load_dataset(_HALUEVAL, cfg, split="data")
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        idx = idx[:max_pairs_per_config]
        for i in idx:
            r = ds[i]
            query = str(r.get(qf, "")) if qf else ""
            context = str(r.get(cf, "")) if cf else None
            right = str(r.get(rf, "")).strip()
            hall = str(r.get(hf, "")).strip()
            if right:
                rows.append(TrainRow(query, right, context, 0, cfg))
            if hall:
                rows.append(TrainRow(query, hall, context, 1, cfg))
    rng.shuffle(rows)
    return rows
