"""Loader for the local HalluciGuard eval sets (EVAL-ONLY, held out from training).

Each row: ``{"input": {"user_prompt", "llm_response", "judge": {...}}}``.
Label from ``judge.overall_verdict``: hallucinated -> 1, supported -> 0.

LEAKAGE GUARD: ``judge.claims[].evidence[].text`` (the gold source/reasoning) is
NEVER exposed as detector input. Only ``user_prompt`` and ``llm_response`` are
returned. ``context`` stays None here precisely because these files do not carry
a clean grounding document (see project notes on the RAGTruth trap).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

# Resolve ../../Datasets relative to the SDC-II root, overridable via env.
_DEFAULT_DIR = os.getenv(
    "HALLUCIGUARD_DATASETS",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "Datasets")),
)

LOCAL_DATASETS = {
    "halluciguard_dataset": "halluciguard_dataset.jsonl",
    "ragtruth": "halluciguard_ragtruth.jsonl",
    "truthfulqa": "halluciguard_truthfulqa.jsonl",
}

_LABELS = {"hallucinated": 1, "supported": 0}


@dataclass
class EvalRow:
    dataset: str
    user_prompt: str
    llm_response: str
    label: int  # 1 = hallucinated, 0 = supported
    context: Optional[str] = None


def _row_from_obj(obj: dict, dataset: str) -> Optional[EvalRow]:
    inp = obj.get("input") or {}
    judge = inp.get("judge") or {}
    verdict = str(judge.get("overall_verdict", "")).strip().lower()
    if verdict not in _LABELS:
        return None
    resp = inp.get("llm_response")
    if not resp or not str(resp).strip():
        return None
    return EvalRow(
        dataset=dataset,
        user_prompt=str(inp.get("user_prompt", "")),
        llm_response=str(resp),
        label=_LABELS[verdict],
        context=None,  # deliberately not populated from judge.evidence (leakage)
    )


def load_local_dataset(name: str, base_dir: str = _DEFAULT_DIR) -> List[EvalRow]:
    if name not in LOCAL_DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {list(LOCAL_DATASETS)}")
    path = os.path.join(base_dir, LOCAL_DATASETS[name])
    rows: List[EvalRow] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = _row_from_obj(json.loads(line), name)
            if row is not None:
                rows.append(row)
    return rows
