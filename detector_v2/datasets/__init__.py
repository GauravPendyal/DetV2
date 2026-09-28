"""Dataset access.

Stage 0 only needs the local labeled eval sets (held-out; never trained on).
Public training-corpus loaders (HaluEval/RAGTruth/AggreFact) arrive with the
Stage-2 encoder, with dataset-wise splits and one domain fully held out.
"""
from __future__ import annotations

from .local_eval import EvalRow, load_local_dataset, LOCAL_DATASETS

__all__ = ["EvalRow", "load_local_dataset", "LOCAL_DATASETS"]
