"""DeBERTa-v3 claim-level encoder signal — Stage 2.

The always-on learned detector signal: an ORIGINAL classification head we
fine-tune on a general-purpose DeBERTa-v3 backbone over heterogeneous data,
scoring per-claim hallucination risk. It is deliberately NOT the HaluEval
DistilBERT and NOT a wrapped third-party detector — see ``model.py``.

Neural deps (torch/transformers) are imported lazily inside these modules, so
importing this package is cheap and Stages 0-1 run without them installed.
"""
from .model import DebertaClaimEncoder  # noqa: F401
from .signal import EncoderSignal  # noqa: F401

__all__ = ["DebertaClaimEncoder", "EncoderSignal"]
