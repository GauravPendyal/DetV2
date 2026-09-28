"""MiniCheck-style grounded entailment signal — Stage 3 (context-aware, optional).

An ORIGINAL ``(context, claim) -> P(unsupported)`` classifier we fine-tune
ourselves (see ``tools/train_stage3.py``), applied MiniCheck-style: chunk the
document, take the best-supporting chunk per claim. It abstains honestly when no
context is present. This is NOT MiniCheck's downloaded model.
"""
from .signal import EntailmentSignal, chunk_context  # noqa: F401

__all__ = ["EntailmentSignal", "chunk_context"]
