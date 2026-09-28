"""DeBERTa-v3 claim-level encoder — the always-on learned detector signal.

This is an ORIGINAL classification head fine-tuned by us on a general-purpose
DeBERTa-v3 language-model backbone (``microsoft/deberta-v3-*``). It is NOT a
downloaded hallucination detector, NOT the HaluEval DistilBERT, and NOT a wrapped
third-party model — the backbone is a plain masked-LM encoder that has never seen
a hallucination label until we train it here.

Input is a ``(query, claim)`` pair; output is P(claim is hallucinated) in [0, 1].
Response-level risk comes from aggregating per-claim scores upstream, so the model
stays claim-level. torch/transformers are imported lazily so the rest of Detector
V2 (Stages 0-1) keeps working without the neural dependency installed.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_BACKBONE = "microsoft/deberta-v3-xsmall"
_META = "encoder_meta.json"

Pair = Tuple[str, str]


def _torch():
    import torch  # local import: neural deps are optional until Stage 2 is used
    return torch


class DebertaClaimEncoder:
    """Thin wrapper around a sequence-classification DeBERTa-v3: load / predict /
    save. Training lives in ``train.py`` so this class stays inference-focused and
    trivially testable with an injected tiny model."""

    def __init__(self, tokenizer, model, *, max_len: int = 128, model_name: str = DEFAULT_BACKBONE,
                 device: Optional[str] = None, metadata: Optional[dict] = None):
        torch = _torch()
        self.tokenizer = tokenizer
        self.model = model
        self.max_len = max_len
        self.model_name = model_name
        self.metadata = metadata or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    # -- construction ----------------------------------------------------
    @classmethod
    def from_pretrained(cls, model_name: str = DEFAULT_BACKBONE, *, num_labels: int = 2,
                        max_len: int = 128, device: Optional[str] = None) -> "DebertaClaimEncoder":
        """Fresh head on a pretrained backbone — the starting point for training."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = _load_tokenizer(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        return cls(tok, model, max_len=max_len, model_name=model_name, device=device)

    @classmethod
    def load(cls, path: str, *, max_len: Optional[int] = None,
             device: Optional[str] = None) -> "DebertaClaimEncoder":
        """Load a fine-tuned encoder saved with :meth:`save`."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if not path or not os.path.isdir(path):
            raise FileNotFoundError(f"encoder model dir not found: {path!r}")
        meta = {}
        meta_path = os.path.join(path, _META)
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        return cls(
            tok, model,
            max_len=max_len or int(meta.get("max_len", 128)),
            model_name=meta.get("model_name", DEFAULT_BACKBONE),
            device=device, metadata=meta,
        )

    # -- inference -------------------------------------------------------
    def predict_proba(self, pairs: Sequence[Pair], batch_size: int = 16) -> np.ndarray:
        """P(hallucinated) for each ``(query, claim)`` pair. Batched, no-grad."""
        torch = _torch()
        if not pairs:
            return np.zeros(0, dtype=float)
        self.model.eval()
        out: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                a = [(p[0] or "") for p in batch]
                b = [(p[1] or "") for p in batch]
                enc = self.tokenizer(
                    a, b, truncation=True, max_length=self.max_len,
                    padding=True, return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1)[:, 1]
                out.append(probs.detach().cpu().numpy())
        return np.concatenate(out)

    # -- persistence -----------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        meta = dict(self.metadata)
        meta.setdefault("model_name", self.model_name)
        meta.setdefault("max_len", self.max_len)
        with open(os.path.join(path, _META), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)


def _load_tokenizer(model_name: str):
    """DeBERTa-v3 ships a SentencePiece tokenizer; the fast variant needs the
    conversion which occasionally errors, so fall back to the slow tokenizer."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception:  # noqa: BLE001 - fall back to slow (sentencepiece) tokenizer
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)
