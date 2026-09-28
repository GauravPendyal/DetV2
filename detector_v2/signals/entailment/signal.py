"""MiniCheck-STYLE grounded entailment signal — Stage 3 (context-aware, optional).

Methodology (not the model): like MiniCheck, we chunk the source document and ask,
per claim, whether ANY chunk supports it — a claim is grounded if its best-supporting
chunk entails it, so ``risk = min_chunk P(not-supported | chunk, claim)``. The model
is OUR OWN fine-tuned ``(context, claim) -> P(unsupported)`` classifier (see
``tools/train_stage3.py``); we do not download or wrap MiniCheck's weights.

Honest degradation is the whole point of this stage: when no context is present we
report ``available=False`` with reason ``context_absent`` and invent NOTHING. The
always-on encoder (Stage 2) still carries the response; this signal simply abstains.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from ...claims.base import Claim
from ...schemas import SignalResult
from ..base import Signal, SignalContext

_SENT = re.compile(r"(?<=[.!?])\s+")


def chunk_context(context: str, chunk_chars: int = 900, overlap_sents: int = 1,
                  max_chunks: int = 12) -> List[str]:
    """Split a document into overlapping sentence-grouped chunks, bounded so a
    long document cannot blow up latency. Falls back to a hard char split if the
    text has no sentence breaks."""
    text = (context or "").strip()
    if not text:
        return []
    sents = [s.strip() for s in _SENT.split(text) if s.strip()]
    if not sents:
        sents = [text]
    chunks: List[str] = []
    i = 0
    while i < len(sents):
        cur: List[str] = []
        size = 0
        j = i
        while j < len(sents) and (size + len(sents[j]) <= chunk_chars or not cur):
            cur.append(sents[j])
            size += len(sents[j]) + 1
            j += 1
        chunks.append(" ".join(cur))
        if j >= len(sents):
            break
        i = max(j - overlap_sents, i + 1)  # step back for overlap, always progress
        if len(chunks) >= max_chunks:
            break
    return chunks


class EntailmentSignal(Signal):
    name = "entailment"

    def __init__(
        self,
        model_dir: Optional[str] = None,
        model=None,
        *,
        max_len: int = 256,
        batch_size: int = 16,
        chunk_chars: int = 900,
        max_chunks: int = 12,
    ):
        self._model_dir = model_dir
        self._model = model
        self._loaded = model is not None
        self._load_error: Optional[str] = None
        self.max_len = max_len
        self.batch_size = batch_size
        self.chunk_chars = chunk_chars
        self.max_chunks = max_chunks
        self._cache: Dict[int, float] = {}
        self._reason: Optional[str] = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            from ..encoder.model import DebertaClaimEncoder
            self._model = DebertaClaimEncoder.load(self._model_dir, max_len=self.max_len)
        except Exception as exc:  # noqa: BLE001
            self._model = None
            self._load_error = f"entailment_load_failed: {type(exc).__name__}: {exc}"

    def prepare(self, claims: List[Claim], ctx: SignalContext) -> None:
        self._cache = {}
        self._reason = None
        # Context-absent is the defining case: abstain, never fabricate evidence.
        if not ctx.context or not str(ctx.context).strip():
            self._reason = "context_absent"
            return
        self._ensure_loaded()
        if self._model is None:
            self._reason = self._load_error or "entailment_model_unavailable"
            return
        chunks = chunk_context(ctx.context, self.chunk_chars, max_chunks=self.max_chunks)
        if not chunks:
            self._reason = "context_absent"
            return
        # One batched pass over every (chunk, claim) pair; reduce to best support.
        pairs = [(chunk, c.text) for c in claims for chunk in chunks]
        try:
            probs = self._model.predict_proba(pairs, batch_size=self.batch_size)
        except Exception as exc:  # noqa: BLE001
            self._reason = f"entailment_predict_failed: {type(exc).__name__}: {exc}"
            return
        m = len(chunks)
        for ci, claim in enumerate(claims):
            window = probs[ci * m:(ci + 1) * m]
            # risk = min over chunks of P(unsupported): the best-supporting chunk wins.
            self._cache[claim.claim_id] = float(min(window)) if len(window) else 0.5

    def score(self, claim: Claim, ctx: SignalContext) -> SignalResult:
        if claim.claim_id in self._cache:
            v = self._cache[claim.claim_id]
            return SignalResult(
                name=self.name, available=True, value=round(v, 6),
                features={"entailment_unsupported": round(v, 6)},
            )
        return self.unavailable(self._reason or "context_absent")
