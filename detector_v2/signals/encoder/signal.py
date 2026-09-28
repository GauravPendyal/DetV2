"""``EncoderSignal`` — wires the DeBERTa-v3 claim encoder into the signal API.

Per-claim interface, but batched under the hood: :meth:`prepare` runs every claim
of a response through the model in one pass and caches the scores; :meth:`score`
is then a dict lookup. The encoder is dependency-injectable so tests can drive the
integration with a stub instead of a real model.

Honest degradation (per the brief): if the model dir is missing, torch is not
installed, or a forward pass throws, the signal reports ``available=False`` with a
reason. It never fabricates a low-risk score — the pipeline then degrades and
routes the response to Verify.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ...claims.base import Claim
from ...schemas import SignalResult
from ..base import Signal, SignalContext


class EncoderSignal(Signal):
    name = "encoder"

    def __init__(
        self,
        model_dir: Optional[str] = None,
        encoder=None,
        *,
        max_len: int = 128,
        batch_size: int = 16,
        condition_on_query: bool = True,
    ):
        self._model_dir = model_dir
        self._encoder = encoder
        self._loaded = encoder is not None
        self._load_error: Optional[str] = None
        self.max_len = max_len
        self.batch_size = batch_size
        self.condition_on_query = condition_on_query
        self._cache: Dict[int, float] = {}

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True  # attempt exactly once; don't retry per claim
        try:
            from .model import DebertaClaimEncoder
            self._encoder = DebertaClaimEncoder.load(self._model_dir, max_len=self.max_len)
        except Exception as exc:  # noqa: BLE001 - missing model/torch -> unavailable
            self._encoder = None
            self._load_error = f"encoder_load_failed: {type(exc).__name__}: {exc}"

    def prepare(self, claims: List[Claim], ctx: SignalContext) -> None:
        self._cache = {}
        self._ensure_loaded()
        if self._encoder is None or not claims:
            return
        query = ctx.user_query if self.condition_on_query else ""
        pairs = [(query, c.text) for c in claims]
        try:
            probs = self._encoder.predict_proba(pairs, batch_size=self.batch_size)
            for c, p in zip(claims, probs):
                self._cache[c.claim_id] = float(p)
        except Exception as exc:  # noqa: BLE001 - forward failure -> unavailable
            self._cache = {}
            self._load_error = f"encoder_predict_failed: {type(exc).__name__}: {exc}"

    def score(self, claim: Claim, ctx: SignalContext) -> SignalResult:
        if self._encoder is None:
            return self.unavailable(self._load_error or "encoder_model_unavailable")
        if claim.claim_id not in self._cache:
            # prepare() wasn't run (or failed) — score this claim on its own.
            query = ctx.user_query if self.condition_on_query else ""
            try:
                p = float(self._encoder.predict_proba([(query, claim.text)])[0])
                self._cache[claim.claim_id] = p
            except Exception as exc:  # noqa: BLE001
                return self.unavailable(f"encoder_predict_failed: {type(exc).__name__}: {exc}")
        v = self._cache[claim.claim_id]
        return SignalResult(
            name=self.name, available=True, value=round(v, 6),
            features={"encoder_prob": round(v, 6)},
        )
