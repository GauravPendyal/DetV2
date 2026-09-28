"""Signal strategy interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..claims.base import Claim
from ..schemas import SignalResult


class SignalContext:
    """Everything a signal *might* need. Signals read what they can and report
    unavailable for what they cannot."""

    def __init__(
        self,
        user_query: str = "",
        context: Optional[str] = None,
        response: str = "",
        logprobs=None,
        domain: str = "general",
    ):
        self.user_query = user_query
        self.context = context
        self.response = response
        self.logprobs = logprobs
        self.domain = domain


class Signal(ABC):
    """Compute one ``SignalResult`` per claim. Must never raise on ordinary
    missing-input conditions — return ``available=False`` instead."""

    name: str = "signal"

    def prepare(self, claims: "List[Claim]", ctx: SignalContext) -> None:
        """Optional batched precompute hook, called once per response before the
        per-claim ``score`` loop. Cheap signals ignore it; a neural signal uses
        it to run all of a response's claims through the model in ONE batch and
        cache the results, so ``score`` becomes a lookup. Must never raise on
        ordinary failure — degrade so ``score`` reports ``available=False``."""
        return None

    @abstractmethod
    def score(self, claim: Claim, ctx: SignalContext) -> SignalResult:
        raise NotImplementedError

    def unavailable(self, reason: str) -> SignalResult:
        return SignalResult(name=self.name, available=False, value=None, unavailable_reason=reason)
