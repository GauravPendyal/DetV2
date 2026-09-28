"""Claim scorer interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..schemas import SignalResult


class ClaimScorer(ABC):
    """Fuse a claim's available signals into a single risk in [0, 1].

    Implementations MUST ignore signals with ``available=False`` rather than
    treating a missing value as 0 or 0.5.
    """

    @abstractmethod
    def score_claim(self, signals: List[SignalResult]) -> Optional[float]:
        """Return fused risk, or None if no signal was available."""
        raise NotImplementedError

    @staticmethod
    def available_values(signals: List[SignalResult]) -> List[float]:
        return [s.value for s in signals if s.available and s.value is not None]
