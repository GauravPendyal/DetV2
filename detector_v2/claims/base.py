"""Claim segmenter interface + the Claim value object."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class Claim:
    """A single risk-unit extracted from a response.

    ``span`` is the [start, end) char offset into the original response so
    downstream reporting can point back at the exact text.
    """

    claim_id: int
    text: str
    span: List[int] = field(default_factory=lambda: [0, 0])


class ClaimSegmenter(ABC):
    """Strategy interface. Stage 0 ships a rule-based implementation; later
    stages may add a spaCy/LLM decomposer behind this same interface without
    touching any caller."""

    @abstractmethod
    def segment(self, text: str) -> List[Claim]:
        raise NotImplementedError
