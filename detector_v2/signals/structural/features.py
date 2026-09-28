"""Structural feature extraction over a single claim.

Pure-Python (regex + curated lexicons), so it runs with zero model downloads.
Features capture *checkability* and *assertion style* — a claim dense with
specific, checkable facts stated with high certainty is riskier to leave
un-verified than a hedged, generic sentence.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_NUMBER = re.compile(r"(?<![A-Za-z])[$£€]?\d[\d,]*(?:\.\d+)?%?")
_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_MONTHS = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
_PROPER = re.compile(r"\b([A-Z][a-z]+)\b")

_SUPERLATIVE = {
    "best", "worst", "most", "least", "largest", "smallest", "greatest", "highest",
    "lowest", "first", "last", "only", "biggest", "fastest", "oldest", "newest",
}
_ABSOLUTE = {
    "all", "every", "everyone", "everybody", "everything", "none", "nobody",
    "never", "always", "no", "nothing", "entirely", "completely", "totally",
    "absolutely", "any", "each",
}
_OVERASSERT = {
    "definitely", "certainly", "undoubtedly", "guaranteed", "proven", "clearly",
    "obviously", "surely", "indisputably", "unquestionably", "irrefutable",
}
_HEDGE = {
    "may", "might", "could", "possibly", "perhaps", "reportedly", "allegedly",
    "seems", "appears", "approximately", "around", "roughly", "likely", "probably",
    "suggests", "estimated",
}
_FALSE_PREMISE = [
    "as we all know", "it is well known", "everyone knows", "needless to say",
    "of course", "as is well known", "it is a fact that", "it is common knowledge",
]


@dataclass
class StructuralFeatures:
    n_tokens: int = 0
    num_count: int = 0
    date_count: int = 0
    proper_noun_count: int = 0
    superlative_count: int = 0
    absolute_count: int = 0
    overassertion_count: int = 0
    hedge_count: int = 0
    false_premise_cue_count: int = 0
    specificity: float = 0.0  # density of checkable specifics per token

    def as_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def extract_features(text: str) -> StructuralFeatures:
    lowered = text.lower()
    words = _WORD.findall(text)
    n_tokens = max(len(words), 1)

    tokens_lower = {w.lower() for w in words}
    num_count = len(_NUMBER.findall(text))
    date_count = len(_YEAR.findall(text)) + len(_MONTHS.findall(text))
    # Proper nouns: capitalized words that are not the first token of the claim.
    proper = _PROPER.findall(text)
    proper_noun_count = max(len(proper) - (1 if text[:1].isupper() else 0), 0)

    superlative_count = sum(1 for w in words if w.lower() in _SUPERLATIVE or w.lower().endswith("est") and len(w) > 4)
    absolute_count = sum(1 for w in words if w.lower() in _ABSOLUTE)
    overassertion_count = len(tokens_lower & _OVERASSERT)
    hedge_count = sum(1 for w in words if w.lower() in _HEDGE)
    false_premise_cue_count = sum(lowered.count(cue) for cue in _FALSE_PREMISE)

    specificity = (num_count + date_count + proper_noun_count) / n_tokens

    return StructuralFeatures(
        n_tokens=n_tokens,
        num_count=num_count,
        date_count=date_count,
        proper_noun_count=proper_noun_count,
        superlative_count=superlative_count,
        absolute_count=absolute_count,
        overassertion_count=overassertion_count,
        hedge_count=hedge_count,
        false_premise_cue_count=false_premise_cue_count,
        specificity=round(specificity, 4),
    )
