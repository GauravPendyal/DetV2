"""Rule-based, dependency-free claim segmenter.

Sentence segmentation with light protection for decimals and common
abbreviations, plus a conservative semicolon split. This is intentionally simple
for the Stage-0 baseline; a spaCy/LLM decomposer can be dropped in later behind
the same ``ClaimSegmenter`` interface.
"""
from __future__ import annotations

import re
from typing import List

from .base import Claim, ClaimSegmenter

# Abbreviations whose trailing period must not end a sentence.
_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "u.s", "u.k", "fig", "no", "vol", "inc", "ltd", "co", "gov", "gen", "sen",
}
# Sentence boundary: ., !, or ? followed by whitespace and an uppercase/quote/digit.
_BOUNDARY = re.compile(r'([.!?]+)(\s+)(?=["\'(\[]?[A-Z0-9])')


class RuleBasedSegmenter(ClaimSegmenter):
    def __init__(self, min_claim_chars: int = 8, max_claims: int = 64, split_semicolons: bool = True):
        self.min_claim_chars = min_claim_chars
        self.max_claims = max_claims
        self.split_semicolons = split_semicolons

    def segment(self, text: str) -> List[Claim]:
        if not text or not text.strip():
            return []

        spans = self._sentence_spans(text)
        if self.split_semicolons:
            spans = self._split_on_semicolons(text, spans)
        spans = self._merge_short(text, spans)

        claims: List[Claim] = []
        for start, end in spans[: self.max_claims]:
            snippet = text[start:end].strip()
            if not snippet:
                continue
            # Recover exact trimmed span.
            lead = len(text[start:end]) - len(text[start:end].lstrip())
            s = start + lead
            claims.append(Claim(claim_id=len(claims), text=snippet, span=[s, s + len(snippet)]))
        return claims

    # -- internals -------------------------------------------------------
    def _sentence_spans(self, text: str) -> List[List[int]]:
        spans: List[List[int]] = []
        start = 0
        for m in _BOUNDARY.finditer(text):
            end = m.end(1)  # include the punctuation, drop the following space
            candidate = text[start:end]
            last_word = re.split(r"[\s]", candidate.strip().rstrip(".!?"))[-1].lower()
            if last_word in _ABBREV or re.search(r"\d[.]\d*$", candidate.rstrip()):
                continue  # false boundary (abbreviation or decimal)
            spans.append([start, end])
            start = m.end()  # skip the whitespace matched in group 2
        if start < len(text):
            spans.append([start, len(text)])
        return spans

    def _split_on_semicolons(self, text: str, spans: List[List[int]]) -> List[List[int]]:
        out: List[List[int]] = []
        for start, end in spans:
            seg_start = start
            for m in re.finditer(r";", text[start:end]):
                pos = start + m.start()
                out.append([seg_start, pos + 1])
                seg_start = pos + 1
            out.append([seg_start, end])
        return [s for s in out if text[s[0]:s[1]].strip()]

    def _merge_short(self, text: str, spans: List[List[int]]) -> List[List[int]]:
        out: List[List[int]] = []
        for span in spans:
            snippet = text[span[0]:span[1]].strip()
            if out and len(snippet) < self.min_claim_chars:
                out[-1][1] = span[1]  # glue tiny fragment onto previous claim
            else:
                out.append(list(span))
        return out
