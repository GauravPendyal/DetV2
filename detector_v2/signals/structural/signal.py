"""Structural signal: map red-flag features onto a heuristic risk value.

The weights below are **hand-set priors**, not learned — this is the Stage-0
baseline whose whole job is to establish an honest floor. Stage 1 replaces this
fixed logistic with a logistic-regression model fit on the same features, and
later stages add learned signals alongside it. Nothing here is presented as a
calibrated probability.
"""
from __future__ import annotations

import math

from ...claims.base import Claim
from ...schemas import SignalResult
from ..base import Signal, SignalContext
from .features import extract_features

# Heuristic priors on feature *densities* (per-token unless noted).
_BIAS = -1.6
_WEIGHTS = {
    "specificity": 2.5,          # dense checkable specifics -> riskier to leave unverified
    "superlative_density": 4.0,  # superlatives are often unverifiable/overstated
    "absolute_density": 3.0,     # sweeping absolutes overgeneralize
    "overassert_density": 4.0,   # high certainty without support
    "false_premise_density": 3.0,
    "hedge_density": -2.0,       # explicit hedging = lower factual commitment
}


def _sigmoid(x: float) -> float:
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


class StructuralSignal(Signal):
    name = "structural"

    def score(self, claim: Claim, ctx: SignalContext) -> SignalResult:
        f = extract_features(claim.text)
        n = float(f.n_tokens)

        densities = {
            "specificity": f.specificity,
            "superlative_density": f.superlative_count / n,
            "absolute_density": f.absolute_count / n,
            "overassert_density": f.overassertion_count / n,
            "false_premise_density": min(f.false_premise_cue_count, 2) / 1.0,
            "hedge_density": f.hedge_count / n,
        }
        z = _BIAS + sum(_WEIGHTS[k] * v for k, v in densities.items())
        value = _sigmoid(z)

        features = f.as_dict()
        features.update({f"d_{k}": round(v, 4) for k, v in densities.items()})
        return SignalResult(name=self.name, available=True, value=round(value, 6), features=features)
