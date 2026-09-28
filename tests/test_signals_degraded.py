from detector_v2.claims.base import Claim
from detector_v2.schemas import SignalResult
from detector_v2.scoring.heuristic import HeuristicScorer
from detector_v2.signals.base import Signal, SignalContext


class _UnavailableSignal(Signal):
    name = "needs_context"

    def score(self, claim, ctx):
        if not ctx.context:
            return self.unavailable("no_context")
        return SignalResult(name=self.name, available=True, value=0.9)


def test_unavailable_signal_reports_reason_not_value():
    r = _UnavailableSignal().score(Claim(0, "x", [0, 1]), SignalContext(context=None))
    assert r.available is False
    assert r.value is None
    assert r.unavailable_reason == "no_context"


def test_scorer_ignores_unavailable_signals():
    scorer = HeuristicScorer()
    signals = [
        SignalResult(name="a", available=True, value=0.8),
        SignalResult(name="needs_context", available=False, value=None),
    ]
    # Only the available 0.8 should count — not averaged against a phantom 0.
    assert scorer.score_claim(signals) == 0.8


def test_scorer_returns_none_when_all_unavailable():
    scorer = HeuristicScorer()
    signals = [SignalResult(name="a", available=False, value=None)]
    assert scorer.score_claim(signals) is None


def test_scorer_weighted_mean():
    scorer = HeuristicScorer(weights={"a": 3.0, "b": 1.0})
    signals = [
        SignalResult(name="a", available=True, value=1.0),
        SignalResult(name="b", available=True, value=0.0),
    ]
    assert abs(scorer.score_claim(signals) - 0.75) < 1e-9
