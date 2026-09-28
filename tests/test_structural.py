from detector_v2.claims.base import Claim
from detector_v2.signals.base import SignalContext
from detector_v2.signals.structural.features import extract_features
from detector_v2.signals.structural.signal import StructuralSignal


def test_numbers_and_dates_detected():
    f = extract_features("In 1889 the tower reached 300 metres in height.")
    assert f.num_count >= 2      # 1889 and 300
    assert f.date_count >= 1     # 1889 as a year


def test_superlative_and_absolute_detected():
    f = extract_features("This is the best and largest device that everyone always uses.")
    assert f.superlative_count >= 2   # best, largest
    assert f.absolute_count >= 2      # everyone, always


def test_hedge_detected():
    f = extract_features("This might possibly be roughly correct.")
    assert f.hedge_count >= 2


def test_false_premise_cue_detected():
    f = extract_features("As we all know, the earth is obviously flat.")
    assert f.false_premise_cue_count >= 1


def test_signal_value_in_range():
    sig = StructuralSignal()
    r = sig.score(Claim(0, "Some ordinary sentence here.", [0, 10]), SignalContext())
    assert r.available is True
    assert 0.0 <= r.value <= 1.0


def test_assertive_specific_scores_higher_than_hedged_generic():
    sig = StructuralSignal()
    ctx = SignalContext()
    assertive = sig.score(
        Claim(0, "The company definitely earned exactly 4.2 billion dollars in 1999.", [0, 1]), ctx
    )
    hedged = sig.score(
        Claim(1, "This might generally be somewhat helpful in some cases.", [0, 1]), ctx
    )
    assert assertive.value > hedged.value
