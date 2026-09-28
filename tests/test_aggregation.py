import math

from detector_v2.aggregation.aggregators import (
    agg_max, agg_mean, agg_noisy_or, agg_topk_mean, aggregate,
)


def test_empty_is_zero():
    for fn in (agg_max, agg_mean, agg_topk_mean, agg_noisy_or):
        assert fn([]) == 0.0


def test_max():
    assert agg_max([0.1, 0.9, 0.3]) == 0.9


def test_mean():
    assert abs(agg_mean([0.2, 0.4, 0.6]) - 0.4) < 1e-9


def test_topk_mean_resists_dilution():
    vals = [0.9] + [0.0] * 9
    # mean would bury the 0.9; top-1 recovers it.
    assert agg_topk_mean(vals, k=1) == 0.9
    assert agg_mean(vals) < 0.1


def test_noisy_or_bounds_and_value():
    assert abs(agg_noisy_or([0.5, 0.5]) - 0.75) < 1e-9
    assert 0.0 <= agg_noisy_or([0.3, 0.3, 0.3]) <= 1.0


def test_clamps_out_of_range():
    assert agg_max([1.5, -0.2]) == 1.0


def test_dispatch_default():
    assert aggregate([0.9, 0.1], method="unknown_defaults_to_topk", k=1) == 0.9
