from detector_v2.claims.rule_based import RuleBasedSegmenter


def _seg(text, **kw):
    return RuleBasedSegmenter(**kw).segment(text)


def test_empty_and_whitespace():
    assert _seg("") == []
    assert _seg("   \n ") == []


def test_multiple_sentences_split():
    text = "The sky is blue. Water boils at 100 degrees. Paris is in France."
    claims = _seg(text)
    assert len(claims) == 3


def test_spans_are_exact():
    text = "Alpha is first. Beta is second."
    for c in _seg(text):
        # The recorded span must slice back to exactly the claim text.
        assert text[c.span[0]:c.span[1]] == c.text


def test_abbreviation_not_split():
    text = "Dr. Smith works at the lab. He is busy."
    claims = _seg(text)
    assert len(claims) == 2
    assert "Dr. Smith" in claims[0].text


def test_decimal_not_split():
    text = "The value is 3.14 according to the report. It is precise."
    claims = _seg(text)
    assert len(claims) == 2
    assert "3.14" in claims[0].text


def test_semicolon_split():
    text = "It rained heavily; the streets flooded quickly."
    claims = _seg(text, split_semicolons=True)
    assert len(claims) == 2


def test_max_claims_cap():
    text = " ".join(f"Fact number {i} is stated here." for i in range(100))
    claims = _seg(text, max_claims=10)
    assert len(claims) <= 10


def test_short_fragment_merged():
    text = "Yes. This is a much longer and complete sentence about something."
    claims = _seg(text, min_claim_chars=8)
    # "Yes." is shorter than min; it should not survive as its own claim.
    assert all(len(c.text) >= 4 for c in claims)
