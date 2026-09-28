import numpy as np

from detector_v2.features.response import featurize_claims, featurize_text, feature_names
from detector_v2.pipeline import DetectorV2Pipeline


def test_feature_vector_length_matches_names():
    v = featurize_text("q", "The tower was built in 1889 and is the tallest one.")
    assert len(v) == len(feature_names())


def test_text_and_claims_featurizers_agree():
    query = "What happened?"
    response = "In 1889 the tower was built. It is definitely the tallest structure in Paris."
    out = DetectorV2Pipeline().detect(query, response)
    v_text = featurize_text(query, response)
    v_claims = featurize_claims(out.claims)
    # Same segmenter + structural signal on both paths -> identical vectors.
    assert np.allclose(v_text, v_claims)


def test_empty_response_is_zero_vector():
    v = featurize_text("q", "")
    assert np.allclose(v, np.zeros(len(feature_names())))
