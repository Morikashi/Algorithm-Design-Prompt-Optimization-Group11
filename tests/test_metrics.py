from src.evaluation.metrics import ExactMatchMetric, RougeLMetric


def test_exact_match_normalization():
    m = ExactMatchMetric()
    assert m.score("Paris", "paris").score == 1.0
    assert m.score("Paris!", "paris").score == 1.0
    assert m.score("The answer is Paris", "Paris").score == 0.0


def test_rouge_l_basic():
    m = RougeLMetric()
    s = m.score("the cat sat on the mat", "cat sat on mat").score
    assert 0.0 < s <= 1.0
    assert m.score("", "").score == 1.0
    assert m.score("hello", "").score == 0.0
