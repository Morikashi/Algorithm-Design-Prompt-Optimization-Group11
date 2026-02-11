from src.analysis.vector_math import cosine_similarity


def test_cosine_similarity_basic():
    a = [1.0, 0.0]
    b = [1.0, 0.0]
    c = [0.0, 1.0]
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-9
    assert abs(cosine_similarity(a, c) - 0.0) < 1e-9
