import numpy as np
import pytest

from ragcheck.generation_metrics import (
    answer_relevancy,
    context_precision,
    cosine_similarity,
    faithfulness,
)


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_does_not_raise():
    a = np.zeros(3)
    b = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, b) == 0.0


def test_answer_relevancy_is_cosine_similarity():
    q = np.array([1.0, 0.0])
    a = np.array([1.0, 0.0])
    assert answer_relevancy(q, a) == pytest.approx(1.0)


def test_faithfulness_picks_best_matching_context():
    answer = np.array([1.0, 0.0])
    good_context = np.array([1.0, 0.0])
    bad_context = np.array([0.0, 1.0])
    assert faithfulness(answer, [bad_context, good_context]) == pytest.approx(1.0)


def test_faithfulness_no_contexts_is_zero():
    assert faithfulness(np.array([1.0, 0.0]), []) == 0.0


def test_context_precision_counts_only_above_threshold():
    ground_truth = np.array([1.0, 0.0])
    relevant_ctx = np.array([0.99, 0.14])   # high similarity
    irrelevant_ctx = np.array([0.0, 1.0])   # zero similarity
    precision = context_precision(ground_truth, [relevant_ctx, irrelevant_ctx], threshold=0.5)
    assert precision == pytest.approx(0.5)


def test_context_precision_empty_contexts_is_zero():
    assert context_precision(np.array([1.0, 0.0]), [], threshold=0.5) == 0.0
