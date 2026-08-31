import math

import pytest

from ragcheck.retrieval_metrics import (
    dcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k_all_relevant():
    assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0


def test_precision_at_k_partial():
    assert precision_at_k(["a", "b", "c"], {"a"}, 3) == pytest.approx(1 / 3)


def test_precision_at_k_none_relevant():
    assert precision_at_k(["a", "b", "c"], {"x", "y"}, 3) == 0.0


def test_precision_at_k_respects_k():
    # Only the first 2 are considered even though a relevant doc is 3rd.
    assert precision_at_k(["a", "b", "c"], {"c"}, 2) == 0.0


def test_precision_at_k_empty_retrieved():
    assert precision_at_k([], {"a"}, 3) == 0.0


def test_precision_at_k_invalid_k():
    assert precision_at_k(["a"], {"a"}, 0) == 0.0


def test_recall_at_k_all_found():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0


def test_recall_at_k_partial():
    assert recall_at_k(["a", "b", "c"], {"a", "z"}, 3) == pytest.approx(0.5)


def test_recall_at_k_no_relevant_docs():
    assert recall_at_k(["a", "b"], set(), 2) == 0.0


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_not_found():
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_mean_reciprocal_rank():
    retrieved = [["a", "b"], ["x", "c"]]
    relevant = [{"a"}, {"c"}]
    # rr's are 1.0 and 0.5 -> mean 0.75
    assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(0.75)


def test_mean_reciprocal_rank_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        mean_reciprocal_rank([["a"]], [{"a"}, {"b"}])


def test_mean_reciprocal_rank_empty():
    assert mean_reciprocal_rank([], []) == 0.0


def test_dcg_at_k_matches_hand_computed_value():
    # relevant docs at rank 1 and 3: 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
    dcg = dcg_at_k(["a", "x", "b"], {"a", "b"}, 3)
    assert dcg == pytest.approx(1.0 + 1.0 / math.log2(4))


def test_ndcg_at_k_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, 3) == pytest.approx(1.0)


def test_ndcg_at_k_worst_ranking_is_less_than_perfect():
    perfect = ndcg_at_k(["a", "b", "x"], {"a", "b"}, 3)
    worst = ndcg_at_k(["x", "a", "b"], {"a", "b"}, 3)
    assert worst < perfect


def test_ndcg_at_k_no_relevant_docs_is_zero():
    assert ndcg_at_k(["a", "b"], set(), 2) == 0.0
