import pytest

from ragcheck import RagEvaluator


@pytest.fixture
def toy_dataset():
    return [
        {
            "question": "How do I reset MFA for a user in Entra ID?",
            "retrieved_doc_ids": ["kb-104", "kb-12", "kb-88"],
            "relevant_doc_ids": ["kb-104"],
            "retrieved_contexts": [
                "To reset MFA, go to the user's authentication methods in Entra ID and select Require re-register MFA.",
                "SharePoint site permissions can be audited from the admin center.",
            ],
            "answer": "Go to the user's authentication methods in Entra ID and require re-registration of MFA.",
            "ground_truth": "In Entra ID, open the user's Authentication methods and require MFA re-registration.",
        },
        {
            "question": "What does the disaster recovery runbook cover?",
            "retrieved_doc_ids": ["doc-dr-1", "doc-net-2"],
            "relevant_doc_ids": ["doc-dr-1", "doc-dr-3"],
            "retrieved_contexts": [
                "The disaster recovery runbook covers Entra ID, Exchange Online, SharePoint, Intune, and core Azure infrastructure recovery.",
            ],
            "answer": "It covers recovery steps for identity, mail, SharePoint, Intune, and Azure infrastructure.",
            "ground_truth": "It covers recovery procedures for Entra ID, Exchange Online, SharePoint, Intune, and Azure.",
        },
    ]


def test_evaluate_returns_one_row_per_query(toy_dataset):
    result = RagEvaluator(k=3).evaluate(toy_dataset)
    assert len(result.per_query) == 2
    assert len(result) == 2


def test_evaluate_produces_expected_metric_keys(toy_dataset):
    result = RagEvaluator(k=3).evaluate(toy_dataset)
    expected = {
        "precision_at_k",
        "recall_at_k",
        "reciprocal_rank",
        "ndcg_at_k",
        "answer_relevancy",
        "faithfulness",
        "context_precision",
    }
    assert expected.issubset(result.per_query[0].keys())
    assert expected.issubset(result.aggregate.keys())


def test_evaluate_first_query_perfect_retrieval_rank(toy_dataset):
    # kb-104 is the only relevant doc and it's retrieved first -> RR == 1.0
    result = RagEvaluator(k=3).evaluate(toy_dataset)
    assert result.per_query[0]["reciprocal_rank"] == pytest.approx(1.0)


def test_evaluate_scores_are_bounded(toy_dataset):
    result = RagEvaluator(k=3).evaluate(toy_dataset)
    for row in result.per_query:
        for key, value in row.items():
            if isinstance(value, float):
                assert -1e-9 <= value <= 1.0 + 1e-9, f"{key}={value} out of bounds"


def test_evaluate_empty_dataset_raises():
    with pytest.raises(ValueError):
        RagEvaluator().evaluate([])


def test_evaluate_invalid_k_raises():
    with pytest.raises(ValueError):
        RagEvaluator(k=0)


def test_evaluate_handles_missing_optional_fields():
    # A minimal record with only retrieval info, no answer/ground_truth.
    dataset = [
        {
            "question": "minimal record",
            "retrieved_doc_ids": ["a", "b"],
            "relevant_doc_ids": ["a"],
        }
    ]
    result = RagEvaluator(k=2).evaluate(dataset)
    row = result.per_query[0]
    assert "precision_at_k" in row
    assert "answer_relevancy" not in row
