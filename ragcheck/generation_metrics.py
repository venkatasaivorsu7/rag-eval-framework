"""Embedding-based generation-quality metrics: answer relevancy, faithfulness,
and context precision.

These are deliberately simple, explainable approximations of the metrics
popularized by RAGAS and similar frameworks - built on cosine similarity
between embeddings rather than an LLM-as-judge call, so they run for free,
offline, and deterministically. Swap in SentenceTransformerEmbedder (or an
LLM-judge scorer you write yourself, matching the same function signatures)
when you want higher-fidelity scores and are willing to pay for them.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors. Returns 0.0 (rather than
    NaN) when either vector is all-zero, which happens for embeddings of
    empty/degenerate text."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def answer_relevancy(question_vec: np.ndarray, answer_vec: np.ndarray) -> float:
    """How on-topic the answer is relative to the question it's answering.
    A low score usually means the model answered a different question than
    the one asked (a common RAG failure mode when retrieval pulls in a
    plausible-but-wrong context)."""
    return cosine_similarity(question_vec, answer_vec)


def faithfulness(answer_vec: np.ndarray, context_vecs: Sequence[np.ndarray]) -> float:
    """How grounded the answer is in the retrieved context, approximated as
    the similarity to the single closest retrieved passage. Low faithfulness
    with high answer_relevancy is the signature of hallucination: the model
    answered the right question, just not from the retrieved evidence."""
    if len(context_vecs) == 0:
        return 0.0
    return max(cosine_similarity(answer_vec, c) for c in context_vecs)


def context_precision(
    ground_truth_vec: np.ndarray,
    context_vecs: Sequence[np.ndarray],
    threshold: float = 0.15,
) -> float:
    """Fraction of retrieved contexts that are actually relevant to the
    ground-truth answer (cosine similarity to the ground truth at or above
    `threshold`). Measures retrieval precision from the generation side:
    a low score means the retriever is burning context-window budget on
    passages that don't help answer the question."""
    if len(context_vecs) == 0:
        return 0.0
    relevant = sum(1 for c in context_vecs if cosine_similarity(ground_truth_vec, c) >= threshold)
    return relevant / len(context_vecs)
