"""Rank-based retrieval metrics: precision@k, recall@k, MRR, NDCG@k.

All functions take plain Python lists/sets of document identifiers (strings
or ints - anything hashable) so they have zero dependency on how you produced
the ranking. Relevance here is binary (a doc id is either in the relevant
set or it isn't); that covers the overwhelming majority of RAG eval datasets,
which are built from human-labeled or LLM-labeled relevant-chunk sets.
"""

from __future__ import annotations

import math
from typing import Hashable, Iterable, Sequence


def precision_at_k(retrieved: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    """Fraction of the top-k retrieved docs that are relevant."""
    if k <= 0:
        return 0.0
    relevant = set(relevant)
    top_k = list(retrieved)[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    """Fraction of all relevant docs that appear in the top-k retrieved."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    top_k = list(retrieved)[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: Sequence[Hashable], relevant: Iterable[Hashable]) -> float:
    """1 / (rank of the first relevant doc), or 0.0 if none appear at all."""
    relevant = set(relevant)
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(
    all_retrieved: Sequence[Sequence[Hashable]],
    all_relevant: Sequence[Iterable[Hashable]],
) -> float:
    """MRR across multiple queries. Both sequences must be the same length,
    one entry per query."""
    if len(all_retrieved) != len(all_relevant):
        raise ValueError("all_retrieved and all_relevant must be the same length")
    if not all_retrieved:
        return 0.0
    rrs = [reciprocal_rank(r, rel) for r, rel in zip(all_retrieved, all_relevant)]
    return sum(rrs) / len(rrs)


def dcg_at_k(retrieved: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    """Discounted cumulative gain over the top-k, with binary relevance."""
    relevant = set(relevant)
    top_k = list(retrieved)[:k]
    return sum(
        (1.0 if doc in relevant else 0.0) / math.log2(rank + 1)
        for rank, doc in enumerate(top_k, start=1)
    )


def ndcg_at_k(retrieved: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    """Normalized DCG@k: DCG@k divided by the best-possible DCG@k for this
    relevant set (i.e. all relevant docs ranked first). Returns 0.0 when
    there are no relevant docs to rank (IDCG would be 0)."""
    relevant = set(relevant)
    dcg = dcg_at_k(retrieved, relevant, k)

    ideal_ranking = list(relevant)[:k]
    idcg = dcg_at_k(ideal_ranking, relevant, k)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg
