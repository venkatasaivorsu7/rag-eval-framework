"""Orchestrates a full evaluation run: fits the embedder over the dataset's
corpus once, scores every query on the retrieval + generation metrics that
apply to it, and aggregates the results.

Dataset format: a list of dicts, one per query. All keys are optional except
`question` - metrics that need a key you didn't provide are simply skipped
for that query (e.g. no `ground_truth` means no context_precision).

    {
        "question": "How do I reset a user's MFA in Entra ID?",
        "retrieved_doc_ids": ["kb-104", "kb-12", "kb-88"],
        "relevant_doc_ids": ["kb-104", "kb-77"],
        "retrieved_contexts": ["...passage text...", "...passage text..."],
        "answer": "...the model's generated answer...",
        "ground_truth": "...the reference answer..."
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import generation_metrics as gm
from . import retrieval_metrics as rm
from .embeddings import Embedder, TfidfEmbedder


@dataclass
class EvalResult:
    per_query: List[Dict[str, Any]] = field(default_factory=list)
    aggregate: Dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.per_query)


class RagEvaluator:
    """Evaluates a set of RAG (question, retrieval, answer) records.

    Args:
        embedder: embedding backend to use for the generation metrics.
            Defaults to TfidfEmbedder (offline, no model download).
        k: cutoff used for precision@k / recall@k / ndcg@k.
        context_precision_threshold: cosine-similarity threshold passed to
            generation_metrics.context_precision.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        k: int = 5,
        context_precision_threshold: float = 0.15,
    ):
        if k <= 0:
            raise ValueError("k must be a positive integer")
        self.embedder = embedder or TfidfEmbedder()
        self.k = k
        self.context_precision_threshold = context_precision_threshold

    def evaluate(self, dataset: List[Dict[str, Any]]) -> EvalResult:
        if not dataset:
            raise ValueError("dataset is empty - nothing to evaluate")

        self._fit_embedder(dataset)

        per_query = [self._evaluate_one(item) for item in dataset]
        aggregate = self._aggregate(per_query)
        return EvalResult(per_query=per_query, aggregate=aggregate)

    # -- internals ---------------------------------------------------------

    def _fit_embedder(self, dataset: List[Dict[str, Any]]) -> None:
        corpus: List[str] = []
        for item in dataset:
            corpus.append(item.get("question") or "")
            corpus.append(item.get("answer") or "")
            corpus.append(item.get("ground_truth") or "")
            corpus.extend(item.get("retrieved_contexts") or [])

        corpus = [t for t in corpus if t]
        if corpus:
            self.embedder.fit(corpus)

    def _evaluate_one(self, item: Dict[str, Any]) -> Dict[str, Any]:
        row: Dict[str, Any] = {"question": item.get("question", "")}

        retrieved_ids = item.get("retrieved_doc_ids")
        relevant_ids = item.get("relevant_doc_ids")
        if retrieved_ids is not None and relevant_ids is not None:
            row["precision_at_k"] = rm.precision_at_k(retrieved_ids, relevant_ids, self.k)
            row["recall_at_k"] = rm.recall_at_k(retrieved_ids, relevant_ids, self.k)
            row["reciprocal_rank"] = rm.reciprocal_rank(retrieved_ids, relevant_ids)
            row["ndcg_at_k"] = rm.ndcg_at_k(retrieved_ids, relevant_ids, self.k)

        question = item.get("question") or ""
        answer = item.get("answer") or ""
        ground_truth = item.get("ground_truth") or ""
        contexts = item.get("retrieved_contexts") or []

        if answer and question:
            q_vec = self.embedder.embed([question])[0]
            a_vec = self.embedder.embed([answer])[0]
            row["answer_relevancy"] = gm.answer_relevancy(q_vec, a_vec)

            if contexts:
                c_vecs = self.embedder.embed(contexts)
                row["faithfulness"] = gm.faithfulness(a_vec, c_vecs)
            else:
                row["faithfulness"] = 0.0

        if ground_truth and contexts:
            gt_vec = self.embedder.embed([ground_truth])[0]
            c_vecs = self.embedder.embed(contexts)
            row["context_precision"] = gm.context_precision(
                gt_vec, c_vecs, threshold=self.context_precision_threshold
            )

        return row

    def _aggregate(self, per_query: List[Dict[str, Any]]) -> Dict[str, float]:
        numeric_keys = set()
        for row in per_query:
            numeric_keys.update(k for k, v in row.items() if isinstance(v, (int, float)))

        aggregate = {}
        for key in numeric_keys:
            values = [row[key] for row in per_query if key in row]
            aggregate[key] = sum(values) / len(values) if values else 0.0
        return aggregate
