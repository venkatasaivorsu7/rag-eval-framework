# ragcheck — RAG Evaluation & Validation Framework

[![CI](https://github.com/venkatasaivorsu7/rag-eval-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/venkatasaivorsu7/rag-eval-framework/actions/workflows/ci.yml)

A small, dependency-light Python framework for evaluating retrieval-augmented
generation (RAG) pipelines — both the retrieval side (did we fetch the right
chunks?) and the generation side (did the model actually use them, and stay
on topic?).

Built to run entirely offline: the default embedding backend is a corpus-fit
TF-IDF vectorizer (via scikit-learn), so `ragcheck evaluate` needs no API key,
no model download, and no network call. Swap in real sentence embeddings
(`SentenceTransformerEmbedder`) or an LLM-as-judge scorer when you want higher
fidelity and are willing to pay for it — the metric functions don't care where
the vectors came from.

## Why

Most public "RAG eval" tooling either requires an LLM API key to run a single
test, or ships as a hosted product. I wanted something I could point at a
JSON file and run in CI on every commit, with metrics I could explain line by
line — so this is that, built from the retrieval-metrics and embedding-based
generation-metrics literature (precision/recall/MRR/NDCG on the retrieval
side; cosine-similarity approximations of answer relevancy, faithfulness, and
context precision on the generation side, in the spirit of RAGAS).

## Install

```bash
pip install -e ".[dev,report]"
```

## Quickstart

```bash
ragcheck evaluate examples/sample_dataset.json --report report.md --chart chart.png
```

```
Evaluated 5 queries.

Aggregate metrics:
  answer_relevancy     0.097
  context_precision    0.567
  faithfulness         0.393
  ndcg_at_k            1.000
  precision_at_k       0.567
  recall_at_k          1.000
  reciprocal_rank      1.000
```

(This is a real run against `examples/sample_dataset.json` — exact numbers
depend on the TF-IDF vocabulary built from whatever dataset you point it at.
The low `answer_relevancy` here is a real artifact of TF-IDF: short answers
and questions share few exact tokens even when they're clearly related, which
is precisely the kind of thing swapping in `SentenceTransformerEmbedder`
fixes.)

Or use it as a library:

```python
from ragcheck import RagEvaluator

dataset = [
    {
        "question": "How do I reset MFA for a user?",
        "retrieved_doc_ids": ["kb-104", "kb-12"],
        "relevant_doc_ids": ["kb-104"],
        "retrieved_contexts": ["Reset MFA from the user's authentication methods page in Entra ID."],
        "answer": "Go to the user's authentication methods in Entra ID and reset MFA.",
        "ground_truth": "In Entra ID, open Authentication methods for the user and reset MFA.",
    },
]

result = RagEvaluator(k=3).evaluate(dataset)
print(result.aggregate)
print(result.per_query[0])
```

## Metrics

### Retrieval (`ragcheck/retrieval_metrics.py`) — rank-based, binary relevance

| Metric | What it answers |
|---|---|
| `precision_at_k` | Of the top-k retrieved chunks, how many were actually relevant? |
| `recall_at_k` | Of all relevant chunks, how many did the top-k retrieval surface? |
| `reciprocal_rank` / `mean_reciprocal_rank` | How far down the list was the first relevant result? |
| `ndcg_at_k` | Rank-aware quality score, normalized against the best-possible ordering for this query |

### Generation (`ragcheck/generation_metrics.py`) — embedding cosine-similarity based

| Metric | What it answers |
|---|---|
| `answer_relevancy` | Is the answer actually on-topic for the question asked? |
| `faithfulness` | Is the answer grounded in the retrieved context, or hallucinated? |
| `context_precision` | Of the retrieved chunks, how many were relevant to the *reference* answer? |

Every metric is a small, pure function — read the two files above end to end
in about five minutes if you want to know exactly what a number means before
you trust it.

## Dataset format

A JSON list of records; every key except `question` is optional; the
evaluator only computes the metrics whose required fields are present:

```json
{
  "question": "...",
  "retrieved_doc_ids": ["doc-1", "doc-2"],
  "relevant_doc_ids": ["doc-1"],
  "retrieved_contexts": ["...passage text...", "...passage text..."],
  "answer": "...model's generated answer...",
  "ground_truth": "...reference answer..."
}
```

See [`examples/sample_dataset.json`](./examples/sample_dataset.json) for a
full worked example, built around a small internal-docs Q&A scenario.

## Using a real embedding model instead of TF-IDF

```python
from ragcheck import RagEvaluator
from ragcheck.embeddings import SentenceTransformerEmbedder

evaluator = RagEvaluator(embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"))
result = evaluator.evaluate(dataset)
```

Requires `pip install sentence-transformers` (kept as an optional dependency
so the core package and CI stay fast and offline).

## Development

```bash
pip install -e ".[dev,report]"
pytest --cov=ragcheck --cov-report=term-missing
```

38 tests covering every metric's edge cases (empty inputs, zero vectors, k
larger than the retrieved list, missing optional dataset fields) plus the
CLI's error paths — 87% line coverage. CI runs the full suite on Python 3.9,
3.11, and 3.12 on every push — see
[`.github/workflows/ci.yml`](./.github/workflows/ci.yml).

## License

MIT — see [LICENSE](./LICENSE).
