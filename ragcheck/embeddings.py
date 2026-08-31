"""Pluggable text embedding backends.

The evaluator only needs *relative* similarity between texts in the same
evaluation run, not a general-purpose embedding space - so the default
backend is a corpus-fit TF-IDF vectorizer: no model download, no network
call, deterministic, and fast enough to run in CI on every push.

Swap in SentenceTransformerEmbedder (or write your own Embedder subclass,
e.g. wrapping an OpenAI/Cohere embeddings API) when you want real semantic
similarity instead of lexical overlap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class Embedder(ABC):
    """Interface every embedding backend implements."""

    def fit(self, corpus: Sequence[str]) -> None:
        """Optional corpus-level fitting step. Default: no-op."""
        return None

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n_texts, dim) array of embeddings for `texts`."""
        raise NotImplementedError


class TfidfEmbedder(Embedder):
    """Default embedder: a scikit-learn TF-IDF vectorizer fit once over the
    full evaluation corpus (all questions/answers/contexts/ground-truths in
    the dataset), then reused to transform any subset of that text.

    Fitting over the whole corpus - rather than per call - is what makes the
    resulting cosine similarities meaningful: IDF weights need a real corpus
    to be anything other than noise.
    """

    def __init__(self, **vectorizer_kwargs):
        self._vectorizer_kwargs = vectorizer_kwargs
        self._vectorizer = None

    def fit(self, corpus: Sequence[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        corpus = [t for t in corpus if t]
        if not corpus:
            raise ValueError("TfidfEmbedder.fit() received an empty corpus")

        self._vectorizer = TfidfVectorizer(**self._vectorizer_kwargs).fit(corpus)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._vectorizer is None:
            # No explicit fit() call - fall back to fitting on this batch so
            # the embedder is still usable standalone (e.g. in a notebook).
            self.fit(texts)

        texts = list(texts)
        if not texts:
            return np.zeros((0, len(self._vectorizer.vocabulary_)))

        # Empty strings would otherwise raise inside the vectorizer's
        # tokenizer; map them to a zero vector explicitly instead.
        safe_texts = [t if t else " " for t in texts]
        return self._vectorizer.transform(safe_texts).toarray()


class SentenceTransformerEmbedder(Embedder):
    """Optional real-semantic-embedding backend using sentence-transformers.

    Not a default dependency of ragcheck (kept out of the core install so CI
    and quick local runs never need to download model weights) - install
    with `pip install ragcheck[sbert]` or `pip install sentence-transformers`
    yourself, then pass an instance of this class to RagEvaluator.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
            raise ImportError(
                "SentenceTransformerEmbedder requires the 'sentence-transformers' "
                "package. Install it with: pip install sentence-transformers"
            ) from exc

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = [t if t else " " for t in texts]
        return np.asarray(self._model.encode(list(texts)))
