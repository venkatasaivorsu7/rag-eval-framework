"""ragcheck: a small, dependency-light framework for evaluating RAG pipelines.

Public API:
    RagEvaluator - runs retrieval + generation metrics over a dataset
    EvalResult   - per-query and aggregate results returned by RagEvaluator
    Embedder     - interface for pluggable embedding backends
    TfidfEmbedder - default, offline, no-download embedding backend
"""

from .evaluator import RagEvaluator, EvalResult
from .embeddings import Embedder, TfidfEmbedder

__all__ = ["RagEvaluator", "EvalResult", "Embedder", "TfidfEmbedder"]
__version__ = "0.1.0"
