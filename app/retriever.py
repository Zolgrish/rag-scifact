"""Exact dense document retrieval with adaptive chunk candidate expansion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.embedder import EmbeddingError
from app.indexer import (
    IndexBuildError, IndexBundle, IndexCompatibilityError, IndexLoadError,
    validate_vectors,
)
from app.models import RetrievedDocument

if TYPE_CHECKING:
    from app.embedder import MiniLMEmbedder


class QueryValidationError(ValueError):
    """Invalid user query, Top-K, or missing query ID."""


class RetrievalConfigurationError(RuntimeError):
    """Retrieval runtime configuration is unsupported or violates the benchmark contract."""


def validate_retrieval_settings(mode: str, max_top_k: int) -> None:
    if mode != "dense":
        raise RetrievalConfigurationError(
            f"E3 supports retrieval.mode='dense' only; got {mode!r}"
        )
    if type(max_top_k) is not int or not 1 <= max_top_k <= 10:
        raise RetrievalConfigurationError(
            f"retrieval.max_top_k must be an integer in 1..10; got {max_top_k!r}"
        )


def validate_query(query: str, top_k: int, *, max_top_k: int = 10) -> str:
    if not isinstance(query, str):
        raise QueryValidationError("query must be a string")
    trimmed = query.strip()
    if not 1 <= len(trimmed) <= 2000:
        raise QueryValidationError("Trimmed query must contain 1..2000 characters")
    if type(max_top_k) is not int or not 1 <= max_top_k <= 10:
        raise RetrievalConfigurationError(
            f"retrieval.max_top_k must be an integer in 1..10; got {max_top_k!r}"
        )
    if type(top_k) is not int or not 1 <= top_k <= max_top_k:
        raise QueryValidationError(
            f"top_k must be an integer in 1..{max_top_k} (bool is invalid)"
        )
    return trimmed


class DenseRetriever:
    def __init__(
        self,
        bundle: IndexBundle,
        embedder: MiniLMEmbedder,
        *,
        max_top_k: int = 10,
    ) -> None:
        import faiss

        validate_retrieval_settings("dense", max_top_k)
        if type(bundle.index) is not faiss.IndexFlatIP or bundle.index.d != 384:
            raise IndexCompatibilityError("Dense retrieval requires a 384-D IndexFlatIP")
        if bundle.index.ntotal <= 0 or bundle.index.ntotal != len(bundle.chunks):
            raise IndexLoadError("Retrieval index must be non-empty and aligned with mapping")
        runtime = embedder.runtime_metadata()
        for key in ("model_id", "revision", "output_dimension", "dtype", "normalized", "effective_model_input_limit"):
            recorded = bundle.manifest["embedding"].get(key)
            if type(runtime.get(key)) is not type(recorded) or runtime.get(key) != recorded:
                raise IndexCompatibilityError(f"Query embedding runtime differs from bundle: {key}")
        self.bundle = bundle
        self.embedder = embedder
        self.max_top_k = max_top_k

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        import numpy as np

        query = validate_query(query, top_k, max_top_k=self.max_top_k)
        vector = self.embedder.encode_query(query)
        if not isinstance(vector, np.ndarray) or vector.shape != (384,):
            raise EmbeddingError("Query vector must be a numpy.ndarray with shape (384,)")
        try:
            validate_vectors(vector.reshape(1, 384), 1)
        except IndexBuildError as exc:
            raise EmbeddingError(f"Invalid query vector: {exc}") from exc
        ntotal = self.bundle.index.ntotal
        candidate_k = min(ntotal, max(32, top_k * 4))
        while True:
            try:
                scores, positions = self.bundle.index.search(vector.reshape(1, 384), candidate_k)
            except RuntimeError as exc:
                raise IndexLoadError(f"FAISS search failed: {exc}") from exc
            candidates = [(float(score), int(position)) for score, position in zip(scores[0], positions[0])]
            if any(not np.isfinite(score) or not 0 <= pos < ntotal for score, pos in candidates):
                raise IndexLoadError("FAISS returned invalid scores/vector positions")
            candidates.sort(key=lambda item: (-item[0], item[1]))
            selected: list[tuple[float, int]] = []
            docs: set[str] = set()
            for score, position in candidates:
                doc_id = self.bundle.chunks[position].doc_id
                if doc_id not in docs:
                    docs.add(doc_id)
                    selected.append((score, position))
            enough = len(selected) >= top_k
            # Expand through cutoff ties too: unseen tied vectors can have a
            # smaller position. Candidate count must not change final semantics.
            ties_complete = enough and candidates[-1][0] < selected[top_k - 1][0]
            if candidate_k == ntotal or ties_complete:
                break
            candidate_k = min(ntotal, candidate_k * 2)
        results: list[RetrievedDocument] = []
        for rank, (score, position) in enumerate(selected[:top_k], start=1):
            chunk = self.bundle.chunks[position]
            results.append(RetrievedDocument(
                doc_id=chunk.doc_id, chunk_id=chunk.chunk_id, rank=rank,
                score=score, title=chunk.title, text=chunk.text,
            ))
        return results

