"""Lazy SentenceTransformer runtime with the locked MiniLM vector contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, TYPE_CHECKING

from app.chunker import ChunkingError, default_e2_config, validate_e2_config
from app.config import EmbeddingConfig
from app.models import Chunk

if TYPE_CHECKING:
    import numpy as np

OUTPUT_DIMENSION = 384
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class EmbeddingError(RuntimeError):
    """The embedding runtime or output violates the E2 contract."""


def _effective_input_limit(model: Any) -> int:
    """Respect the SentenceTransformer limit and any smaller tokenizer limit."""
    model_limit = getattr(model, "max_seq_length", None)
    if type(model_limit) is not int or model_limit <= 0:
        raise EmbeddingError("SentenceTransformer must expose a positive max_seq_length")
    limits = [model_limit]
    tokenizer_limit = getattr(model.tokenizer, "model_max_length", None)
    # HF uses a huge integer sentinel when a tokenizer has no known limit.
    if type(tokenizer_limit) is int and 0 < tokenizer_limit < 1_000_000:
        limits.append(tokenizer_limit)
    return min(limits)


class MiniLMEmbedder:
    """One shared corpus/query model; injected backends are for offline tests."""

    def __init__(
        self, config: EmbeddingConfig | None = None, *, device: str | None = None,
        local_files_only: bool = False, cache_folder: str | Path | None = None,
        backend: Any | None = None,
    ) -> None:
        self.config = config if config is not None else default_e2_config()
        try:
            validate_e2_config(self.config)
        except ChunkingError as exc:
            raise EmbeddingError(str(exc)) from exc
        if backend is None:
            try:
                from sentence_transformers import SentenceTransformer

                backend = SentenceTransformer(
                    self.config.model_id, device=device,
                    local_files_only=local_files_only,
                    cache_folder=str(cache_folder) if cache_folder is not None else None,
                    revision=MODEL_REVISION,
                )
            except Exception as exc:
                raise EmbeddingError(f"Cannot load {self.config.model_id}: {exc}") from exc
        self._model = backend
        self.tokenizer = backend.tokenizer
        self.effective_input_limit = _effective_input_limit(backend)
        dimension = backend.get_sentence_embedding_dimension()
        if dimension != OUTPUT_DIMENSION:
            raise EmbeddingError(f"MiniLM dimension must be 384; runtime reports {dimension}")

    def encode_texts(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        """Encode without hidden input truncation; enforce float32 unit vectors."""
        import numpy as np

        if isinstance(texts, (str, bytes)):
            raise EmbeddingError("encode_texts requires a sequence of strings")
        if type(batch_size) is not int or batch_size <= 0:
            raise EmbeddingError("batch_size must be a positive integer")
        items = list(texts)
        for text in items:
            if not isinstance(text, str) or not text.strip():
                raise EmbeddingError("Embedding inputs must be non-empty strings")
            count = len(self.tokenizer(
                text, add_special_tokens=True, truncation=False,
            )["input_ids"])
            if count > self.effective_input_limit:
                raise EmbeddingError(
                    f"Embedding input has {count} tokens; limit is {self.effective_input_limit}"
                )
        if not items:
            return np.empty((0, OUTPUT_DIMENSION), dtype=np.float32)
        try:
            raw = self._model.encode(
                items, batch_size=batch_size, convert_to_numpy=True,
                normalize_embeddings=True, precision="float32",
                show_progress_bar=False, prompt="",
            )
            vectors = np.array(raw, dtype=np.float32, copy=True, order="C")
        except Exception as exc:
            raise EmbeddingError(f"MiniLM encoding failed: {exc}") from exc
        expected_shape = (len(items), OUTPUT_DIMENSION)
        if vectors.shape != expected_shape:
            raise EmbeddingError(f"Embedding shape must be {expected_shape}; got {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise EmbeddingError("Embedding contains non-finite values")
        norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
        if not np.isfinite(norms).all() or np.any(norms == 0):
            raise EmbeddingError("Embedding has zero or invalid norm")
        np.divide(vectors, norms[:, None], out=vectors)
        if not np.isfinite(vectors).all() or not np.allclose(
            np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5, atol=1e-6,
        ):
            raise EmbeddingError("Cannot satisfy L2-normalized embedding contract")
        return vectors

    def encode_chunks(self, chunks: Sequence[Chunk], *, batch_size: int = 32) -> np.ndarray:
        return self.encode_texts([chunk.embedding_text for chunk in chunks], batch_size=batch_size)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]

    def runtime_metadata(self) -> dict[str, object]:
        """E3 can merge these observed values under its manifest guards."""
        return {
            "model_id": self.config.model_id, "output_dimension": OUTPUT_DIMENSION,
            "dtype": "float32", "normalize": True, "normalized": True,
            "runtime": "SentenceTransformer", "device": str(self._model.device),
            "revision": MODEL_REVISION,
            "effective_model_input_limit": self.effective_input_limit,
        }

