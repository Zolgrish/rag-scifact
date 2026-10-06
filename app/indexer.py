"""Validated FAISS IndexFlatIP and self-contained index bundles (E3)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Mapping, Sequence, TYPE_CHECKING

from app.chunker import MODEL_ID
from app.embedder import MODEL_REVISION, OUTPUT_DIMENSION
from app.models import Chunk

if TYPE_CHECKING:
    import numpy as np


class IndexErrorBase(RuntimeError):
    """Base class for index infrastructure errors."""


class IndexBuildError(IndexErrorBase):
    """Invalid build inputs or failed persistence."""


class IndexLoadError(IndexErrorBase):
    """Missing, corrupt, or misaligned persisted bundle."""


class IndexCompatibilityError(IndexLoadError):
    """Bundle identity differs from the current runtime contract."""


@dataclass(frozen=True)
class IndexBundle:
    index: Any
    chunks: tuple[Chunk, ...]
    manifest: dict[str, Any]


def validate_vectors(vectors: np.ndarray, row_count: int) -> None:
    """Fail rather than normalize upstream contract violations."""
    import numpy as np

    if not isinstance(vectors, np.ndarray):
        raise IndexBuildError("Vectors must be a numpy.ndarray")
    if vectors.ndim != 2 or vectors.shape != (row_count, OUTPUT_DIMENSION):
        raise IndexBuildError(f"Vector shape must be ({row_count}, 384); got {vectors.shape}")
    if vectors.dtype != np.float32:
        raise IndexBuildError("Vectors must have dtype float32")
    if not np.isfinite(vectors).all():
        raise IndexBuildError("Vectors contain non-finite values")
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if np.any(norms == 0) or not np.allclose(norms, 1.0, atol=1e-6, rtol=1e-5):
        raise IndexBuildError("Vectors must have non-zero L2-normalized unit norms")


def _validate_chunks(chunks: Sequence[Chunk]) -> None:
    ids: set[str] = set()
    for chunk in chunks:
        if not isinstance(chunk, Chunk):
            raise IndexBuildError("Mapping must contain Chunk objects")
        for field in ("doc_id", "chunk_id", "title", "text", "embedding_title"):
            if not isinstance(getattr(chunk, field), str):
                raise IndexBuildError(f"Chunk {field} must be a string")
        if not chunk.doc_id.strip() or not chunk.text.strip():
            raise IndexBuildError("Chunk document ID and evidence must be non-empty")
        if chunk.chunk_id in ids:
            raise IndexBuildError(f"Duplicate chunk_id: {chunk.chunk_id}")
        ids.add(chunk.chunk_id)
        for field in ("token_start", "token_end", "char_start", "char_end", "body_token_count"):
            if type(getattr(chunk, field)) is not int:
                raise IndexBuildError(f"Chunk {field} must be an integer")
        if not (0 <= chunk.token_start < chunk.token_end):
            raise IndexBuildError("Invalid chunk token span")
        if not (0 <= chunk.char_start < chunk.char_end):
            raise IndexBuildError("Invalid chunk character span")
        if chunk.char_end - chunk.char_start != len(chunk.text):
            raise IndexBuildError("Chunk character span does not match exact evidence length")
        if chunk.body_token_count != chunk.token_end - chunk.token_start or not 1 <= chunk.body_token_count <= 220:
            raise IndexBuildError("Chunk body token count violates E2")
        if chunk.chunk_id != f"{chunk.doc_id}:{chunk.token_start}-{chunk.token_end}":
            raise IndexBuildError("Chunk ID does not match deterministic E2 span scheme")
        if type(chunk.title_truncated) is not bool:
            raise IndexBuildError("Chunk title_truncated must be a boolean")
        if chunk.title_truncated:
            if chunk.embedding_title == chunk.title or not chunk.title.startswith(chunk.embedding_title):
                raise IndexBuildError("Truncated embedding title must be a shorter source prefix")
        elif chunk.embedding_title != chunk.title:
            raise IndexBuildError("Untruncated embedding title must equal original title")


def _expect(section: Mapping[str, Any], expected: Mapping[str, Any], name: str) -> None:
    for key, value in expected.items():
        actual = section.get(key)
        if type(actual) is not type(value) or actual != value:
            raise IndexCompatibilityError(f"Incompatible {name}.{key}: expected {value!r}, got {actual!r}")


def validate_manifest(manifest: Mapping[str, Any], *, expected_corpus_sha256: str | None = None) -> None:
    """Compare recorded identities without constructing MiniLM or a tokenizer."""
    _expect(manifest, {"schema_version": 1, "index_type": "IndexFlatIP", "metric": "inner_product", "dimension": 384}, "index")
    embedding = manifest.get("embedding")
    chunking = manifest.get("chunking")
    corpus = manifest.get("corpus")
    if not all(isinstance(part, Mapping) for part in (embedding, chunking, corpus)):
        raise IndexLoadError("Manifest embedding/chunking/corpus sections are required")
    _expect(embedding, {
        "model_id": MODEL_ID, "revision": MODEL_REVISION, "output_dimension": 384,
        "dtype": "float32", "normalize": True, "normalized": True, "effective_model_input_limit": 256,
    }, "embedding")
    _expect(chunking, {
        "tokenizer": MODEL_ID, "chunk_size_tokens": 220, "overlap_tokens": 30,
        "effective_model_input_limit": 256, "chunk_id_scheme": "<doc_id>:<token_start>-<token_end>",
        "boundary_policy": "stable_wordpiece_word_boundaries",
        "overlap_policy": "exact_30_standalone_minilm_token_ids",
        "body_token_count_basis": "standalone_chunk_tokenization", "embedding_separator": "\n\n",
        "span_convention": "half_open_source_tokens_and_characters",
        "title_truncation_policy": "source_token_prefix_checked_against_actual_combined_input",
        "empty_body_policy": "zero_chunks",
    }, "chunking")
    digest = corpus.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise IndexLoadError("Manifest must contain a valid corpus SHA256")
    if expected_corpus_sha256 is not None and digest != expected_corpus_sha256:
        raise IndexCompatibilityError("Incompatible corpus SHA256 identity")
    for key in ("ntotal", "mapping_row_count"):
        if type(manifest.get(key)) is not int or manifest[key] <= 0:
            raise IndexLoadError(f"Manifest {key} must be a positive integer")
    if manifest["ntotal"] != manifest["mapping_row_count"]:
        raise IndexLoadError("Manifest ntotal and mapping row count differ")
    if type(corpus.get("documents")) is not int or corpus["documents"] <= 0:
        raise IndexLoadError("Manifest corpus documents must be a positive integer")
    if not isinstance(manifest.get("run_id"), str) or not manifest["run_id"]:
        raise IndexLoadError("Manifest run_id is required")
    try:
        built_at = datetime.fromisoformat(manifest["built_at"])
        if built_at.utcoffset() is None or built_at.utcoffset().total_seconds() != 0:
            raise ValueError("not UTC")
    except (KeyError, TypeError, ValueError) as exc:
        raise IndexLoadError("Manifest built_at must be a UTC timestamp") from exc


def build_index(
    vectors: np.ndarray, chunks: Sequence[Chunk], *, run_id: str,
    corpus: Mapping[str, Any], embedding: Mapping[str, Any], chunking: Mapping[str, Any],
) -> IndexBundle:
    """Build in the caller's deterministic chunk order; no evaluator inputs."""
    import faiss

    chunks = tuple(chunks)
    if not chunks:
        raise IndexBuildError("Cannot build an empty index")
    _validate_chunks(chunks)
    validate_vectors(vectors, len(chunks))
    if corpus.get("documents") != len({c.doc_id for c in chunks}):
        raise IndexBuildError("Corpus document count differs from chunk document count")
    manifest = {
        "schema_version": 1, "status": "built", "built_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id, "index_type": "IndexFlatIP", "metric": "inner_product",
        "dimension": OUTPUT_DIMENSION, "ntotal": len(chunks), "mapping_row_count": len(chunks),
        "corpus": deepcopy(dict(corpus)), "embedding": deepcopy(dict(embedding)),
        "chunking": deepcopy(dict(chunking)),
        "ordering": "corpus_jsonl_order_then_chunker_output_order",
        "versions": {"python": platform.python_version(), "faiss": faiss.__version__,
                     "numpy": version("numpy"), "sentence_transformers": version("sentence-transformers")},
    }
    try:
        validate_manifest(manifest)
        index = faiss.IndexFlatIP(OUTPUT_DIMENSION)
        index.add(vectors)
        if index.ntotal != len(chunks):
            raise IndexBuildError("FAISS vector count differs from chunk count")
    except IndexLoadError as exc:
        raise IndexBuildError(str(exc)) from exc
    except RuntimeError as exc:
        raise IndexBuildError(f"FAISS build failed: {exc}") from exc
    return IndexBundle(index, chunks, manifest)


def _validate_alignment(bundle: IndexBundle) -> None:
    import faiss

    validate_manifest(bundle.manifest)
    index = bundle.index
    if type(index) is not faiss.IndexFlatIP or index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise IndexCompatibilityError("FAISS index must be exactly IndexFlatIP with inner product")
    if index.d != OUTPUT_DIMENSION:
        raise IndexCompatibilityError(f"FAISS dimension must be 384; got {index.d}")
    if index.ntotal <= 0 or index.ntotal != bundle.manifest["ntotal"]:
        raise IndexLoadError("FAISS ntotal does not match non-empty manifest")
    if len(bundle.chunks) != index.ntotal:
        raise IndexLoadError("Mapping row count does not match FAISS ntotal")
    try:
        _validate_chunks(bundle.chunks)
        validate_vectors(index.reconstruct_n(0, index.ntotal), index.ntotal)
    except IndexBuildError as exc:
        raise IndexLoadError(f"Invalid persisted index/mapping: {exc}") from exc
    if len({c.doc_id for c in bundle.chunks}) != bundle.manifest["corpus"]["documents"]:
        raise IndexLoadError("Mapping document count differs from corpus identity")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def save_bundle(bundle: IndexBundle, directory: str | Path) -> dict[str, Any]:
    """Stage all files, replace data, publish the completion manifest last.

    A failed replacement cannot yield a usable partial bundle: readers check
    both data hashes against one manifest. Existing unrelated files are kept.
    """
    import faiss

    target = Path(directory)
    try:
        _validate_alignment(bundle)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".index-stage-", dir=target.parent) as tmp:
            staging = Path(tmp)
            faiss.write_index(bundle.index, str(staging / "index.faiss"))
            with (staging / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
                for position, chunk in enumerate(bundle.chunks):
                    row = {"position": position, **asdict(chunk)}
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            manifest = deepcopy(bundle.manifest)
            manifest["files"] = {
                name: {"sha256": hashlib.sha256((staging / name).read_bytes()).hexdigest(),
                       "size_bytes": (staging / name).stat().st_size}
                for name in ("index.faiss", "chunks.jsonl")
            }
            _write_json(staging / "index_manifest.json", manifest)
            target.mkdir(parents=True, exist_ok=True)
            for name in ("index.faiss", "chunks.jsonl", "index_manifest.json"):
                os.replace(staging / name, target / name)
        return manifest
    except (IndexErrorBase, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise IndexBuildError(f"Cannot persist index bundle {target}: {exc}") from exc


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _read_mapping(payload: bytes) -> tuple[Chunk, ...]:
    chunks: list[Chunk] = []
    for position, line in enumerate(payload.decode("utf-8").splitlines()):
        try:
            row = json.loads(line, parse_constant=_reject_json_constant)
            if not isinstance(row, dict):
                raise ValueError("Expected a metadata object")
            if type(row.get("position")) is not int or row["position"] != position:
                raise ValueError("Positions must be contiguous 0..N-1")
            row.pop("position")
            chunks.append(Chunk(**row))
        except (ValueError, TypeError) as exc:
            raise IndexLoadError(f"chunks.jsonl:{position + 1}: {exc}") from exc
    return tuple(chunks)


def load_bundle(
    directory: str | Path, *, expected_corpus_sha256: str | None = None,
) -> IndexBundle:
    """Load validated bytes and mapping; never load/chunk/embed the corpus."""
    import faiss
    import numpy as np

    directory = Path(directory)
    try:
        # Hash precisely the byte snapshots that are parsed, avoiding a
        # hash-then-reopen race with a concurrently published replacement.
        manifest_bytes = (directory / "index_manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"), parse_constant=_reject_json_constant)
        if not isinstance(manifest, dict):
            raise IndexLoadError("Index manifest must be a JSON object")
        validate_manifest(manifest, expected_corpus_sha256=expected_corpus_sha256)
        payloads: dict[str, bytes] = {}
        for name in ("index.faiss", "chunks.jsonl"):
            payload = (directory / name).read_bytes()
            recorded = manifest.get("files", {}).get(name, {})
            if hashlib.sha256(payload).hexdigest() != recorded.get("sha256"):
                raise IndexLoadError(f"{name} SHA256 mismatch")
            if type(recorded.get("size_bytes")) is not int or len(payload) != recorded["size_bytes"]:
                raise IndexLoadError(f"{name} size mismatch")
            payloads[name] = payload
        index = faiss.deserialize_index(np.frombuffer(payloads["index.faiss"], dtype=np.uint8))
        bundle = IndexBundle(index, _read_mapping(payloads["chunks.jsonl"]), manifest)
        _validate_alignment(bundle)
        if (directory / "index_manifest.json").read_bytes() != manifest_bytes:
            raise IndexLoadError("Bundle changed while loading; retry the load")
        return bundle
    except IndexErrorBase:
        raise
    except (OSError, UnicodeError, ValueError, TypeError, AttributeError, RuntimeError) as exc:
        raise IndexLoadError(f"Cannot load index bundle {directory}: {exc}") from exc

