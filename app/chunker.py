"""MiniLM body-token windows and exact source spans (E2)."""

from __future__ import annotations

import logging
from typing import Any, Iterator, Sequence, TYPE_CHECKING

from app.config import EmbeddingConfig
from app.loader import CorpusDocument
from app.models import Chunk

if TYPE_CHECKING:
    from app.embedder import MiniLMEmbedder

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
_LOGGER = logging.getLogger(__name__)


class ChunkingError(RuntimeError):
    """A tokenizer or chunking contract cannot be satisfied."""


def token_windows(
    token_count: int,
    *,
    chunk_size: int = 220,
    overlap: int = 30,
    stable_boundaries: Sequence[int] | None = None,
) -> Iterator[tuple[int, int]]:
    """Yield half-open windows with exact overlap on stable token boundaries."""
    if type(token_count) is not int or token_count < 0:
        raise ChunkingError("token_count must be a non-negative integer")
    if type(chunk_size) is not int or chunk_size != 220:
        raise ChunkingError("E2 chunk_size is locked at 220 tokens")
    if type(overlap) is not int or overlap != 30:
        raise ChunkingError("E2 overlap is locked at 30 tokens")

    if stable_boundaries is None:
        boundaries = list(range(token_count + 1))
        boundary_set = set(boundaries)
    else:
        boundaries = list(stable_boundaries)
        if any(type(value) is not int for value in boundaries):
            raise ChunkingError("stable token boundaries must be integers")
        if boundaries != sorted(set(boundaries)):
            raise ChunkingError("stable token boundaries must be sorted and unique")
        if not boundaries or boundaries[0] != 0 or boundaries[-1] != token_count:
            raise ChunkingError("stable token boundaries must include 0 and token_count")
        boundary_set = set(boundaries)

    start = 0
    while start < token_count:
        if start not in boundary_set:
            raise ChunkingError("chunk start is not a stable token boundary")
        if token_count - start <= chunk_size:
            end = token_count
        else:
            candidates = [
                end
                for end in boundaries
                if start + overlap < end <= start + chunk_size
                and end - overlap in boundary_set
            ]
            if not candidates:
                raise ChunkingError(
                    "Cannot satisfy 220-token body and exact 30-token overlap "
                    "on stable WordPiece boundaries"
                )
            end = max(candidates)
        yield start, end
        if end == token_count:
            break
        start = end - overlap


def source_span(
    offsets: Sequence[tuple[int, int]], start: int, end: int
) -> tuple[int, int]:
    """Map a non-empty token window to exact source character boundaries."""
    if not 0 <= start < end <= len(offsets):
        raise ChunkingError("Invalid token window for source offsets")
    return offsets[start][0], offsets[end - 1][1]


def validate_e2_config(config: EmbeddingConfig) -> None:
    """Reject benchmark overrides at the E2 runtime boundary."""
    expected = {
        "model_id": MODEL_ID, "chunk_size_tokens": 220,
        "chunk_overlap_tokens": 30, "normalize": True, "dtype": "float32",
    }
    for key, value in expected.items():
        actual = getattr(config, key)
        if actual != value or type(actual) is not type(value):
            raise ChunkingError(f"E2 embedding.{key} is locked at {value!r}; got {actual!r}")


def default_e2_config() -> EmbeddingConfig:
    return EmbeddingConfig(MODEL_ID, 220, 30, True, "float32")


class MiniLMChunker:
    """Use the runtime's fast tokenizer, without decoding evidence tokens.

    Construct via ``from_embedder`` in production. Direct tokenizer injection
    makes offset/budget failure paths independently testable without downloads.
    """

    def __init__(
        self, tokenizer: Any, effective_input_limit: int,
        *, config: EmbeddingConfig | None = None,
    ) -> None:
        self.config = config if config is not None else default_e2_config()
        validate_e2_config(self.config)
        if not getattr(tokenizer, "is_fast", False):
            raise ChunkingError("MiniLM chunking requires a fast tokenizer with reliable offsets")
        if type(effective_input_limit) is not int or effective_input_limit <= 0:
            raise ChunkingError("Effective MiniLM input limit must be a positive integer")
        self.tokenizer = tokenizer
        self.effective_input_limit = effective_input_limit

    @classmethod
    def from_embedder(cls, embedder: MiniLMEmbedder) -> MiniLMChunker:
        return cls(
            embedder.tokenizer, embedder.effective_input_limit,
            config=embedder.config,
        )

    def count_tokens(self, text: str, *, add_special_tokens: bool = False) -> int:
        return len(self.tokenizer(
            text, add_special_tokens=add_special_tokens, truncation=False,
        )["input_ids"])

    def _encoding(
        self, text: str
    ) -> tuple[list[int], list[tuple[int, int]], list[int | None]]:
        try:
            encoded = self.tokenizer(
                text, add_special_tokens=False, return_offsets_mapping=True,
                truncation=False,
            )
            input_ids = list(encoded["input_ids"])
            offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
            try:
                word_ids = list(encoded.word_ids())
            except (AttributeError, TypeError, ValueError, NotImplementedError) as exc:
                raise ChunkingError(
                    "Tokenizer cannot provide reliable WordPiece word identities"
                ) from exc
            if len(offsets) != len(input_ids) or len(word_ids) != len(input_ids):
                raise ChunkingError("Tokenizer token/offset lengths differ")
            previous_start = previous_end = 0
            for start, end in offsets:
                if type(start) is not int or type(end) is not int:
                    raise ChunkingError("Tokenizer source offsets must be integers")
                if not (0 <= start < end <= len(text)):
                    raise ChunkingError("Tokenizer returned invalid source offsets")
                if start < previous_start or end < previous_end:
                    raise ChunkingError("Tokenizer source offsets must be monotonic")
                previous_start, previous_end = start, end
            return input_ids, offsets, word_ids
        except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise ChunkingError("Tokenizer cannot provide reliable source offsets") from exc

    def _offsets(self, text: str) -> list[tuple[int, int]]:
        return self._encoding(text)[1]

    @staticmethod
    def _stable_boundaries(word_ids: Sequence[int | None]) -> list[int]:
        """Return token boundaries that do not split a tokenizer word."""
        token_count = len(word_ids)
        if token_count == 0:
            return [0]
        boundaries = [0]
        for index in range(1, token_count):
            if word_ids[index] != word_ids[index - 1]:
                boundaries.append(index)
        boundaries.append(token_count)
        return boundaries

    def _embedding_title(
        self, document: CorpusDocument, body: str, body_count: int,
        title_offsets: Sequence[tuple[int, int]],
    ) -> tuple[str, bool]:
        title = document.title
        combined = title + "\n\n" + body if title else body
        if self.count_tokens(combined, add_special_tokens=True) <= self.effective_input_limit:
            return title, False
        body_input_count = self.count_tokens(body, add_special_tokens=True)
        if body_input_count > self.effective_input_limit:
            raise ChunkingError(f"Body evidence cannot fit MiniLM input limit for doc_id={document.doc_id}")
        # An arithmetic cap only narrows the search. Every candidate is measured
        # as the actual combined string, including separator and special tokens.
        cap = min(len(title_offsets), self.effective_input_limit - body_input_count)
        for retained in range(cap, -1, -1):
            candidate = title[:title_offsets[retained - 1][1]] if retained else ""
            combined = candidate + "\n\n" + body if candidate else body
            if self.count_tokens(combined, add_special_tokens=True) <= self.effective_input_limit:
                _LOGGER.info(
                    "title_truncated doc_id=%s original_title_tokens=%d "
                    "retained_title_tokens=%d model_input_limit=%d body_token_count=%d",
                    document.doc_id, len(title_offsets), self.count_tokens(candidate),
                    self.effective_input_limit, body_count,
                )
                return candidate, True
        raise ChunkingError("Cannot fit embedding input after title truncation")

    def chunk_document(self, document: CorpusDocument) -> list[Chunk]:
        if not document.text.strip():
            return []
        input_ids, offsets, word_ids = self._encoding(document.text)
        if not offsets:
            raise ChunkingError(f"Non-empty body has no MiniLM tokens for doc_id={document.doc_id}")
        title_offsets = self._offsets(document.title)
        stable_boundaries = self._stable_boundaries(word_ids)
        chunks: list[Chunk] = []
        for start, end in token_windows(
            len(offsets), stable_boundaries=stable_boundaries
        ):
            char_start, char_end = source_span(offsets, start, end)
            body = document.text[char_start:char_end]
            standalone_ids = list(self.tokenizer(
                body, add_special_tokens=False, truncation=False,
            )["input_ids"])
            expected_ids = input_ids[start:end]
            if standalone_ids != expected_ids:
                raise ChunkingError(
                    f"Exact source span retokenizes differently for doc_id={document.doc_id} "
                    f"token_span={start}:{end}"
                )
            if len(standalone_ids) > self.config.chunk_size_tokens:
                raise ChunkingError(
                    f"Standalone body exceeds 220 MiniLM tokens for doc_id={document.doc_id}"
                )
            embedding_title, truncated = self._embedding_title(
                document, body, len(standalone_ids), title_offsets,
            )
            chunk = Chunk(
                doc_id=document.doc_id, chunk_id=f"{document.doc_id}:{start}-{end}",
                title=document.title, text=body, token_start=start, token_end=end,
                char_start=char_start, char_end=char_end,
                body_token_count=len(standalone_ids),
                embedding_title=embedding_title, title_truncated=truncated,
            )
            if self.count_tokens(chunk.embedding_text, add_special_tokens=True) > self.effective_input_limit:
                raise ChunkingError("Final embedding input exceeds MiniLM limit")
            chunks.append(chunk)
        return chunks

    def runtime_metadata(self) -> dict[str, object]:
        """Actual E2 metadata for E3; this does not update a manifest/index."""
        return {
            "tokenizer": self.config.model_id, "chunk_size_tokens": 220,
            "overlap_tokens": 30, "effective_model_input_limit": self.effective_input_limit,
            "chunk_id_scheme": "<doc_id>:<token_start>-<token_end>",
            "span_convention": "half_open_source_tokens_and_characters",
            "body_token_count_basis": "standalone_chunk_tokenization",
            "boundary_policy": "stable_wordpiece_word_boundaries",
            "overlap_policy": "exact_30_standalone_minilm_token_ids",
            "embedding_separator": "\n\n",
            "title_truncation_policy": "source_token_prefix_checked_against_actual_combined_input",
            "empty_body_policy": "zero_chunks",
        }

