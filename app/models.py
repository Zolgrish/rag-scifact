"""Shared lightweight domain models used across the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class Chunk:
    """Exact body evidence with half-open source token/character spans."""

    doc_id: str
    chunk_id: str
    title: str
    text: str
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    body_token_count: int
    embedding_title: str
    title_truncated: bool

    @property
    def embedding_text(self) -> str:
        return (
            self.embedding_title + "\n\n" + self.text
            if self.embedding_title else self.text
        )


class SemanticStatus(str, Enum):
    ANSWERED = "ANSWERED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


@dataclass(frozen=True)
class Citation:
    doc_id: str
    chunk_id: str
    quote: str


@dataclass(frozen=True)
class RetrievedDocument:
    doc_id: str
    chunk_id: str
    rank: int
    score: float
    title: str = ""
    text: str = ""


@dataclass
class RAGResponse:
    request_id: str
    answer: str
    status: SemanticStatus
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[RetrievedDocument] = field(default_factory=list)
    timing_ms: dict[str, float] = field(default_factory=dict)
    model_id: str = ""

