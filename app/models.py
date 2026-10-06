"""Shared lightweight domain models used across the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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

