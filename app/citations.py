"""Strict model JSON parsing and grounding against included context only."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re

from app.context import ContextBundle
from app.models import Citation, SemanticStatus


class RAGOutputValidationError(RuntimeError):
    """Model output violates schema, citation authority or structural status rules."""


@dataclass(frozen=True)
class ParsedRAGOutput:
    status: SemanticStatus
    answer: str
    citations: tuple[Citation, ...]


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RAGOutputValidationError("Duplicate JSON object key")
        result[key] = value
    return result


def _constant(value: str) -> object:
    raise RAGOutputValidationError("Non-standard JSON constant")


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RAGOutputValidationError(f"{field} must be a non-empty string")
    return value


def parse_output(text: str) -> ParsedRAGOutput:
    try:
        value = json.loads(text, object_pairs_hook=_object, parse_constant=_constant)
    except (ValueError, TypeError, RecursionError) as exc:
        raise RAGOutputValidationError("Expected exactly one strict JSON object") from exc
    if not isinstance(value, dict) or set(value) != {"status", "answer", "citations"}:
        raise RAGOutputValidationError("Output must contain only status, answer, citations")
    try:
        status = SemanticStatus(_string(value["status"], "status"))
    except ValueError as exc:
        raise RAGOutputValidationError("Unknown semantic status") from exc
    answer = _string(value["answer"], "answer")
    if not isinstance(value["citations"], list):
        raise RAGOutputValidationError("citations must be an array")
    citations: list[Citation] = []
    for entry in value["citations"]:
        if not isinstance(entry, dict) or set(entry) != {"doc_id", "chunk_id", "quote"}:
            raise RAGOutputValidationError("Citation must contain only doc_id, chunk_id, quote")
        citations.append(Citation(*(_string(entry[key], key)
                                    for key in ("doc_id", "chunk_id", "quote"))))
    return ParsedRAGOutput(status, answer, tuple(citations))


def _whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def validate_output(output: ParsedRAGOutput, context: ContextBundle) -> None:
    """Reject the whole output on any invalid or identical repeated citation."""
    _string(output.answer, "answer")
    authority = {(item.doc_id, item.chunk_id): item.text for item in context.included}
    seen: set[Citation] = set()
    for citation in output.citations:
        _string(citation.quote, "quote")
        if citation in seen:
            raise RAGOutputValidationError("Duplicate citation")
        seen.add(citation)
        source = authority.get((citation.doc_id, citation.chunk_id))
        if source is None:
            raise RAGOutputValidationError("Citation doc/chunk pair is absent from included context")
        if (citation.quote not in source
                and _whitespace(citation.quote) not in _whitespace(source)):
            raise RAGOutputValidationError("Quote is not a verbatim context substring")
    if output.status == SemanticStatus.ANSWERED and not output.citations:
        raise RAGOutputValidationError("ANSWERED requires at least one valid citation")
    if output.status == SemanticStatus.CONFLICTING_EVIDENCE:
        if len(output.citations) < 2 or len({c.doc_id for c in output.citations}) < 2:
            raise RAGOutputValidationError("CONFLICTING_EVIDENCE requires two distinct documents")
    if output.status not in set(SemanticStatus):
        raise RAGOutputValidationError("Unknown semantic status")

