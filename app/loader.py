"""Strict reusable loaders for the SciFact BEIR package.

Runtime query loading intentionally exposes only query ID and text. SciFact
query metadata can contain evaluator/gold evidence, so it is never carried into
retrieval or generation objects.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping


class DatasetError(RuntimeError):
    """Base class for SciFact dataset failures."""


class DatasetFormatError(DatasetError):
    """Raised when a source file is missing, malformed, or internally invalid."""


class DatasetMismatchError(DatasetError):
    """Raised when staged SciFact does not match the benchmark contract."""


@dataclass(frozen=True)
class CorpusDocument:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: str
    text: str


@dataclass(frozen=True)
class Qrel:
    query_id: str
    doc_id: str
    relevance: float


@dataclass(frozen=True)
class SciFactDataset:
    corpus: dict[str, CorpusDocument]
    queries: dict[str, BenchmarkQuery]
    train_qrels: tuple[Qrel, ...]
    test_qrels: tuple[Qrel, ...]


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise DatasetFormatError(f"Required dataset file does not exist: {path}")


def _normalize_id(value: Any, *, path: Path, line_no: int, field: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise DatasetFormatError(
            f"{path}:{line_no}: field {field!r} must be a string/integer ID"
        )
    normalized = str(value).strip()
    if not normalized:
        raise DatasetFormatError(f"{path}:{line_no}: field {field!r} is empty")
    return normalized


def _read_jsonl(path: Path) -> list[tuple[int, Mapping[str, Any]]]:
    _require_file(path)
    records: list[tuple[int, Mapping[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise DatasetFormatError(f"{path}:{line_no}: blank JSONL record")
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise DatasetFormatError(
                    f"{path}:{line_no}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(value, Mapping):
                raise DatasetFormatError(
                    f"{path}:{line_no}: expected a JSON object record"
                )
            records.append((line_no, value))
    return records


def _required_string(
    record: Mapping[str, Any],
    field: str,
    *,
    path: Path,
    line_no: int,
    allow_empty: bool,
) -> str:
    if field not in record:
        raise DatasetFormatError(f"{path}:{line_no}: missing required field {field!r}")
    value = record[field]
    if not isinstance(value, str):
        raise DatasetFormatError(
            f"{path}:{line_no}: field {field!r} must be a string"
        )
    if not allow_empty and not value.strip():
        raise DatasetFormatError(f"{path}:{line_no}: field {field!r} is empty")
    return value


def load_corpus(path: str | Path) -> dict[str, CorpusDocument]:
    """Load corpus JSONL with strict duplicate/field validation."""

    resolved = Path(path)
    corpus: dict[str, CorpusDocument] = {}
    for line_no, record in _read_jsonl(resolved):
        if "_id" not in record:
            raise DatasetFormatError(
                f"{resolved}:{line_no}: missing required field '_id'"
            )
        doc_id = _normalize_id(
            record["_id"], path=resolved, line_no=line_no, field="_id"
        )
        if doc_id in corpus:
            raise DatasetFormatError(
                f"{resolved}:{line_no}: duplicate corpus _id {doc_id!r}"
            )
        title = _required_string(
            record, "title", path=resolved, line_no=line_no, allow_empty=True
        )
        text = _required_string(
            record, "text", path=resolved, line_no=line_no, allow_empty=False
        )
        corpus[doc_id] = CorpusDocument(doc_id=doc_id, title=title, text=text)
    return corpus


def load_queries(path: str | Path) -> dict[str, BenchmarkQuery]:
    """Load runtime-safe query IDs/text only; source metadata is ignored."""

    resolved = Path(path)
    queries: dict[str, BenchmarkQuery] = {}
    for line_no, record in _read_jsonl(resolved):
        if "_id" not in record:
            raise DatasetFormatError(
                f"{resolved}:{line_no}: missing required field '_id'"
            )
        query_id = _normalize_id(
            record["_id"], path=resolved, line_no=line_no, field="_id"
        )
        if query_id in queries:
            raise DatasetFormatError(
                f"{resolved}:{line_no}: duplicate query _id {query_id!r}"
            )
        text = _required_string(
            record, "text", path=resolved, line_no=line_no, allow_empty=False
        )
        queries[query_id] = BenchmarkQuery(query_id=query_id, text=text)
    return queries


def load_qrels(path: str | Path) -> tuple[Qrel, ...]:
    """Load evaluator-only BEIR qrels with strict TSV validation."""

    resolved = Path(path)
    _require_file(resolved)
    rows: list[Qrel] = []
    seen_pairs: set[tuple[str, str]] = set()

    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        expected_fields = ["query-id", "corpus-id", "score"]
        try:
            header = next(reader)
        except StopIteration as exc:
            raise DatasetFormatError(
                f"{resolved}: qrels header must be exactly {expected_fields!r}; "
                "file is empty"
            ) from exc
        if header != expected_fields:
            raise DatasetFormatError(
                f"{resolved}: qrels header must be exactly {expected_fields!r}; "
                f"got {header!r}"
            )

        for row in reader:
            line_no = reader.line_num
            if len(row) != len(expected_fields):
                raise DatasetFormatError(
                    f"{resolved}:{line_no}: qrels row must contain exactly "
                    f"{len(expected_fields)} tab-separated fields; got {len(row)}"
                )

            raw_query_id, raw_doc_id, raw_score = row
            query_id = _normalize_id(
                raw_query_id,
                path=resolved,
                line_no=line_no,
                field="query-id",
            )
            doc_id = _normalize_id(
                raw_doc_id,
                path=resolved,
                line_no=line_no,
                field="corpus-id",
            )
            try:
                relevance = float(raw_score) if raw_score is not None else math.nan
            except (TypeError, ValueError) as exc:
                raise DatasetFormatError(
                    f"{resolved}:{line_no}: invalid qrels score {raw_score!r}"
                ) from exc
            if not math.isfinite(relevance) or relevance < 0:
                raise DatasetFormatError(
                    f"{resolved}:{line_no}: qrels score must be finite and >= 0"
                )

            pair = (query_id, doc_id)
            if pair in seen_pairs:
                raise DatasetFormatError(
                    f"{resolved}:{line_no}: duplicate qrel pair "
                    f"query={query_id!r}, corpus={doc_id!r}"
                )
            seen_pairs.add(pair)
            rows.append(Qrel(query_id=query_id, doc_id=doc_id, relevance=relevance))

    return tuple(rows)


def _sortable_id(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def validate_qrel_references(
    qrels: tuple[Qrel, ...],
    *,
    queries: Mapping[str, BenchmarkQuery],
    corpus: Mapping[str, CorpusDocument],
    source_name: str,
) -> None:
    """Ensure qrels do not point at absent query/document IDs."""

    missing_queries = sorted(
        {row.query_id for row in qrels if row.query_id not in queries},
        key=_sortable_id,
    )
    missing_docs = sorted(
        {row.doc_id for row in qrels if row.doc_id not in corpus},
        key=_sortable_id,
    )
    if missing_queries or missing_docs:
        raise DatasetFormatError(
            f"{source_name}: qrels reference missing IDs; "
            f"queries={missing_queries[:10]}, corpus={missing_docs[:10]}"
        )


def qrel_query_ids(qrels: tuple[Qrel, ...]) -> set[str]:
    """Return every unique query ID present in a qrels file."""

    return {row.query_id for row in qrels}


def relevant_query_ids(qrels: tuple[Qrel, ...]) -> set[str]:
    """Return query IDs with positive relevance judgments."""

    return {row.query_id for row in qrels if row.relevance > 0}


def load_scifact_dataset(
    *,
    corpus_path: str | Path,
    queries_path: str | Path,
    qrels_train_path: str | Path,
    qrels_test_path: str | Path,
) -> SciFactDataset:
    """Load and cross-validate the complete staged SciFact package."""

    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    train_qrels = load_qrels(qrels_train_path)
    test_qrels = load_qrels(qrels_test_path)

    validate_qrel_references(
        train_qrels,
        queries=queries,
        corpus=corpus,
        source_name=str(qrels_train_path),
    )
    validate_qrel_references(
        test_qrels,
        queries=queries,
        corpus=corpus,
        source_name=str(qrels_test_path),
    )

    return SciFactDataset(
        corpus=corpus,
        queries=queries,
        train_qrels=train_qrels,
        test_qrels=test_qrels,
    )


__all__ = [
    "BenchmarkQuery",
    "CorpusDocument",
    "DatasetError",
    "DatasetFormatError",
    "DatasetMismatchError",
    "Qrel",
    "SciFactDataset",
    "load_corpus",
    "load_qrels",
    "load_queries",
    "load_scifact_dataset",
    "qrel_query_ids",
    "relevant_query_ids",
    "validate_qrel_references",
]
