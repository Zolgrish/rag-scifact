"""E1 SciFact audit, deterministic split, hashing, and overlap checks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
import re
from pathlib import Path
import unicodedata
from typing import Iterable, Mapping

from app.config import AppConfig, REPO_ROOT
from app.loader import (
    BenchmarkQuery,
    DatasetMismatchError,
    Qrel,
    SciFactDataset,
    load_scifact_dataset,
    qrel_query_ids,
)


EXPECTED_CORPUS_COUNT = 5183
EXPECTED_QUERY_COUNT = 1109
EXPECTED_TRAIN_QUERY_COUNT = 809
EXPECTED_TEST_QUERY_COUNT = 300
SPLIT_SEED = 42
DEV_SIZE = 100

EXPECTED_FILE_SHA256 = {
    "corpus": "dec31c8182f3d744c7d2c09423756fd1d17cbef75808db13ba01cc0aab4d1ac6",
    "queries": "8ff84a7c903f722981cd8d595c022660140c51867b27608a6d4910db86080313",
    "qrels_train": "a53f2114831916c096b6c37d9e54da68cef4efdcdbd5ed46533601af972acf1d",
    "qrels_test": "0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c",
}

OVERLAP_POLICY_VERSION = "query_overlap_v1"
NEAR_MAX_EDIT_FRACTION = 0.10
NEAR_MIN_TOKENS = 5

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class DatasetSplit:
    seed: int
    dev_ids: tuple[str, ...]
    practice_ids: tuple[str, ...]


def file_sha256(path: str | Path) -> str:
    """SHA256 of raw source bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def _numeric_id(value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise DatasetMismatchError(
            f"SciFact split contract requires numeric query IDs, got {value!r}"
        ) from exc


def build_dev_practice_split(
    train_qrels: tuple[Qrel, ...],
    *,
    seed: int = SPLIT_SEED,
    dev_size: int = DEV_SIZE,
) -> DatasetSplit:
    """Canonical numeric-sort, Random(42)-shuffle split."""

    if seed != SPLIT_SEED:
        raise DatasetMismatchError(
            f"E1 split seed is fixed at {SPLIT_SEED}; got {seed}"
        )

    ids = sorted(qrel_query_ids(train_qrels), key=_numeric_id)
    rng = random.Random(seed)
    rng.shuffle(ids)
    if len(ids) < dev_size:
        raise DatasetMismatchError(
            f"Cannot create dev split of {dev_size}; only {len(ids)} train query IDs"
        )
    return DatasetSplit(
        seed=seed,
        dev_ids=tuple(ids[:dev_size]),
        practice_ids=tuple(ids[dev_size:]),
    )


def normalize_query_text(text: str) -> str:
    """Exact-overlap normalization used only by the integrity audit."""

    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def tokenize_query_text(text: str) -> tuple[str, ...]:
    """Near-overlap tokenization: Unicode NFKC/casefold word tokens."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(_TOKEN_RE.findall(normalized))


def token_edit_distance(
    left: tuple[str, ...],
    right: tuple[str, ...],
    *,
    limit: int | None = None,
) -> int:
    """Levenshtein distance over tokens, with optional early cutoff."""

    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) > len(right):
        left, right = right, left
    if limit is not None and len(right) - len(left) > limit:
        return limit + 1

    previous = list(range(len(left) + 1))
    for row_index, right_token in enumerate(right, start=1):
        current = [row_index]
        row_minimum = current[0]
        for col_index, left_token in enumerate(left, start=1):
            insertion = current[col_index - 1] + 1
            deletion = previous[col_index] + 1
            substitution = previous[col_index - 1] + (
                0 if left_token == right_token else 1
            )
            value = min(insertion, deletion, substitution)
            current.append(value)
            row_minimum = min(row_minimum, value)
        if limit is not None and row_minimum > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _near_budget(left: tuple[str, ...], right: tuple[str, ...]) -> int | None:
    max_tokens = max(len(left), len(right))
    if max_tokens < NEAR_MIN_TOKENS:
        return None
    return max(1, int(NEAR_MAX_EDIT_FRACTION * max_tokens))


def _is_near_duplicate(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    budget = _near_budget(left, right)
    if budget is None:
        return False
    if abs(len(left) - len(right)) > budget:
        return False
    return token_edit_distance(left, right, limit=budget) <= budget


def audit_query_overlap(
    queries: Mapping[str, BenchmarkQuery],
    *,
    practice_ids: Iterable[str],
    dev_ids: Iterable[str],
    test_ids: Iterable[str],
) -> dict[str, object]:
    """Aggregate exact/near duplicate counts across split boundaries only.

    No test query text or test query IDs are emitted. The result is an
    integrity report and must never drive query removal, denominator changes,
    retrieval tuning, prompt tuning, or model selection.
    """

    split_ids = {
        "practice": tuple(practice_ids),
        "dev": tuple(dev_ids),
        "test": tuple(test_ids),
    }
    missing = {
        query_id
        for ids in split_ids.values()
        for query_id in ids
        if query_id not in queries
    }
    if missing:
        raise DatasetMismatchError(
            "Query-overlap audit references missing query IDs: "
            + ", ".join(sorted(missing, key=_numeric_id)[:10])
        )

    raw_text = {
        query_id: queries[query_id].text
        for ids in split_ids.values()
        for query_id in ids
    }
    canonical = {
        query_id: normalize_query_text(text) for query_id, text in raw_text.items()
    }
    tokens = {
        query_id: tokenize_query_text(text) for query_id, text in raw_text.items()
    }

    comparisons = (("practice", "dev"), ("practice", "test"), ("dev", "test"))
    by_pair: dict[str, dict[str, int]] = {}
    totals = {
        "pairs_compared": 0,
        "raw_exact_duplicate_pairs": 0,
        "canonical_exact_duplicate_pairs": 0,
        "near_duplicate_pairs": 0,
    }

    for left_name, right_name in comparisons:
        counts = {
            "pairs_compared": 0,
            "raw_exact_duplicate_pairs": 0,
            "canonical_exact_duplicate_pairs": 0,
            "near_duplicate_pairs": 0,
        }
        for left_id in split_ids[left_name]:
            for right_id in split_ids[right_name]:
                counts["pairs_compared"] += 1
                if raw_text[left_id] == raw_text[right_id]:
                    counts["raw_exact_duplicate_pairs"] += 1
                    counts["canonical_exact_duplicate_pairs"] += 1
                    continue
                if canonical[left_id] == canonical[right_id]:
                    counts["canonical_exact_duplicate_pairs"] += 1
                    continue
                if _is_near_duplicate(tokens[left_id], tokens[right_id]):
                    counts["near_duplicate_pairs"] += 1

        by_pair[f"{left_name}_vs_{right_name}"] = counts
        for key in totals:
            totals[key] += counts[key]

    return {
        "policy": {
            "version": OVERLAP_POLICY_VERSION,
            "purpose": "cross-split data-integrity audit only; never used for tuning",
            "raw_exact": "source query strings are equal after JSON decoding",
            "canonical_exact": "Unicode NFKC + casefold + whitespace collapse",
            "near_duplicate": {
                "exact_pairs_excluded": True,
                "tokenization": "Unicode NFKC + casefold + word tokens",
                "minimum_max_token_count": NEAR_MIN_TOKENS,
                "distance": "token-level Levenshtein",
                "max_distance": "max(1, floor(0.10 * max_token_count))",
                "stemming": False,
                "stopword_removal": False,
                "semantic_embeddings": False,
            },
            "published_data": "aggregate counts only; no test query text or IDs",
        },
        "totals": totals,
        "by_split_pair": by_pair,
    }


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _hash_entry(path: Path) -> dict[str, str]:
    return {"path": _relative(path), "sha256": file_sha256(path)}


def audit_scifact_dataset(
    config: AppConfig,
) -> tuple[SciFactDataset, DatasetSplit, dict[str, object]]:
    """Validate the staged package and return deterministic E1 audit data."""

    dataset = load_scifact_dataset(
        corpus_path=config.paths.corpus,
        queries_path=config.paths.queries,
        qrels_train_path=config.paths.qrels_train,
        qrels_test_path=config.paths.qrels_test,
    )

    if config.seed != SPLIT_SEED:
        raise DatasetMismatchError(
            f"E1 split seed is fixed at {SPLIT_SEED}; config has {config.seed}"
        )

    train_ids = qrel_query_ids(dataset.train_qrels)
    test_ids = qrel_query_ids(dataset.test_qrels)
    counts = {
        "corpus_count": len(dataset.corpus),
        "query_count": len(dataset.queries),
        "train_query_count": len(train_ids),
        "test_query_count": len(test_ids),
        "train_qrel_row_count": len(dataset.train_qrels),
        "test_qrel_row_count": len(dataset.test_qrels),
    }
    expected = {
        "corpus_count": EXPECTED_CORPUS_COUNT,
        "query_count": EXPECTED_QUERY_COUNT,
        "train_query_count": EXPECTED_TRAIN_QUERY_COUNT,
        "test_query_count": EXPECTED_TEST_QUERY_COUNT,
    }
    mismatch = {
        key: {"expected": value, "actual": counts[key]}
        for key, value in expected.items()
        if counts[key] != value
    }
    if mismatch:
        raise DatasetMismatchError(
            "SciFact dataset/version mismatch: "
            + json.dumps(mismatch, sort_keys=True)
        )

    train_test_overlap = sorted(train_ids & test_ids, key=_numeric_id)
    if train_test_overlap:
        raise DatasetMismatchError(
            f"Train/test query IDs overlap: {train_test_overlap[:10]}"
        )

    split = build_dev_practice_split(
        dataset.train_qrels,
        seed=SPLIT_SEED,
        dev_size=DEV_SIZE,
    )
    if len(split.dev_ids) != 100 or len(split.practice_ids) != 709:
        raise DatasetMismatchError(
            "Deterministic split size mismatch: "
            f"dev={len(split.dev_ids)}, practice={len(split.practice_ids)}"
        )

    hashes = {
        "corpus": _hash_entry(config.paths.corpus),
        "queries": _hash_entry(config.paths.queries),
        "qrels_train": _hash_entry(config.paths.qrels_train),
        "qrels_test": _hash_entry(config.paths.qrels_test),
    }
    hash_mismatch = {
        name: {
            "expected": EXPECTED_FILE_SHA256[name],
            "actual": entry["sha256"],
        }
        for name, entry in hashes.items()
        if entry["sha256"] != EXPECTED_FILE_SHA256[name]
    }
    if hash_mismatch:
        raise DatasetMismatchError(
            "SciFact dataset hash mismatch: "
            + json.dumps(hash_mismatch, sort_keys=True)
        )

    overlap_audit = audit_query_overlap(
        dataset.queries,
        practice_ids=split.practice_ids,
        dev_ids=split.dev_ids,
        test_ids=sorted(test_ids, key=_numeric_id),
    )

    audit: dict[str, object] = {
        "status": "PASS",
        "counts": counts,
        "expected_counts": expected,
        "hashes": hashes,
        "validation": {
            "duplicate_corpus_ids": 0,
            "duplicate_query_ids": 0,
            "duplicate_qrel_pairs": 0,
            "missing_qrel_query_references": 0,
            "missing_qrel_corpus_references": 0,
            "train_test_query_id_overlap": 0,
            "malformed_records": 0,
        },
        "split": {
            "seed": split.seed,
            "dev_count": len(split.dev_ids),
            "practice_count": len(split.practice_ids),
            "dev_first_five_ids": list(split.dev_ids[:5]),
        },
        "query_overlap_audit": overlap_audit,
    }
    return dataset, split, audit


__all__ = [
    "DEV_SIZE",
    "DatasetSplit",
    "EXPECTED_CORPUS_COUNT",
    "EXPECTED_FILE_SHA256",
    "EXPECTED_QUERY_COUNT",
    "EXPECTED_TEST_QUERY_COUNT",
    "EXPECTED_TRAIN_QUERY_COUNT",
    "NEAR_MAX_EDIT_FRACTION",
    "NEAR_MIN_TOKENS",
    "OVERLAP_POLICY_VERSION",
    "SPLIT_SEED",
    "audit_query_overlap",
    "audit_scifact_dataset",
    "build_dev_practice_split",
    "file_sha256",
    "normalize_query_text",
    "token_edit_distance",
    "tokenize_query_text",
]
