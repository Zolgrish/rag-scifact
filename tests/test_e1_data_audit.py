"""E1 deterministic split, overlap policy, and real-data audit tests."""

from __future__ import annotations

from dataclasses import replace as dc_replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from app.config import load_config
from app.data_audit import (
    EXPECTED_FILE_SHA256,
    SPLIT_SEED,
    audit_query_overlap,
    audit_scifact_dataset,
    build_dev_practice_split,
    file_sha256,
    normalize_query_text,
    token_edit_distance,
)
from app.loader import BenchmarkQuery, DatasetMismatchError, Qrel


class E1DataAuditTests(unittest.TestCase):
    def test_split_is_deterministic_and_has_required_sizes(self) -> None:
        qrels = tuple(Qrel(str(i), "doc", 1.0) for i in range(809))
        first = build_dev_practice_split(qrels)
        second = build_dev_practice_split(qrels)
        self.assertEqual(first, second)
        self.assertEqual(first.seed, 42)
        self.assertEqual(len(first.dev_ids), 100)
        self.assertEqual(len(first.practice_ids), 709)
        self.assertFalse(set(first.dev_ids) & set(first.practice_ids))
        self.assertEqual(
            set(first.dev_ids) | set(first.practice_ids),
            {str(i) for i in range(809)},
        )

    def test_split_helper_rejects_noncanonical_seed(self) -> None:
        qrels = tuple(Qrel(str(i), "doc", 1.0) for i in range(101))
        with self.assertRaisesRegex(DatasetMismatchError, "split seed is fixed at 42"):
            build_dev_practice_split(qrels, seed=43)

    def test_split_uses_every_unique_qrel_query_id_even_score_zero(self) -> None:
        qrels = (
            Qrel("1", "d1", 0.0),
            Qrel("2", "d2", 1.0),
            Qrel("3", "d3", 1.0),
        )
        split = build_dev_practice_split(qrels, seed=SPLIT_SEED, dev_size=1)
        self.assertEqual(
            set(split.dev_ids) | set(split.practice_ids),
            {"1", "2", "3"},
        )

    def test_audit_rejects_config_seed_other_than_42(self) -> None:
        config = load_config()
        bad_config = dc_replace(config, seed=43)
        with self.assertRaisesRegex(DatasetMismatchError, "split seed is fixed at 42"):
            audit_scifact_dataset(bad_config)

    def test_exact_normalization_handles_unicode_and_preserves_punctuation_numbers(self) -> None:
        self.assertEqual(
            normalize_query_text("\u00c1BC Test"),
            normalize_query_text("A\u0301bc   test"),
        )
        self.assertEqual(
            normalize_query_text("\uff21\uff22\uff23 Test"),
            normalize_query_text("abc test"),
        )
        self.assertNotEqual(
            normalize_query_text("risk is 10%"),
            normalize_query_text("risk is 20%"),
        )
        self.assertNotEqual(
            normalize_query_text("A, B"),
            normalize_query_text("A B"),
        )

    def test_token_edit_distance(self) -> None:
        self.assertEqual(token_edit_distance(("a", "b"), ("a", "b")), 0)
        self.assertEqual(token_edit_distance(("a", "b"), ("a", "c")), 1)
        self.assertEqual(token_edit_distance(("a", "b"), ("a", "x", "b")), 1)

    def test_overlap_audit_exact_near_and_short_query_policy(self) -> None:
        queries = {
            "1": BenchmarkQuery("1", "Same scientific claim appears here."),
            "2": BenchmarkQuery("2", "Aspirin reduces risk of heart attack in adults"),
            "3": BenchmarkQuery("3", "  SAME scientific claim appears here.  "),
            "4": BenchmarkQuery("4", "Aspirin reduces the risk of heart attack in adults"),
            "5": BenchmarkQuery("5", "short text"),
            "6": BenchmarkQuery("6", "short texts"),
        }
        result = audit_query_overlap(
            queries,
            practice_ids=("1", "2", "5"),
            dev_ids=("3",),
            test_ids=("4", "6"),
        )
        self.assertEqual(
            result["by_split_pair"]["practice_vs_dev"]["canonical_exact_duplicate_pairs"],
            1,
        )
        self.assertEqual(
            result["by_split_pair"]["practice_vs_test"]["near_duplicate_pairs"],
            1,
        )
        self.assertEqual(result["totals"]["near_duplicate_pairs"], 1)
        serialized = repr(result)
        self.assertNotIn("Aspirin", serialized)
        self.assertNotIn("Same scientific", serialized)

    def test_audit_rejects_same_shape_dataset_with_changed_bytes(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmp:
            corpus_copy = Path(tmp) / "corpus.jsonl"
            shutil.copy2(config.paths.corpus, corpus_copy)

            lines = corpus_copy.read_text(encoding="utf-8").splitlines()
            first = json.loads(lines[0])
            first["text"] = first["text"] + " "
            lines[0] = json.dumps(first, ensure_ascii=False)
            corpus_copy.write_text("\n".join(lines) + "\n", encoding="utf-8")

            bad_paths = dc_replace(config.paths, corpus=corpus_copy)
            bad_config = dc_replace(config, paths=bad_paths)
            with self.assertRaisesRegex(DatasetMismatchError, "dataset hash mismatch"):
                audit_scifact_dataset(bad_config)

    def test_real_scifact_counts_hashes_and_split(self) -> None:
        config = load_config()
        dataset, split, audit = audit_scifact_dataset(config)
        self.assertEqual(len(dataset.corpus), 5183)
        self.assertEqual(len(dataset.queries), 1109)
        self.assertEqual(audit["counts"]["train_query_count"], 809)
        self.assertEqual(audit["counts"]["test_query_count"], 300)
        self.assertEqual(audit["counts"]["train_qrel_row_count"], 919)
        self.assertEqual(audit["counts"]["test_qrel_row_count"], 339)
        self.assertEqual(len(split.dev_ids), 100)
        self.assertEqual(len(split.practice_ids), 709)
        self.assertEqual(split.dev_ids[:5], ("1297", "1276", "383", "611", "1126"))

        overlap = audit["query_overlap_audit"]
        self.assertEqual(overlap["totals"]["raw_exact_duplicate_pairs"], 2)
        self.assertEqual(overlap["totals"]["canonical_exact_duplicate_pairs"], 2)
        self.assertEqual(overlap["totals"]["near_duplicate_pairs"], 98)
        self.assertEqual(
            overlap["by_split_pair"]["practice_vs_dev"]["near_duplicate_pairs"], 25
        )
        self.assertEqual(
            overlap["by_split_pair"]["practice_vs_test"]["near_duplicate_pairs"], 66
        )
        self.assertEqual(
            overlap["by_split_pair"]["dev_vs_test"]["near_duplicate_pairs"], 7
        )

        paths = {
            "corpus": config.paths.corpus,
            "queries": config.paths.queries,
            "qrels_train": config.paths.qrels_train,
            "qrels_test": config.paths.qrels_test,
        }
        for name, path in paths.items():
            with self.subTest(name=name):
                self.assertEqual(file_sha256(path), EXPECTED_FILE_SHA256[name])


if __name__ == "__main__":
    unittest.main()
