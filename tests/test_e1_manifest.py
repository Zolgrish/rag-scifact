"""E1 manifest merge and conflict-safety tests."""

from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

from app.config import load_config
from app.data_audit import DatasetSplit
from app.manifest import ManifestConflictError, merge_e1_manifest


def _audit(hash_value: str = "abc") -> dict[str, object]:
    return {
        "status": "PASS",
        "counts": {
            "corpus_count": 5183,
            "query_count": 1109,
            "train_query_count": 809,
            "test_query_count": 300,
            "train_qrel_row_count": 919,
            "test_qrel_row_count": 339,
        },
        "hashes": {
            "corpus": {"path": "data/scifact/corpus.jsonl", "sha256": hash_value},
            "queries": {"path": "data/scifact/queries.jsonl", "sha256": "q"},
            "qrels_train": {"path": "data/scifact/qrels/train.tsv", "sha256": "tr"},
            "qrels_test": {"path": "data/scifact/qrels/test.tsv", "sha256": "te"},
        },
        "validation": {"malformed_records": 0},
        "query_overlap_audit": {"policy": {"version": "query_overlap_v1"}, "totals": {}},
    }


class E1ManifestTests(unittest.TestCase):
    def test_merge_preserves_unrelated_sections(self) -> None:
        config = load_config()
        split = DatasetSplit(seed=42, dev_ids=("1",), practice_ids=("2",))
        existing = {"index": {"status": "future-value"}, "custom": {"keep": True}}
        merged = merge_e1_manifest(existing, config=config, split=split, audit=_audit())
        self.assertEqual(merged["index"], {"status": "future-value"})
        self.assertEqual(merged["custom"], {"keep": True})
        self.assertEqual(merged["split"]["seed"], 42)
        self.assertEqual(merged["dataset"]["corpus_count"], 5183)

    def test_same_e1_identity_can_merge_idempotently(self) -> None:
        config = load_config()
        split = DatasetSplit(seed=42, dev_ids=("1",), practice_ids=("2",))
        first = merge_e1_manifest(
            {"custom": {"keep": True}},
            config=config,
            split=split,
            audit=_audit("same"),
        )
        second = merge_e1_manifest(
            deepcopy(first),
            config=config,
            split=split,
            audit=_audit("same"),
        )
        self.assertEqual(second["dataset"], first["dataset"])
        self.assertEqual(second["split"], first["split"])
        self.assertEqual(second["custom"], {"keep": True})

    def test_frozen_manifest_allows_same_clean_commit(self) -> None:
        config = load_config()
        split = DatasetSplit(seed=42, dev_ids=("1",), practice_ids=("2",))
        existing = {
            "git_commit": "abc123",
            "git_dirty": False,
            "freeze": {
                "final_test_frozen": True,
                "runtime_profile_locked": True,
            },
        }
        with patch("app.manifest._git_state", return_value=("abc123", False)):
            merged = merge_e1_manifest(
                existing,
                config=config,
                split=split,
                audit=_audit("same"),
            )
        self.assertTrue(merged["freeze"]["final_test_frozen"])
        self.assertFalse(merged["git_dirty"])
        self.assertEqual(merged["git_commit"], "abc123")

    def test_frozen_manifest_rejects_dirty_working_tree(self) -> None:
        config = load_config()
        split = DatasetSplit(seed=42, dev_ids=("1",), practice_ids=("2",))
        existing = {
            "git_commit": "abc123",
            "git_dirty": False,
            "freeze": {
                "final_test_frozen": True,
                "runtime_profile_locked": True,
            },
        }
        with patch("app.manifest._git_state", return_value=("abc123", True)):
            with self.assertRaisesRegex(
                ManifestConflictError, "dirty working tree"
            ):
                merge_e1_manifest(
                    existing,
                    config=config,
                    split=split,
                    audit=_audit("same"),
                )

    def test_frozen_manifest_rejects_different_commit(self) -> None:
        config = load_config()
        split = DatasetSplit(seed=42, dev_ids=("1",), practice_ids=("2",))
        existing = {
            "git_commit": "abc123",
            "git_dirty": False,
            "freeze": {
                "final_test_frozen": True,
                "runtime_profile_locked": True,
            },
        }
        with patch("app.manifest._git_state", return_value=("def456", False)):
            with self.assertRaisesRegex(
                ManifestConflictError, "different git commit"
            ):
                merge_e1_manifest(
                    existing,
                    config=config,
                    split=split,
                    audit=_audit("same"),
                )

    def test_frozen_manifest_rejects_preexisting_dirty_freeze(self) -> None:
        config = load_config()
        split = DatasetSplit(seed=42, dev_ids=("1",), practice_ids=("2",))
        existing = {
            "git_commit": "abc123",
            "git_dirty": True,
            "freeze": {
                "final_test_frozen": True,
                "runtime_profile_locked": True,
            },
        }
        with patch("app.manifest._git_state", return_value=("abc123", False)):
            with self.assertRaisesRegex(
                ManifestConflictError, "must record git_dirty=false"
            ):
                merge_e1_manifest(
                    existing,
                    config=config,
                    split=split,
                    audit=_audit("same"),
                )

    def test_conflicting_dataset_hash_fails_instead_of_overwriting(self) -> None:
        config = load_config()
        split = DatasetSplit(seed=42, dev_ids=("1",), practice_ids=("2",))
        first = merge_e1_manifest({}, config=config, split=split, audit=_audit("one"))
        with self.assertRaises(ManifestConflictError):
            merge_e1_manifest(
                deepcopy(first),
                config=config,
                split=split,
                audit=_audit("two"),
            )


if __name__ == "__main__":
    unittest.main()
