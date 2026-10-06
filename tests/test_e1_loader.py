"""E1 strict SciFact loader tests."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from app.loader import DatasetFormatError, load_corpus, load_qrels, load_queries, load_scifact_dataset


class E1LoaderTests(unittest.TestCase):
    def test_ids_are_strings_and_query_metadata_does_not_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.jsonl"
            queries = root / "queries.jsonl"
            train = root / "train.tsv"
            test = root / "test.tsv"

            corpus.write_text(
                json.dumps({"_id": 7, "title": "T", "text": "Body", "metadata": {}})
                + "\n",
                encoding="utf-8",
            )
            queries.write_text(
                json.dumps(
                    {
                        "_id": 3,
                        "text": "Claim",
                        "metadata": {"gold-doc": [{"label": "SUPPORT"}]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            train.write_text("query-id\tcorpus-id\tscore\n3\t7\t1\n", encoding="utf-8")
            test.write_text("query-id\tcorpus-id\tscore\n3\t7\t1\n", encoding="utf-8")

            dataset = load_scifact_dataset(
                corpus_path=corpus,
                queries_path=queries,
                qrels_train_path=train,
                qrels_test_path=test,
            )
            self.assertEqual(dataset.corpus["7"].doc_id, "7")
            self.assertEqual(dataset.queries["3"].query_id, "3")
            self.assertFalse(hasattr(dataset.queries["3"], "metadata"))
            self.assertEqual(dataset.train_qrels[0].relevance, 1.0)

    def test_duplicate_corpus_id_is_explicit_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.jsonl"
            line = json.dumps({"_id": "1", "title": "", "text": "Body"})
            path.write_text(line + "\n" + line + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetFormatError, r":2: duplicate corpus"):
                load_corpus(path)

    def test_malformed_query_json_reports_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queries.jsonl"
            path.write_text('{"_id":"1","text":"ok"}\n{"_id":\n', encoding="utf-8")
            with self.assertRaisesRegex(DatasetFormatError, r":2: invalid JSON"):
                load_queries(path)

    def test_duplicate_query_and_missing_text_are_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            duplicate = root / "duplicate.jsonl"
            duplicate.write_text(
                '{"_id":"1","text":"A"}\n{"_id":"1","text":"B"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DatasetFormatError, "duplicate query"):
                load_queries(duplicate)

            missing = root / "missing.jsonl"
            missing.write_text('{"_id":"1"}\n', encoding="utf-8")
            with self.assertRaisesRegex(DatasetFormatError, "missing required field"):
                load_queries(missing)

    def test_invalid_score_and_duplicate_qrel_pair_are_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            invalid = root / "invalid.tsv"
            invalid.write_text(
                "query-id\tcorpus-id\tscore\n1\t10\tnot-a-number\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DatasetFormatError, "invalid qrels score"):
                load_qrels(invalid)

            duplicate = root / "duplicate.tsv"
            duplicate.write_text(
                "query-id\tcorpus-id\tscore\n1\t10\t1\n1\t10\t1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DatasetFormatError, "duplicate qrel pair"):
                load_qrels(duplicate)

    def test_qrels_missing_header_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qrels.tsv"
            path.write_text("query-id\tcorpus-id\n1\t2\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetFormatError, "header must be exactly"):
                load_qrels(path)

    def test_qrels_reject_extra_or_reordered_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            extra = root / "extra.tsv"
            extra.write_text(
                "query-id\tcorpus-id\tscore\textra\n1\t2\t1\tx\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DatasetFormatError, "header must be exactly"):
                load_qrels(extra)

            reordered = root / "reordered.tsv"
            reordered.write_text(
                "corpus-id\tquery-id\tscore\n2\t1\t1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DatasetFormatError, "header must be exactly"):
                load_qrels(reordered)

    def test_qrels_reject_extra_missing_and_blank_data_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            extra = root / "extra_row.tsv"
            extra.write_text(
                "query-id\tcorpus-id\tscore\n1\t2\t1\textra\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                DatasetFormatError, "row must contain exactly 3 tab-separated fields"
            ):
                load_qrels(extra)

            missing = root / "missing_row.tsv"
            missing.write_text(
                "query-id\tcorpus-id\tscore\n1\t2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                DatasetFormatError, "row must contain exactly 3 tab-separated fields"
            ):
                load_qrels(missing)

            blank = root / "blank_row.tsv"
            blank.write_text(
                "query-id\tcorpus-id\tscore\n1\t2\t1\n\n3\t4\t1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                DatasetFormatError, "row must contain exactly 3 tab-separated fields"
            ):
                load_qrels(blank)

    def test_unknown_qrel_reference_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.jsonl"
            queries = root / "queries.jsonl"
            train = root / "train.tsv"
            test = root / "test.tsv"
            corpus.write_text('{"_id":"2","title":"T","text":"Body"}\n', encoding="utf-8")
            queries.write_text('{"_id":"1","text":"Claim"}\n', encoding="utf-8")
            train.write_text("query-id\tcorpus-id\tscore\n999\t2\t1\n", encoding="utf-8")
            test.write_text("query-id\tcorpus-id\tscore\n1\t2\t1\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetFormatError, "missing IDs"):
                load_scifact_dataset(
                    corpus_path=corpus,
                    queries_path=queries,
                    qrels_train_path=train,
                    qrels_test_path=test,
                )

    def test_qrels_preserve_multiple_relevant_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qrels.tsv"
            path.write_text(
                "query-id\tcorpus-id\tscore\n1\t10\t1\n1\t11\t1\n",
                encoding="utf-8",
            )
            rows = load_qrels(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual([row.doc_id for row in rows], ["10", "11"])


if __name__ == "__main__":
    unittest.main()
