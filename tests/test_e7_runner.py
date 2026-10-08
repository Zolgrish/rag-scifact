"""Query failure denominators, strict raw artifacts, no evaluator labels in retrieval."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from app.config import REPO_ROOT
from app.loader import BenchmarkQuery
from app.retrieval_evaluator import (aggregate, evaluate_queries, failure_candidates,
    parse_retrieval_rows, publish_run, retrieval_rows, strict_json, RetrievalEvaluationError)
from tests.test_e7_metrics import results


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.queries = tuple(BenchmarkQuery(str(i), "Public question " + str(i)) for i in range(100))
        self.gold = {q.query_id: {"a": 1.} for q in self.queries}

    def outcomes(self):
        retriever = Mock()
        retriever.retrieve.side_effect = [RuntimeError("unexpected query failure")] + [results(["a"])] * 99
        ticks = iter(i / 1000 for i in range(200))
        outcomes = evaluate_queries(retriever, self.queries, clock=lambda: next(ticks))
        self.assertEqual(retriever.retrieve.call_count, 100)
        self.assertEqual(retriever.retrieve.call_args_list[0].args, (self.queries[0].text, 10))
        self.assertTrue(all(call.args == (q.text, 10) and not call.kwargs
                            for call, q in zip(retriever.retrieve.call_args_list, self.queries)))
        return outcomes

    def test_one_top10_call_per_query_and_error_kept_in_denominator(self):
        outcomes = self.outcomes()
        summary = aggregate(outcomes, self.gold)
        self.assertEqual(summary["denominator"], 100)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(set(summary["metrics"].values()), {.99})
        self.assertEqual([o.query_id for o in outcomes], [q.query_id for q in self.queries])
        self.assertTrue(all(o.retrieval_ms >= 0 for o in outcomes))

    def test_roundtrip_fixed_order_explicit_failure_no_fake_doc(self):
        outcomes = self.outcomes()
        rows = retrieval_rows(outcomes, "run")
        self.assertNotIn("doc_id", rows[0])
        self.assertEqual(rows[0]["status"], "ERROR")
        self.assertEqual(parse_retrieval_rows(rows, [q.query_id for q in self.queries], "run"), outcomes)
        for modified in (rows[1:], rows + [rows[-1]], rows[::-1]):
            with self.assertRaises(RetrievalEvaluationError):
                parse_retrieval_rows(modified, [q.query_id for q in self.queries], "run")

    def test_failure_candidates_sorted_with_bounded_root_cause_and_improvement(self):
        retriever = Mock()
        retriever.retrieve.side_effect = [results(["x"]) for _ in self.queries]
        outcomes = evaluate_queries(retriever, self.queries)
        cases = failure_candidates(outcomes, self.queries, self.gold)
        self.assertEqual([c["query_id"] for c in cases], [str(i) for i in range(100)])
        self.assertTrue(all(c["category"] == "retrieval_miss" for c in cases))
        self.assertTrue(all("Top-10 dense-retrieval miss" in c["root_cause_assessment"] for c in cases))
        self.assertTrue(all("does not prove a deeper" in c["root_cause_assessment"] for c in cases))
        self.assertTrue(all(c["observed_output"] and c["concrete_possible_improvement"] for c in cases))

    def test_result_contract_violation_aborts_instead_of_becoming_query_failure(self):
        retriever = Mock()
        duplicate = results(["a", "b"])
        duplicate[1] = duplicate[1].__class__("a", "a:0-1", 2, .5)
        retriever.retrieve.return_value = duplicate
        with self.assertRaisesRegex(RetrievalEvaluationError, "unique documents"):
            evaluate_queries(retriever, self.queries[:1])

    def test_atomic_strict_publication_and_immutable_directory(self):
        outcomes = self.outcomes()
        rows = retrieval_rows(outcomes, "run")
        cases = failure_candidates(outcomes, self.queries, self.gold)
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "artifacts") as tmp:
            root = Path(tmp) / "run"
            completed = publish_run(root, rows, cases, aggregate(outcomes, self.gold))
            for name in ("retrieval_run.jsonl", "failure_analysis.jsonl", "metrics.json"):
                self.assertNotIn(b"\r", (root / name).read_bytes())
            self.assertEqual(strict_json((root / "metrics.json").read_bytes()), completed)
            self.assertEqual(len([strict_json(line) for line in (root / "retrieval_run.jsonl").read_bytes().splitlines()]), 199)
            with self.assertRaises(FileExistsError): publish_run(root, rows, cases, {})
            with self.assertRaises(ValueError): publish_run(Path(tmp) / "bad", rows, cases, {"nan": math.nan})
            self.assertFalse((Path(tmp) / "bad").exists())
        for value in ('{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{"x":1,"x":2}'):
            with self.assertRaises(ValueError): strict_json(value)


if __name__ == "__main__": unittest.main()
