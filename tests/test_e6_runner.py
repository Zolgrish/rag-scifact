"""Six-case denominator, no retry, strict atomic artifacts and isolated inputs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from app.citations import RAGOutputValidationError
from app.config import REPO_ROOT
from app.evaluator import (GENERATION_CASES, generation_summary, publish_generation_run,
                           run_generation_suite)
from app.generator import GeneratorConnectionError
from tests.test_e6_grader import execution


class RunnerTests(unittest.TestCase):
    def test_only_question_enters_runtime_and_cases_fixed_order(self):
        pipeline = Mock()
        pipeline.ask.side_effect = [execution(c) for c in GENERATION_CASES]
        rows = run_generation_suite(pipeline, run_id="run", top_k=5)
        self.assertEqual([r["case_id"] for r in rows], [c.case_id for c in GENERATION_CASES])
        self.assertTrue(all(r["grading"]["passed"] for r in rows))
        for call, case in zip(pipeline.ask.call_args_list, GENERATION_CASES):
            self.assertEqual(call.args, (case.question, 5))
            self.assertEqual(call.kwargs, {"request_id": "run-" + case.case_id})

    def test_errors_continue_no_retry_and_latency_excludes_missing_responses(self):
        pipeline = Mock()
        pipeline.ask.side_effect = [RAGOutputValidationError("invalid quote"), GeneratorConnectionError("down")] + [execution(c) for c in GENERATION_CASES[2:]]
        rows = run_generation_suite(pipeline, run_id="run", top_k=5)
        self.assertEqual(len(rows), 6)
        self.assertEqual(pipeline.ask.call_count, 6)
        self.assertEqual(rows[0]["grading"]["failure_codes"], ["RAG_OUTPUT_VALIDATION_ERROR"])
        self.assertEqual(rows[1]["grading"]["failure_codes"], ["INFRASTRUCTURE_ERROR"])
        self.assertIsNone(rows[0]["response"])
        summary = generation_summary(rows, provenance={"run_id": "run"})
        self.assertEqual((summary["cases"], summary["passed"], summary["failed"]), (6, 4, 2))
        self.assertEqual(summary["latency_samples"], 4)
        self.assertEqual(summary["latency_ms"]["generation_p50"], 2.)

    def test_unexpected_programming_errors_are_not_infrastructure(self):
        pipeline = Mock()
        pipeline.ask.side_effect = [ValueError("bug"), TypeError("bug")] + [execution(c) for c in GENERATION_CASES[2:]]
        rows = run_generation_suite(pipeline, run_id="run", top_k=5)
        self.assertEqual(rows[0]["grading"]["failure_codes"], ["EVALUATOR_ERROR"])
        self.assertEqual(rows[1]["grading"]["failure_codes"], ["EVALUATOR_ERROR"])
        self.assertEqual(pipeline.ask.call_count, 6)

    def test_utf8_lf_jsonl_fixed_order_and_strict_nonfinite_rejection(self):
        pipeline = Mock()
        pipeline.ask.side_effect = [execution(c) for c in GENERATION_CASES]
        rows = run_generation_suite(pipeline, run_id="run", top_k=5)
        summary = generation_summary(rows, provenance={"run_id": "run"})
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "artifacts") as tmp:
            path = Path(tmp)
            publish_generation_run(path, rows, summary)
            data = (path / "generation_run.jsonl").read_bytes()
            self.assertNotIn(b"\r", data)
            self.assertTrue(data.endswith(b"\n"))
            parsed = [json.loads(line) for line in data.decode("utf-8").splitlines()]
            self.assertEqual([r["case_id"] for r in parsed], [c.case_id for c in GENERATION_CASES])
            before = data
            rows[0]["response"]["timing_ms"]["total"] = float("nan")
            with self.assertRaises(ValueError): publish_generation_run(path, rows, summary)
            self.assertEqual((path / "generation_run.jsonl").read_bytes(), before)


if __name__ == "__main__": unittest.main()
