"""Raw-run reconstruction must defeat forged summary scores and stale provenance."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from app.benchmark_manifest import artifact_record, verify_dev_artifact
from app.manifest import merge_e7_retrieval_manifest
from app.retrieval_evaluator import RetrievalEvaluationError
from app.retrieval_evaluator import QueryOutcome, aggregate, failure_candidates
from tests.e7_support import ArtifactFixture


class ManifestTests(unittest.TestCase):
    def setUp(self): self.fixture = ArtifactFixture(self)

    def verify(self):
        f = self.fixture
        with f.patches():
            return verify_dev_artifact(f.manifest, config=f.config, verification=f.verification)

    def change_summary(self, modify):
        f = self.fixture
        path = f.directory / "metrics.json"
        summary = json.loads(path.read_text())
        modify(summary)
        path.write_text(json.dumps(summary), encoding="utf-8")
        f.verification = {"metrics": artifact_record(path)}

    def test_merge_recomputes_and_preserves_every_prior_section(self):
        f = self.fixture
        before = deepcopy(f.manifest)
        with f.patches():
            merged = merge_e7_retrieval_manifest(f.manifest, config=f.config, verification=f.verification)
        self.assertEqual({k: merged[k] for k in before}, before)
        self.assertEqual(f.manifest, before)
        self.assertEqual(merged["retrieval_benchmark"]["denominator"], 100)
        self.assertEqual(set(merged["retrieval_benchmark"]["metrics"].values()), {1.})

    def test_raw_query_failure_recomputed_as_zero_with_denominator_100(self):
        f = self.fixture
        f.outcomes = (QueryOutcome("0", (), 2., {"type": "RuntimeError", "message": "unexpected"}),) + f.outcomes[1:]
        f.summary.update(run_id="with-error", failure_candidate_count=1, **aggregate(f.outcomes, f.gold))
        f.publish()
        result = self.verify()
        self.assertEqual(result["failure_count"], 1)
        self.assertEqual(result["denominator"], 100)
        self.assertEqual(set(result["metrics"].values()), {.99})

    def test_rejects_forged_metrics_despite_fresh_artifact_hash(self):
        self.change_summary(lambda s: s["metrics"].update({"recall@5": .25}))
        with self.assertRaisesRegex(RetrievalEvaluationError, "recomputation"): self.verify()

    def test_rejects_tampered_raw_hash_and_tampered_raw_with_updated_hash(self):
        f = self.fixture
        path = f.directory / "retrieval_run.jsonl"
        data = path.read_text()
        path.write_text(data.replace('"rank": 1', '"rank": 2', 1), encoding="utf-8")
        with self.assertRaisesRegex(RetrievalEvaluationError, "SHA256"): self.verify()
        self.change_summary(lambda s: s.update(retrieval_run=artifact_record(path)))
        with self.assertRaises(RetrievalEvaluationError): self.verify()

    def test_raw_result_change_recomputes_metrics_instead_of_trusting_summary(self):
        f = self.fixture
        path = f.directory / "retrieval_run.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["doc_id"], rows[1]["chunk_id"] = "b", "b:0-1"
        rows[2]["doc_id"], rows[2]["chunk_id"] = "a", "a:0-1"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        self.change_summary(lambda s: s.update(retrieval_run=artifact_record(path)))
        with self.assertRaisesRegex(RetrievalEvaluationError, "recomputation"): self.verify()

    def test_missing_query_and_extra_query_fail_even_with_fresh_hash(self):
        f = self.fixture
        path = f.directory / "retrieval_run.jsonl"
        rows = path.read_text().splitlines()
        path.write_text("\n".join(rows[3:]) + "\n", encoding="utf-8")
        self.change_summary(lambda s: s.update(retrieval_run=artifact_record(path)))
        with self.assertRaisesRegex(RetrievalEvaluationError, "order"): self.verify()

    def test_summary_identity_denominator_errors_and_analysis_tampering_fail(self):
        original = deepcopy(self.fixture.completed)
        for key, value in (("denominator", 99), ("failure_count", 1), ("source_git_commit", "stale"),
                           ("source_git_dirty", True), ("effective_config", {}), ("index", {}),
                           ("split_identity", {}), ("metric_policy", {}), ("failure_candidate_count", 1),
                           ("schema_version", True), ("failure_count", False), ("runtime", {}),
                           ("metrics", {key: True for key in original["metrics"]})):
            with self.subTest(key=key):
                self.change_summary(lambda s, k=key, v=value: (s.clear(), s.update(original), s.update({k: v})))
                with self.assertRaises(RetrievalEvaluationError): self.verify()
        self.change_summary(lambda s: (s.clear(), s.update(original)))
        path = self.fixture.directory / "failure_analysis.jsonl"
        path.write_text('{"query_id":"fake"}\n', encoding="utf-8")
        self.change_summary(lambda s: s.update(failure_analysis=artifact_record(path)))
        with self.assertRaisesRegex(RetrievalEvaluationError, "Failure analysis"): self.verify()

    def test_dirty_source_outside_path_and_source_change_fail_closed(self):
        f = self.fixture
        with f.patches(), patch("app.manifest.current_git_state", return_value=(f.commit, True)):
            with self.assertRaisesRegex(RetrievalEvaluationError, "clean"): verify_dev_artifact(f.manifest, config=f.config, verification=f.verification)
        with f.patches(states=[(f.commit, False), ("different", False)]):
            with self.assertRaisesRegex(RetrievalEvaluationError, "Source changed"):
                verify_dev_artifact(f.manifest, config=f.config, verification=f.verification)
        with f.patches(), self.assertRaises(RetrievalEvaluationError):
            verify_dev_artifact(f.manifest, config=f.config, verification={"metrics": {"artifact": "../outside", "sha256": "bad"}})


if __name__ == "__main__": unittest.main()
