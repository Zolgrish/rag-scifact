"""Only source-clean all-pass artifacts with matching E5 provenance can merge."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch, Mock

from app.config import REPO_ROOT, effective_config_identity, load_config
from app.data_audit import file_sha256
from app.evaluator import GENERATION_CASES, generation_summary, run_generation_suite, suite_identity, publish_generation_run
from app.fixtures import ATLAS_DOCUMENTS
from app.manifest import (ManifestConflictError, merge_e6_generation_manifest,
                          runtime_profile_from_config, runtime_profile_sha256)
from app.prompt import prompt_identity
from tests.test_e5_manifest import locked_manifest, check_payload
from tests.test_e6_grader import execution


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=REPO_ROOT / "artifacts")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config()
        self.existing = locked_manifest(self.config)
        self.existing.update(prompt=dict(prompt_identity()), prompt_version=prompt_identity()["version"])
        for key in ("chunking", "embedding", "index", "retrieval", "llm"):
            self.existing[key]["future_field"] = "preserve"
        e5 = self.root / "rag_check.json"
        e5.write_text(json.dumps(check_payload(self.config)))
        self.existing["rag_verification"] = {"artifact": str(e5), "artifact_sha256": file_sha256(e5)}
        self.fixture = self.root / "atlas.jsonl"
        self.fixture.write_bytes((REPO_ROOT / "data/fixtures/atlas.jsonl").read_bytes())
        self.bundle = self.root / "atlas"
        self.bundle.mkdir()
        index_manifest = self.bundle / "index_manifest.json"
        index_manifest.write_text("{}")
        self.provenance = {"run_id": "run", "source_git_commit": "source", "source_git_dirty": False,
            "config": {"artifact": str(REPO_ROOT / "config.yaml"), "sha256": file_sha256(REPO_ROOT / "config.yaml")},
            "effective_config": effective_config_identity(self.config),
            "retrieval": {k: getattr(self.config.retrieval, k) for k in ("mode", "top_k", "max_top_k")},
            "prompt": dict(prompt_identity()), "fixture_suite": suite_identity(),
            "runtime_profile_sha256": runtime_profile_sha256(runtime_profile_from_config(self.config)),
            "model_id": self.config.llm.model_id,
            "fixture_corpus": {"artifact": str(self.fixture), "sha256": file_sha256(self.fixture)},
            "fixture_index": {"bundle_path": str(self.bundle), "manifest": {"artifact": str(index_manifest), "sha256": file_sha256(index_manifest)}}}
        identity = prompt_identity()
        executions = [execution(c, request_id="run-" + c.case_id, model_id=self.config.llm.model_id,
                                prompt_version=identity["version"], prompt_sha256=identity["sha256"])
                      for c in GENERATION_CASES]
        pipeline = Mock()
        pipeline.ask.side_effect = executions
        self.rows = run_generation_suite(pipeline, run_id="run", top_k=5)
        self.bundle_object = Mock(chunks=tuple(
            SimpleNamespace(doc_id=r.doc_id, chunk_id=r.chunk_id, title=r.title, text=r.text)
            for r in executions[0].response.retrieved
        ))

    def merge(self, *, changes=None, manifest=None, state=("source", False), states=None, config=None):
        config = config or self.config
        summary = generation_summary(self.rows, provenance=self.provenance)
        summary.update(changes or {})
        publish_generation_run(self.root, self.rows, summary)
        verification = {key: {"artifact": str(self.root / name), "sha256": file_sha256(self.root / name)}
                        for key, name in (("generation_run", "generation_run.jsonl"), ("summary", "generation_summary.json"))}
        with patch("app.manifest.current_git_state", return_value=state, side_effect=states), \
             patch("app.manifest._git_state", return_value=state), \
             patch("app.indexer.load_bundle", return_value=self.bundle_object), \
             patch("app.fixtures.validate_atlas_bundle"):
            return merge_e6_generation_manifest(manifest or self.existing, config=config, verification=verification)

    def test_merge_preserves_every_e1_e5_section_and_all_lock_fields(self):
        before = deepcopy(self.existing)
        merged = self.merge()
        for key, value in before.items(): self.assertEqual(merged[key], value)
        self.assertEqual(self.existing, before)
        self.assertFalse(merged["freeze"]["final_test_frozen"])
        self.assertEqual(merged["generation_fixture"]["passed"], 6)

    def test_dirty_or_stale_source_and_failed_suite_rejected(self):
        for state in (("source", True), ("different", False)):
            with self.assertRaises(ManifestConflictError): self.merge(state=state)
        for changes in ({"status": "FAIL"}, {"passed": 5}, {"source_git_commit": "other"}, {"source_git_dirty": True}, {"prompt": {}}, {"fixture_suite": {}}):
            with self.subTest(changes=changes), self.assertRaises(ManifestConflictError): self.merge(changes=changes)

    def test_stale_e5_artifact_blocks_canonical_merge(self):
        artifact = Path(self.existing["rag_verification"]["artifact"])
        payload = json.loads(artifact.read_text())
        payload["source_git_commit"] = "old-e5-source"
        artifact.write_text(json.dumps(payload))
        self.existing["rag_verification"]["artifact_sha256"] = file_sha256(artifact)
        with self.assertRaisesRegex(ManifestConflictError, "refresh scripts.check_rag"):
            self.merge()

    def test_one_failed_case_row_cannot_override_failed_summary(self):
        self.rows[2]["grading"]["passed"] = False
        with self.assertRaises(ManifestConflictError): self.merge(changes={"status": "PASS", "passed": 6, "failed": 0})

    def test_malformed_or_tampered_passing_rows_are_independently_rejected(self):
        mutations = (
            lambda row: row.update(response=None, trace=None, grading={"passed": True, "checks": {}, "failure_codes": []}),
            lambda row: row.update(question="tampered question"),
            lambda row: row["grading"].update(passed=False),
            lambda row: row["trace"].update(context_ids=[]),
        )
        for mutate in mutations:
            original = deepcopy(self.rows)
            try:
                mutate(self.rows[0])
                with self.subTest(mutation=mutate), self.assertRaises(ManifestConflictError):
                    self.merge(changes={"status": "PASS", "passed": 6, "failed": 0})
            finally:
                self.rows = original

    def test_effective_config_mismatch_rejected_even_when_raw_config_hash_matches(self):
        changed = replace(self.config, seed=999)
        with self.assertRaises(ManifestConflictError):
            self.merge(config=changed)

    def test_source_change_during_artifact_verification_rejected(self):
        with self.assertRaisesRegex(ManifestConflictError, "Git state changed"):
            self.merge(states=[("source", False), ("different", False)])


if __name__ == "__main__": unittest.main()
