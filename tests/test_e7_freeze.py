"""Pre-test sealing and drift rejection use dev artifacts only, without inference."""
from copy import deepcopy
from dataclasses import replace
import unittest
import json
from unittest.mock import patch

from app.benchmark_manifest import artifact_record, load_evaluation_data, verify_dev_artifact
from app.manifest import (ManifestError, freeze_final_test_manifest, validate_final_test_gate)
from app.retrieval_evaluator import RetrievalEvaluationError
from tests.e7_support import ArtifactFixture

ERRORS = (ManifestError, RetrievalEvaluationError)


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ArtifactFixture(self)
        f = self.fixture
        with f.patches():
            f.manifest["retrieval_benchmark"] = verify_dev_artifact(f.manifest, config=f.config, verification=f.verification)

    def frozen(self):
        f = self.fixture
        with f.patches(): return freeze_final_test_manifest(f.manifest, config=f.config)

    def test_seal_preserves_e1_e6_and_never_reads_test_qrels(self):
        f = self.fixture
        with f.patches(), patch("app.benchmark_manifest.load_qrels") as qrels:
            sealed = freeze_final_test_manifest(f.manifest, config=f.config)
            validate_final_test_gate(sealed, config=f.config)
        qrels.assert_not_called()
        self.assertFalse(f.manifest["freeze"]["final_test_frozen"])
        for key in f.manifest:
            if key != "freeze": self.assertEqual(sealed[key], f.manifest[key])
        self.assertTrue(sealed["freeze"]["runtime_profile_locked"])
        self.assertEqual(sealed["freeze"]["frozen_git_commit"], f.commit)
        self.assertEqual(sealed["freeze"]["identity"]["e7_dev_benchmark"]["denominator"], 100)
        with f.patches(), self.assertRaisesRegex(RetrievalEvaluationError, "one-way"):
            freeze_final_test_manifest(sealed, config=f.config)

    def test_dirty_and_different_source_rejected(self):
        f = self.fixture
        for state in ((f.commit, True), ("other-commit", False), ("", False)):
            with self.subTest(state=state), f.patches(), patch("app.manifest.current_git_state", return_value=state):
                with self.assertRaises(ERRORS): freeze_final_test_manifest(f.manifest, config=f.config)
                with self.assertRaises(ERRORS): validate_final_test_gate({**f.manifest, "freeze": {"final_test_frozen": True}}, config=f.config)

    def test_missing_and_stale_e5_e6_e7_rejected(self):
        f = self.fixture
        for section in ("rag_verification", "generation_fixture", "retrieval_benchmark"):
            for value in ({}, {"artifact": "missing-stale.json", "artifact_sha256": "bad"}):
                manifest = deepcopy(f.manifest)
                manifest[section] = value
                with self.subTest(section=section), f.patches(), self.assertRaises(ERRORS):
                    freeze_final_test_manifest(manifest, config=f.config)

    def test_validly_hashed_artifacts_from_stale_e5_and_e6_sources_rejected(self):
        f = self.fixture
        from app.config import REPO_ROOT
        for section, field in (("rag_verification", "artifact"), ("generation_fixture", "summary")):
            manifest = deepcopy(f.manifest)
            record = manifest[section] if field == "artifact" else manifest[section][field]
            payload = json.loads((REPO_ROOT / record["artifact"]).read_text())
            payload["source_git_commit"] = "stale-clean-source"
            path = f.root / (section + ".json")
            path.write_text(json.dumps(payload), encoding="utf-8")
            fresh = artifact_record(path)
            if field == "artifact":
                manifest[section] = {"artifact": fresh["artifact"], "artifact_sha256": fresh["sha256"]}
            else:
                manifest[section][field] = fresh
            with self.subTest(section=section), f.patches(), self.assertRaises(ERRORS):
                freeze_final_test_manifest(manifest, config=f.config)

    def test_all_config_prompt_model_retrieval_index_dataset_drifts_rejected(self):
        f = self.fixture
        sealed = self.frozen()
        changed_configs = (
            replace(f.config, name="changed"),
            replace(f.config, llm=replace(f.config.llm, model_id="different")),
            replace(f.config, llm=replace(f.config.llm, runtime_version="different")),
            replace(f.config, llm=replace(f.config.llm, runtime_model_digest="different")),
            replace(f.config, retrieval=replace(f.config.retrieval, top_k=4)),
        )
        for config in changed_configs:
            with self.subTest(config=config), f.patches(), self.assertRaises(ERRORS):
                validate_final_test_gate(sealed, config=config)
        for section in ("prompt", "dataset", "split", "generation_fixture", "retrieval_benchmark"):
            altered = deepcopy(sealed)
            altered[section] = {}
            with self.subTest(section=section), f.patches(), self.assertRaises(ERRORS):
                validate_final_test_gate(altered, config=f.config)
        f.index = {"different_index": "drift"}
        with f.patches(), self.assertRaises(ERRORS): validate_final_test_gate(sealed, config=f.config)

    def test_unfrozen_test_loader_is_gated_before_any_qrels_access(self):
        f = self.fixture
        with patch("app.benchmark_manifest.load_qrels") as qrels, self.assertRaisesRegex(RetrievalEvaluationError, "gated"):
            load_evaluation_data(f.config, f.manifest, split="test")
        qrels.assert_not_called()

    def test_altered_seal_hash_or_source_cannot_bypass_gate(self):
        f = self.fixture
        for field, value in (("identity_sha256", "0" * 64), ("frozen_git_commit", "other"), ("git_dirty", True)):
            sealed = self.frozen()
            sealed["freeze"][field] = value
            with self.subTest(field=field), f.patches(), self.assertRaises(ERRORS):
                validate_final_test_gate(sealed, config=f.config)


if __name__ == "__main__": unittest.main()
