"""E5 provenance preserves E1-E4 and refuses frozen prompt/config/source drift."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.config import REPO_ROOT, load_config
from app.manifest import (ManifestConflictError, merge_e5_prompt_manifest,
                          runtime_profile_from_config, runtime_profile_sha256,
                          validate_e5_manifest)
from app.prompt import prompt_identity


def locked_manifest(config):
    profile = runtime_profile_from_config(config)
    return {"git_commit": "source", "git_dirty": False,
            "prompt_version": "not_implemented_e1",
            "dataset": {"hash": "preserve"}, "split": {"dev": ["a"]},
            "embedding": {"dim": 384}, "chunking": {"size": 220},
            "index": {"status": "built"},
            "retrieval": {"mode": config.retrieval.mode, "top_k": config.retrieval.top_k,
                          "max_top_k": config.retrieval.max_top_k},
            "freeze": {"runtime_profile_locked": True, "final_test_frozen": False},
            "llm": {"locked_profile": profile, "locked_profile_sha256": runtime_profile_sha256(profile),
                    "verification": {"old": "preserve"}, "future": [1, 2]}}


def check_payload(config):
    return {"status": "PASS", "source_git_commit": "source",
            "source_git_dirty": False, "prompt": dict(prompt_identity()),
            "profile_sha256": runtime_profile_sha256(runtime_profile_from_config(config)),
            "observed_runtime": {"runtime": config.llm.runtime,
                                 "runtime_version": config.llm.runtime_version,
                                 "model_id": config.llm.model_id,
                                 "runtime_model_digest": config.llm.runtime_model_digest + "a" * 52,
                                 "quantization": config.llm.quantization,
                                 "context_length": config.llm.context_length},
            "token_counter_check": {"native_prompt_tokens": 100, "completion_prompt_tokens": 100},
            "trace": {"model_id": config.llm.model_id, "input_token_count": 100},
            "response": {"model_id": config.llm.model_id, "status": "ANSWERED"}}


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.existing = locked_manifest(self.config)

    def merge(self, payload=None, existing=None, config=None, hash_override=None):
        root = REPO_ROOT / "artifacts"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            artifact = Path(tmp) / "rag_check.json"
            artifact.write_text(json.dumps(payload or check_payload(self.config)), encoding="utf-8")
            verification = {"artifact": artifact.relative_to(REPO_ROOT).as_posix(),
                            "artifact_sha256": hash_override or hashlib.sha256(artifact.read_bytes()).hexdigest()}
            with patch("app.manifest._git_state", return_value=("source", False)):
                return merge_e5_prompt_manifest(existing or self.existing,
                       config=config or self.config, verification=verification)

    def test_legacy_migration_preserves_every_prior_section_and_lock_field(self):
        original = deepcopy(self.existing)
        merged = self.merge()
        for key in ("dataset", "split", "embedding", "chunking", "index", "retrieval", "llm", "freeze"):
            self.assertEqual(merged[key], self.existing[key])
        self.assertEqual(self.existing, original)
        self.assertEqual(merged["prompt"], prompt_identity())
        self.assertEqual(merged["prompt_version"], "rag-grounded-json-v1")

    def test_locked_config_mismatch_or_unlocked_update_fails(self):
        for changes in ({"model_id": "different"}, {"runtime_profile_locked": False}):
            config = replace(self.config, llm=replace(self.config.llm, **changes))
            with self.assertRaises(ManifestConflictError):
                self.merge(config=config)

    def test_frozen_prompt_drift_legacy_or_config_drift_fails(self):
        frozen = deepcopy(self.existing)
        frozen["freeze"]["final_test_frozen"] = True
        with self.assertRaises(ManifestConflictError):
            self.merge(existing=frozen)
        frozen["prompt"] = dict(prompt_identity())
        frozen["prompt_version"] = prompt_identity()["version"]
        self.assertEqual(self.merge(existing=frozen)["freeze"], frozen["freeze"])
        frozen["prompt"]["sha256"] = "0" * 64
        with self.assertRaises(ManifestConflictError):
            self.merge(existing=frozen)
        frozen["prompt"] = dict(prompt_identity())
        config = replace(self.config, retrieval=replace(self.config.retrieval, top_k=4))
        with self.assertRaises(ManifestConflictError):
            self.merge(existing=frozen, config=config)

    def test_frozen_identity_requires_same_clean_source(self):
        frozen = deepcopy(self.existing)
        frozen.update(prompt=dict(prompt_identity()), prompt_version=prompt_identity()["version"])
        frozen["freeze"]["final_test_frozen"] = True
        for commit, dirty in (("source", True), ("different", False)):
            with patch("app.manifest._git_state", return_value=(commit, dirty)), self.assertRaises(ManifestConflictError):
                validate_e5_manifest(frozen, config=self.config)

    def test_verification_hash_prompt_runtime_and_counts_are_bound(self):
        with self.assertRaises(ManifestConflictError):
            self.merge(hash_override="0" * 64)
        for key, changes in (("prompt", {"sha256": "bad"}),
                             ("observed_runtime", {"model_id": "different"}),
                             ("token_counter_check", {"completion_prompt_tokens": 99}),
                             ("trace", {"input_token_count": 32768}),
                             ("response", {"status": "ERROR"})):
            payload = check_payload(self.config)
            payload[key].update(changes)
            with self.subTest(key=key), self.assertRaises(ManifestConflictError):
                self.merge(payload)

    def test_verification_source_git_state_must_match_merge_source(self):
        for key, value in (("source_git_commit", "different"),
                           ("source_git_dirty", True)):
            payload = check_payload(self.config)
            payload[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(
                    ManifestConflictError, "source git state"):
                self.merge(payload)


if __name__ == "__main__":
    unittest.main()
