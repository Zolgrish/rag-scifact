"""E4 manifest-authoritative runtime profile lock tests."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.config import REPO_ROOT, load_config
from app.manifest import (
    ManifestConflictError,
    _validate_e4_verification,
    merge_e4_runtime_manifest,
    runtime_profile_from_config,
    runtime_profile_sha256,
    validate_runtime_profile_lock,
)


class E4ManifestTests(unittest.TestCase):
    def locked_config(self):
        base = load_config()
        return replace(base, llm=replace(base.llm, runtime_profile_locked=True))

    def unlocked_config(self, **changes):
        base = load_config()
        return replace(
            base,
            llm=replace(base.llm, runtime_profile_locked=False, **changes),
        )

    def locked_manifest(self, config):
        profile = runtime_profile_from_config(config)
        return {
            "schema_version": 1,
            "git_commit": "old",
            "git_dirty": False,
            "freeze": {
                "final_test_frozen": False,
                "runtime_profile_locked": True,
            },
            "llm": {
                "locked_profile": profile,
                "locked_profile_sha256": runtime_profile_sha256(profile),
                "future_provenance": "preserve-me",
            },
            "dataset": {"corpus_count": 5183},
            "index": {"status": "built", "ntotal": 10359},
        }

    def make_verification(self, config, directory: Path):
        profile = runtime_profile_from_config(config)
        profile_hash = runtime_profile_sha256(profile)
        digest = config.llm.runtime_model_digest
        full_digest = digest if len(digest) >= 64 else digest + "a" * (64 - len(digest))
        observed = {
            "runtime": config.llm.runtime,
            "runtime_version": config.llm.runtime_version,
            "model_id": config.llm.model_id,
            "runtime_model_digest": full_digest,
            "quantization": config.llm.quantization,
            "context_length": config.llm.context_length,
            "size_vram_bytes": 1024,
        }
        artifact = directory / "generator_check.json"
        payload = {
            "status": "PASS",
            "ready": True,
            "profile": profile,
            "profile_sha256": profile_hash,
            "observed_runtime": observed,
            "config_runtime_profile_locked": config.llm.runtime_profile_locked,
            "seed_parameter_accepted": True,
            "smoke": {
                "text": "OK",
                "model_id": config.llm.model_id,
                "exact_text_match": True,
            },
        }
        artifact.write_text(json.dumps(payload), encoding="utf-8")
        digest_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        relative = artifact.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        verification = {
            "status": "PASS",
            "readiness": "PASS",
            "smoke": "PASS",
            "seed_parameter_accepted": True,
            "profile": profile,
            "profile_sha256": profile_hash,
            "observed_runtime": observed,
            "artifact": relative,
            "artifact_sha256": digest_sha,
        }
        return verification, artifact

    def test_profile_hash_is_canonical_and_excludes_lock_flag(self) -> None:
        locked = self.locked_config()
        unlocked = replace(
            locked,
            llm=replace(locked.llm, runtime_profile_locked=False),
        )
        profile = runtime_profile_from_config(locked)
        reversed_profile = dict(reversed(list(profile.items())))
        self.assertEqual(
            runtime_profile_sha256(profile),
            runtime_profile_sha256(reversed_profile),
        )
        self.assertEqual(profile, runtime_profile_from_config(unlocked))

    def test_matching_manifest_lock_passes_and_drift_fails_only_when_enforced(self) -> None:
        config = self.locked_config()
        manifest = self.locked_manifest(config)
        expected_hash = manifest["llm"]["locked_profile_sha256"]
        self.assertEqual(validate_runtime_profile_lock(manifest, config), expected_hash)

        changed_locked = replace(
            config,
            llm=replace(config.llm, model_id="future-model:latest"),
        )
        with self.assertRaisesRegex(ManifestConflictError, "Differing fields: model_id"):
            validate_runtime_profile_lock(manifest, changed_locked)

        changed_unlocked = replace(
            changed_locked,
            llm=replace(changed_locked.llm, runtime_profile_locked=False),
        )
        self.assertIsNone(validate_runtime_profile_lock(manifest, changed_unlocked))

    def test_lock_true_without_manifest_profile_fails_closed(self) -> None:
        with self.assertRaisesRegex(ManifestConflictError, "no locked runtime profile"):
            validate_runtime_profile_lock({}, self.locked_config())

    def test_legacy_locked_manifest_is_accepted_for_migration(self) -> None:
        config = self.locked_config()
        profile = runtime_profile_from_config(config)
        legacy_llm = {key: value for key, value in profile.items() if key != "schema_version"}
        manifest = {
            "freeze": {"runtime_profile_locked": True, "final_test_frozen": False},
            "llm": legacy_llm,
        }
        self.assertEqual(
            validate_runtime_profile_lock(manifest, config),
            runtime_profile_sha256(profile),
        )

    def test_matching_locked_profile_can_refresh_verification(self) -> None:
        config = self.locked_config()
        existing = self.locked_manifest(config)
        artifact_root = REPO_ROOT / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifact_root) as tmp:
            verification, _artifact = self.make_verification(config, Path(tmp))
            with (
                patch("app.manifest._git_state", return_value=("e4commit", True)),
                patch(
                    "app.manifest._runtime_hardware",
                    return_value=(
                        {"python_version": "3.11.9", "cuda_available": True},
                        {"gpu_name": "NVIDIA GeForce RTX 5070"},
                    ),
                ),
            ):
                merged = merge_e4_runtime_manifest(
                    existing,
                    config=config,
                    verification=verification,
                )
        self.assertEqual(merged["dataset"], existing["dataset"])
        self.assertEqual(merged["index"], existing["index"])
        self.assertEqual(merged["llm"]["future_provenance"], "preserve-me")
        self.assertEqual(
            merged["llm"]["locked_profile_sha256"],
            runtime_profile_sha256(runtime_profile_from_config(config)),
        )
        self.assertTrue(merged["freeze"]["runtime_profile_locked"])
        self.assertFalse(merged["freeze"]["final_test_frozen"])

    def test_explicit_relock_replaces_profile_and_preserves_history(self) -> None:
        old_config = self.locked_config()
        existing = self.locked_manifest(old_config)
        new_config = self.unlocked_config(
            model_id="future-model:latest",
            runtime_model_digest="abcdef123456",
            quantization="Q4_K_M",
        )
        artifact_root = REPO_ROOT / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifact_root) as tmp:
            verification, _artifact = self.make_verification(new_config, Path(tmp))
            with (
                patch("app.manifest._git_state", return_value=("newcommit", True)),
                patch(
                    "app.manifest._runtime_hardware",
                    return_value=(
                        {"python_version": "3.11.9"},
                        {"gpu_name": "NVIDIA GeForce RTX 5070"},
                    ),
                ),
            ):
                merged = merge_e4_runtime_manifest(
                    existing,
                    config=new_config,
                    verification=verification,
                    relock=True,
                )
        new_profile = runtime_profile_from_config(new_config)
        self.assertEqual(merged["llm"]["locked_profile"], new_profile)
        self.assertEqual(
            merged["llm"]["locked_profile_sha256"],
            runtime_profile_sha256(new_profile),
        )
        self.assertTrue(merged["freeze"]["runtime_profile_locked"])
        self.assertEqual(len(merged["llm"]["profile_history"]), 1)
        self.assertEqual(
            merged["llm"]["profile_history"][0]["profile_sha256"],
            existing["llm"]["locked_profile_sha256"],
        )

    def test_relock_requires_unlocked_config_and_is_blocked_after_final_freeze(self) -> None:
        config = self.locked_config()
        existing = self.locked_manifest(config)
        artifact_root = REPO_ROOT / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifact_root) as tmp:
            verification, _ = self.make_verification(config, Path(tmp))
            with self.assertRaisesRegex(ManifestConflictError, "requires runtime_profile_locked=false"):
                merge_e4_runtime_manifest(
                    existing,
                    config=config,
                    verification=verification,
                    relock=True,
                )

        unlocked = self.unlocked_config()
        frozen = self.locked_manifest(config)
        frozen["freeze"]["final_test_frozen"] = True
        with tempfile.TemporaryDirectory(dir=artifact_root) as tmp:
            verification, _ = self.make_verification(unlocked, Path(tmp))
            with self.assertRaisesRegex(ManifestConflictError, "final_test_frozen=true"):
                merge_e4_runtime_manifest(
                    frozen,
                    config=unlocked,
                    verification=verification,
                    relock=True,
                )

    def test_ordinary_refresh_cannot_replace_profile(self) -> None:
        old_config = self.locked_config()
        existing = self.locked_manifest(old_config)
        changed = replace(
            old_config,
            llm=replace(old_config.llm, model_id="future-model:latest"),
        )
        artifact_root = REPO_ROOT / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifact_root) as tmp:
            verification, _ = self.make_verification(changed, Path(tmp))
            with self.assertRaisesRegex(ManifestConflictError, "does not match the locked"):
                merge_e4_runtime_manifest(
                    existing,
                    config=changed,
                    verification=verification,
                )

    def test_unlocked_profile_requires_explicit_relock_for_manifest_change(self) -> None:
        config = self.unlocked_config()
        existing = self.locked_manifest(self.locked_config())
        artifact_root = REPO_ROOT / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifact_root) as tmp:
            verification, _ = self.make_verification(config, Path(tmp))
            with self.assertRaisesRegex(ManifestConflictError, "use --relock-runtime-profile"):
                merge_e4_runtime_manifest(
                    existing,
                    config=config,
                    verification=verification,
                )

    def test_verification_is_bound_to_profile_runtime_and_artifact_hash(self) -> None:
        config = self.locked_config()
        artifact_root = REPO_ROOT / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifact_root) as tmp:
            verification, artifact = self.make_verification(config, Path(tmp))
            _validate_e4_verification(config, verification)

            changed_profile = dict(verification)
            changed_profile["profile_sha256"] = "0" * 64
            with self.assertRaises(ManifestConflictError):
                _validate_e4_verification(config, changed_profile)

            changed_hash = dict(verification)
            changed_hash["artifact_sha256"] = "0" * 64
            with self.assertRaises(ManifestConflictError):
                _validate_e4_verification(config, changed_hash)

            payload = json.loads(artifact.read_text(encoding="utf-8"))
            payload["smoke"]["exact_text_match"] = False
            artifact.write_text(json.dumps(payload), encoding="utf-8")
            tampered = dict(verification)
            tampered["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
            with self.assertRaises(ManifestConflictError):
                _validate_e4_verification(config, tampered)


if __name__ == "__main__":
    unittest.main()
