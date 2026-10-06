"""E4 generator readiness CLI tests without live Ollama."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from app.generator import (
    GeneratorConnectionError,
    GeneratorReadiness,
    GeneratorResult,
    RuntimeMetadata,
)
from scripts import check_generator


MODEL = "ministral-3:3b-instruct-2512-q8_0"


def fake_config(root: Path, *, locked: bool = False):
    return SimpleNamespace(
        llm=SimpleNamespace(
            provider="local_openai_compatible",
            runtime="ollama",
            runtime_version="0.24.0",
            reference_model_id="mistralai/Ministral-3-3B-Instruct-2512-BF16",
            model_id=MODEL,
            revision="",
            runtime_model_digest="c269e5748d11",
            base_url="http://127.0.0.1:11434/v1",
            context_length=32768,
            temperature=0.0,
            max_new_tokens=512,
            seed=42,
            dtype="",
            quantization="Q8_0",
            device="NVIDIA GeForce RTX 5070",
            connect_timeout_s=10.0,
            read_timeout_s=120.0,
            runtime_profile_locked=locked,
        ),
        paths=SimpleNamespace(artifact_dir=root),
    )


def observed_metadata() -> RuntimeMetadata:
    return RuntimeMetadata(
        runtime="ollama",
        runtime_version="0.24.0",
        model_id=MODEL,
        runtime_model_digest="c269e5748d11" + "a" * 52,
        quantization="Q8_0",
        context_length=32768,
        size_vram_bytes=1024,
    )


def ready_generator() -> Mock:
    generator = Mock()
    generator.check_readiness.return_value = GeneratorReadiness(
        ready=True,
        model_id=MODEL,
        runtime="ollama",
        response_ms=1.25,
    )
    generator.inspect_runtime_metadata.return_value = observed_metadata()
    generator.generate.return_value = GeneratorResult(
        text="OK",
        model_id=MODEL,
        usage={"total_tokens": 2},
        generation_ms=2.0,
    )
    return generator


class E4GeneratorCliTests(unittest.TestCase):
    def test_unlocked_readiness_only_writes_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generator = ready_generator()
            with (
                patch(
                    "scripts.check_generator.bootstrap",
                    return_value=(fake_config(root, locked=False), Mock(), "run-1"),
                ),
                patch("scripts.check_generator.build_generator", return_value=generator),
                patch("scripts.check_generator.load_manifest", return_value={}),
                patch(
                    "sys.argv",
                    ["check_generator", "--output-dir", str(root / "out")],
                ),
            ):
                code = check_generator.main()

            self.assertEqual(code, 0)
            self.assertTrue((root / "out" / "generator_check.json").is_file())
            generator.generate.assert_not_called()
            generator.inspect_runtime_metadata.assert_called_once_with()

    def test_manifest_actions_require_smoke_before_network(self) -> None:
        for flag in ("--update-manifest", "--relock-runtime-profile"):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with (
                    patch(
                        "scripts.check_generator.bootstrap",
                        return_value=(fake_config(root), Mock(), "run-2"),
                    ),
                    patch("scripts.check_generator.build_generator") as build,
                    patch("sys.argv", ["check_generator", flag]),
                ):
                    code = check_generator.main()
                self.assertEqual(code, 2)
                build.assert_not_called()

    def test_update_manifest_requires_locked_config_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch(
                    "scripts.check_generator.bootstrap",
                    return_value=(fake_config(root, locked=False), Mock(), "run-3"),
                ),
                patch("scripts.check_generator.build_generator") as build,
                patch(
                    "sys.argv",
                    ["check_generator", "--smoke", "--update-manifest"],
                ),
            ):
                code = check_generator.main()
        self.assertEqual(code, 2)
        build.assert_not_called()

    def test_relock_requires_unlocked_config_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch(
                    "scripts.check_generator.bootstrap",
                    return_value=(fake_config(root, locked=True), Mock(), "run-4"),
                ),
                patch("scripts.check_generator.build_generator") as build,
                patch(
                    "sys.argv",
                    ["check_generator", "--smoke", "--relock-runtime-profile"],
                ),
            ):
                code = check_generator.main()
        self.assertEqual(code, 2)
        build.assert_not_called()

    def test_locked_mode_validates_manifest_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch(
                    "scripts.check_generator.bootstrap",
                    return_value=(fake_config(root, locked=True), Mock(), "run-5"),
                ),
                patch("scripts.check_generator.load_manifest", return_value={"freeze": {}}),
                patch(
                    "scripts.check_generator.validate_runtime_profile_lock",
                    side_effect=check_generator.ManifestError("profile mismatch"),
                ) as validate,
                patch("scripts.check_generator.build_generator") as build,
                patch("sys.argv", ["check_generator"]),
            ):
                code = check_generator.main()
        self.assertEqual(code, 2)
        validate.assert_called_once()
        build.assert_not_called()

    def test_external_output_dir_is_rejected_before_relock_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            external = Path(tmp) / "outside"
            with (
                patch(
                    "scripts.check_generator.bootstrap",
                    return_value=(fake_config(Path(tmp), locked=False), Mock(), "run-6"),
                ),
                patch("scripts.check_generator.load_manifest", return_value={}),
                patch("scripts.check_generator.build_generator") as build,
                patch(
                    "sys.argv",
                    [
                        "check_generator",
                        "--smoke",
                        "--relock-runtime-profile",
                        "--output-dir",
                        str(external),
                    ],
                ),
            ):
                code = check_generator.main()
        self.assertEqual(code, 2)
        build.assert_not_called()

    def test_infrastructure_failure_has_distinct_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generator = ready_generator()
            generator.check_readiness.side_effect = GeneratorConnectionError("refused")
            with (
                patch(
                    "scripts.check_generator.bootstrap",
                    return_value=(fake_config(root, locked=False), Mock(), "run-7"),
                ),
                patch("scripts.check_generator.build_generator", return_value=generator),
                patch("scripts.check_generator.load_manifest", return_value={}),
                patch("sys.argv", ["check_generator"]),
            ):
                code = check_generator.main()
        self.assertEqual(code, 3)

    def test_smoke_uses_same_generator_and_runtime_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generator = ready_generator()
            with (
                patch(
                    "scripts.check_generator.bootstrap",
                    return_value=(fake_config(root, locked=False), Mock(), "run-8"),
                ),
                patch("scripts.check_generator.build_generator", return_value=generator),
                patch("scripts.check_generator.load_manifest", return_value={}),
                patch(
                    "sys.argv",
                    [
                        "check_generator",
                        "--smoke",
                        "--output-dir",
                        str(root / "out"),
                    ],
                ),
            ):
                code = check_generator.main()

            self.assertEqual(code, 0)
            generator.generate.assert_called_once_with(
                [{"role": "user", "content": "Reply with exactly: OK"}]
            )
            generator.inspect_runtime_metadata.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
