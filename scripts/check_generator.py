"""Check the configured local generator readiness and optional deterministic smoke."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from app.config import REPO_ROOT
from app.generator import (
    GeneratorConfigurationError,
    GeneratorInfrastructureError,
    build_generator,
    validate_runtime_metadata,
)
from app.manifest import (
    ManifestError,
    load_manifest,
    merge_e4_runtime_manifest,
    runtime_profile_from_config,
    runtime_profile_sha256,
    validate_runtime_profile_lock,
    write_manifest_atomic,
)
from scripts._common import ConfigurationError, bootstrap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Also run one deterministic local chat completion through the Generator.",
    )
    parser.add_argument(
        "--output-dir",
        help="Artifact directory; defaults to artifacts/<run_id>.",
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/manifest.json",
        help="Canonical project manifest path.",
    )
    manifest_actions = parser.add_mutually_exclusive_group()
    manifest_actions.add_argument(
        "--update-manifest",
        action="store_true",
        help="Refresh verification for the already-locked matching runtime profile.",
    )
    manifest_actions.add_argument(
        "--relock-runtime-profile",
        action="store_true",
        help=(
            "After an unlocked successful smoke, explicitly replace the manifest "
            "runtime profile. Forbidden after final_test_frozen=true."
        ),
    )
    return parser


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _repo_relative(path: Path, *, label: str) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise GeneratorConfigurationError(
            f"{label} must be inside the repository when binding manifest provenance"
        ) from exc


def main() -> int:
    args = build_parser().parse_args()
    try:
        config, logger, run_id = bootstrap(args.config)
        mutates_manifest = args.update_manifest or args.relock_runtime_profile
        if mutates_manifest and not args.smoke:
            raise GeneratorConfigurationError(
                "manifest runtime-profile updates require --smoke"
            )
        if args.update_manifest and not config.llm.runtime_profile_locked:
            raise GeneratorConfigurationError(
                "--update-manifest requires llm.runtime_profile_locked=true"
            )
        if args.relock_runtime_profile and config.llm.runtime_profile_locked:
            raise GeneratorConfigurationError(
                "--relock-runtime-profile requires llm.runtime_profile_locked=false "
                "while verifying the replacement profile"
            )

        manifest_path = _repo_path(args.manifest)
        manifest = load_manifest(manifest_path)
        if config.llm.runtime_profile_locked:
            validate_runtime_profile_lock(manifest, config)
        if args.relock_runtime_profile:
            freeze = manifest.get("freeze")
            if isinstance(freeze, dict) and freeze.get("final_test_frozen") is True:
                raise GeneratorConfigurationError(
                    "Cannot relock runtime profile after final_test_frozen=true"
                )

        output_dir = (
            _repo_path(args.output_dir)
            if args.output_dir
            else config.paths.artifact_dir / run_id
        )
        artifact_path = output_dir / "generator_check.json"
        artifact_relative: str | None = None
        if mutates_manifest:
            artifact_relative = _repo_relative(
                artifact_path,
                label="--output-dir artifact",
            )

        generator = build_generator(config.llm)
        logger.info(
            "E4 generator readiness started runtime=%s model=%s endpoint=%s",
            config.llm.runtime,
            config.llm.model_id,
            config.llm.base_url,
        )
        readiness = generator.check_readiness()
        observed_runtime = generator.inspect_runtime_metadata()
        validate_runtime_metadata(config.llm, observed_runtime)
        profile = runtime_profile_from_config(config)
        profile_sha256 = runtime_profile_sha256(profile)
        payload: dict[str, object] = {
            "status": "PASS",
            "run_id": run_id,
            "ready": readiness.ready,
            "profile": profile,
            "profile_sha256": profile_sha256,
            "observed_runtime": asdict(observed_runtime),
            "config_runtime_profile_locked": config.llm.runtime_profile_locked,
            "readiness_ms": readiness.response_ms,
            "smoke": None,
        }
        if args.smoke:
            result = generator.generate(
                [{"role": "user", "content": "Reply with exactly: OK"}]
            )
            payload["smoke"] = {
                "text": result.text,
                "model_id": result.model_id,
                "usage": dict(result.usage) if result.usage is not None else None,
                "generation_ms": result.generation_ms,
                "expected_exact_text": "OK",
                "exact_text_match": result.text.strip() == "OK",
            }
            payload["seed_parameter_accepted"] = True
            if result.text.strip() != "OK":
                raise GeneratorInfrastructureError(
                    "deterministic smoke completion did not return the expected text"
                )

        output_dir.mkdir(parents=True, exist_ok=True)
        if mutates_manifest:
            payload["manifest"] = str(manifest_path)
            payload["manifest_action"] = (
                "relock" if args.relock_runtime_profile else "refresh"
            )
        write_manifest_atomic(artifact_path, payload)

        if mutates_manifest:
            assert artifact_relative is not None
            artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            verification = {
                "status": "PASS",
                "readiness": "PASS",
                "smoke": "PASS",
                "seed_parameter_accepted": True,
                "profile": profile,
                "profile_sha256": profile_sha256,
                "observed_runtime": asdict(observed_runtime),
                "artifact": artifact_relative,
                "artifact_sha256": artifact_sha256,
            }
            merged = merge_e4_runtime_manifest(
                manifest,
                config=config,
                verification=verification,
                relock=args.relock_runtime_profile,
            )
            write_manifest_atomic(manifest_path, merged)
            write_manifest_atomic(output_dir / "runtime_manifest.json", merged)

        logger.info("E4 generator readiness PASS artifact=%s", artifact_path)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    except (ConfigurationError, GeneratorConfigurationError, ManifestError) as exc:
        if "logger" in locals():
            logger.exception("E4 generator configuration failed")
        print(
            json.dumps(
                {
                    "status": "ERROR",
                    "error": "configuration",
                    "message": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 2
    except GeneratorInfrastructureError as exc:
        if "logger" in locals():
            logger.exception("E4 generator infrastructure check failed")
        print(
            json.dumps(
                {"status": "ERROR", "error": exc.to_dict()},
                ensure_ascii=False,
            )
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
