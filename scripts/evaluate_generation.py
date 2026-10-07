"""Evaluate F01-F06 generation behavior (implemented in E6)."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import sys

from app.config import REPO_ROOT, effective_config_identity
from app.data_audit import file_sha256
from app.evaluator import (generation_summary, publish_generation_run,
                           run_generation_suite, suite_identity)
from app.fixtures import (build_atlas_bundle, bundle_hashes, guard_fixture_path,
                          load_atlas, validate_atlas_bundle, FixtureError)
from app.generator import validate_runtime_metadata
from app.indexer import load_bundle
from app.manifest import (current_git_state, load_manifest, merge_e6_generation_manifest,
                          runtime_profile_from_config, runtime_profile_sha256, write_manifest_atomic)
from app.prompt import prompt_identity
from app.rag import build_rag_pipeline_from_bundle
from scripts._common import bootstrap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        default="data/fixtures/atlas.jsonl",
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-folder", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--update-manifest", action="store_true")
    return parser


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        artifact = resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        artifact = str(resolved)
    return {"artifact": artifact, "sha256": file_sha256(path)}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = None
    try:
        config, logger, run_id = bootstrap(args.config)
        source_commit, source_dirty = current_git_state()
        config_record = _record(_path(args.config))
        fixture = _path(args.fixture)
        corpus = load_atlas(fixture)
        corpus_record = _record(fixture)
        canonical = config.paths.index_dir / "scifact"
        before = bundle_hashes(canonical)
        target = guard_fixture_path(_path(args.bundle) if args.bundle else config.paths.index_dir / "fixtures" / "atlas", canonical)
        if args.build_index:
            build_atlas_bundle(config, fixture_path=fixture, bundle_path=target, run_id=run_id,
                               batch_size=args.batch_size, cache_folder=args.cache_folder,
                               local_files_only=args.local_files_only, device=args.device)
        bundle = load_bundle(target, expected_corpus_sha256=corpus_record["sha256"])
        validate_atlas_bundle(bundle, corpus)
        pipeline = build_rag_pipeline_from_bundle(config, bundle_path=target,
                    expected_corpus_sha256=corpus_record["sha256"], logger=logger,
                    cache_folder=args.cache_folder, device=args.device)
        # Setup probes do not enter any fixture generation context.
        pipeline.generator.check_readiness()
        observed = pipeline.generator.inspect_runtime_metadata()
        validate_runtime_metadata(config.llm, observed)
        provenance = {"run_id": run_id, "source_git_commit": source_commit, "source_git_dirty": source_dirty,
                      "config": config_record, "effective_config": effective_config_identity(config),
                      "retrieval": {"mode": config.retrieval.mode,
                      "top_k": config.retrieval.top_k, "max_top_k": config.retrieval.max_top_k},
                      "prompt": dict(prompt_identity()), "fixture_suite": suite_identity(),
                      "runtime_profile_sha256": runtime_profile_sha256(runtime_profile_from_config(config)),
                      "model_id": config.llm.model_id, "observed_runtime": asdict(observed),
                      "fixture_corpus": {**corpus_record, "name": "atlas_fixture", "documents": 6},
                      "fixture_index": {"bundle_path": str(target), "manifest": _record(target / "index_manifest.json"),
                                        "files": bundle_hashes(target)}}
        rows = run_generation_suite(pipeline, run_id=run_id, top_k=config.retrieval.top_k)
        after = bundle_hashes(canonical)
        if after != before:
            raise FixtureError("SciFact bundle changed during fixture evaluation")
        provenance["scifact_integrity"] = {"before": before, "after": after, "unchanged": True}
        summary = generation_summary(rows, provenance=provenance)
        output = config.paths.artifact_dir / run_id
        publish_generation_run(output, rows, summary)
        logger.info("E6 cases=%d passed=%d failed=%d artifact=%s", summary["cases"], summary["passed"], summary["failed"], output)
        if summary["failed"]:
            print(json.dumps({"artifact_dir": str(output), **summary}, allow_nan=False))
            return 2
        if args.update_manifest:
            verification = {"generation_run": _record(output / "generation_run.jsonl"),
                            "summary": _record(output / "generation_summary.json")}
            path = config.paths.artifact_dir / "manifest.json"
            if current_git_state() != (source_commit, source_dirty):
                raise FixtureError("Source Git state changed during E6 evaluation")
            merged = merge_e6_generation_manifest(load_manifest(path), config=config, verification=verification)
            write_manifest_atomic(path, merged)
        print(json.dumps({"artifact_dir": str(output), "manifest_updated": args.update_manifest, **summary}, allow_nan=False))
        return 0
    except Exception as exc:
        if logger: logger.exception("E6 setup/provenance failed")
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
