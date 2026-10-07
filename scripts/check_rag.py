"""Verify real E5 RAG/counting and explicitly record prompt provenance."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import sys

from app.generator import GeneratorMalformedResponseError, validate_runtime_metadata
from app.config import REPO_ROOT
from app.manifest import (E5_CANONICAL_SMOKE_QUERY, current_git_state, load_manifest,
                          merge_e5_prompt_manifest,
                          ManifestConflictError,
                          runtime_profile_from_config, runtime_profile_sha256,
                          write_manifest_atomic)
from app.prompt import build_messages, prompt_identity
from app.rag import build_rag_pipeline, response_payload
from scripts._common import bootstrap


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--query", default=E5_CANONICAL_SMOKE_QUERY)
    parser.add_argument("--update-manifest", action="store_true")
    args = parser.parse_args(argv)
    logger = None
    try:
        config, logger, run_id = bootstrap(args.config)
        source_git_commit, source_git_dirty = current_git_state()
        artifact = config.paths.artifact_dir / run_id / "rag_check.json"
        if args.update_manifest:
            if not config.llm.runtime_profile_locked:
                raise ManifestConflictError("E5 update requires runtime_profile_locked=true")
            if not artifact.resolve().is_relative_to(REPO_ROOT.resolve()):
                raise ManifestConflictError("E5 provenance artifact must be inside repository")
            if args.query != E5_CANONICAL_SMOKE_QUERY:
                raise ManifestConflictError("E5 canonical update requires the canonical smoke query")
        pipeline = build_rag_pipeline(config, logger=logger)
        execution = pipeline.ask(args.query, config.retrieval.top_k, request_id=run_id)
        # Same fixed explicit-system chat via both endpoints; only local runtime calls.
        messages = build_messages("What can the supplied evidence establish?", ())
        native_count = pipeline.generator.count_prompt_tokens(messages)
        completion = pipeline.generator.generate(messages, response_format={"type": "json_object"})
        completion_count = (completion.usage or {}).get("prompt_tokens")
        if native_count != completion_count:
            raise GeneratorMalformedResponseError("Native and completion prompt counts differ")
        observed = pipeline.generator.inspect_runtime_metadata()
        validate_runtime_metadata(config.llm, observed)
        payload = {"status": "PASS", "run_id": run_id, "query": args.query,
                   "source_git_commit": source_git_commit,
                   "source_git_dirty": source_git_dirty,
                   "prompt": dict(prompt_identity()),
                   "profile_sha256": runtime_profile_sha256(runtime_profile_from_config(config)),
                   "observed_runtime": asdict(observed),
                   "token_counter_check": {"native_prompt_tokens": native_count,
                                           "completion_prompt_tokens": completion_count},
                   "response": response_payload(execution.response), "trace": asdict(execution.trace)}
        write_manifest_atomic(artifact, payload)
        if args.update_manifest:
            manifest_path = config.paths.artifact_dir / "manifest.json"
            verification = {"artifact": artifact.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
                            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
            # Re-read to detect concurrent changes; helper validates the lock again.
            merged = merge_e5_prompt_manifest(load_manifest(manifest_path), config=config,
                                              verification=verification)
            write_manifest_atomic(manifest_path, merged)
        print(json.dumps({"status": "PASS", "artifact": str(artifact),
                          "manifest_updated": args.update_manifest, **payload},
                         ensure_ascii=True, allow_nan=False))
        return 0
    except Exception as exc:
        if logger:
            logger.error("E5 check failed error=%s", type(exc).__name__)
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
