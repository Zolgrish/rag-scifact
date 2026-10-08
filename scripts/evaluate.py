"""Evaluate SciFact retrieval metrics (implemented in E7)."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

from app.benchmark_manifest import (artifact_record, canonical_generation_timing, index_identity,
    load_evaluation_data, retrieval_identity)
from app.config import effective_config_identity
from app.embedder import MiniLMEmbedder
from app.manifest import (current_git_state, load_manifest, merge_e7_retrieval_manifest,
    validate_runtime_profile_lock, validate_e5_manifest, validate_final_test_gate, write_manifest_atomic)
from app.retriever import DenseRetriever
from app.retrieval_evaluator import (RetrievalEvaluationError, aggregate, evaluate_queries,
    failure_candidates, metric_policy_identity, publish_run, retrieval_rows)
from scripts._common import bootstrap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cache-folder", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--update-manifest", action="store_true")
    return parser


def runtime_metadata(embedder: MiniLMEmbedder) -> dict[str, object]:
    import faiss
    import torch
    metadata = {"python": platform.python_version(), "platform": platform.platform(),
                "embedding": embedder.runtime_metadata(), "faiss_threads": faiss.omp_get_max_threads(),
                "versions": {name: importlib.metadata.version(name) for name in
                    ("numpy", "faiss-cpu", "sentence-transformers", "torch")},
                "torch_cuda_version": torch.version.cuda}
    if str(metadata["embedding"].get("device", "")).startswith("cuda"):
        device = torch.device(metadata["embedding"]["device"])
        properties = torch.cuda.get_device_properties(device)
        metadata["gpu"] = {"name": properties.name, "total_memory_bytes": properties.total_memory}
    return metadata


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = None
    try:
        config, logger, run_id = bootstrap(args.config)
        commit, dirty = current_git_state()
        manifest_path = config.paths.artifact_dir / "manifest.json"
        manifest_record = artifact_record(manifest_path)
        manifest = load_manifest(manifest_path)
        # The complete seal is checked BEFORE dataset or qrels test access.
        if args.split == "test":
            validate_final_test_gate(manifest, config=config)
            if args.update_manifest:
                raise RetrievalEvaluationError("Canonical updates are dev-only; test reruns produce new artifacts")
        validate_runtime_profile_lock(manifest, config)
        validate_e5_manifest(manifest, config=config)
        retrieval = retrieval_identity(config)
        raw_config = artifact_record(args.config)
        inputs = load_evaluation_data(config, manifest, split=args.split)
        bundle, index = index_identity(config, manifest, inputs.corpus)
        cached = config.paths.artifact_dir / "e2-validation" / "model-cache"
        embedder = MiniLMEmbedder(config.embedding, cache_folder=args.cache_folder or (cached if cached.is_dir() else None),
                                  local_files_only=True, device=args.device)
        retriever = DenseRetriever(bundle, embedder, max_top_k=config.retrieval.max_top_k)
        # Global embedding failures abort before any scored query; no LLM call.
        embedder.encode_query("Retrieval embedding runtime readiness")
        hardware = runtime_metadata(embedder)
        generation_timing = canonical_generation_timing(manifest, config)
        outcomes = evaluate_queries(retriever, inputs.queries)
        candidates = failure_candidates(outcomes, inputs.queries, inputs.gold)
        if artifact_record(manifest_path) != manifest_record or artifact_record(args.config) != raw_config:
            raise RetrievalEvaluationError("Manifest/config changed during benchmark")
        if any(artifact_record(record["artifact"]) != record for record in index["files"].values()):
            raise RetrievalEvaluationError("SciFact bundle changed during benchmark")
        for name, record in inputs.dataset_identity["hashes"].items():
            if name != "qrels_test" or args.split == "test":
                if artifact_record(record["path"])["sha256"] != record["sha256"]:
                    raise RetrievalEvaluationError("Dataset changed during benchmark")
        if args.split == "test":
            # Test access was gated before judgments were loaded. Revalidate the
            # complete frozen seal immediately before publication so a source or
            # baseline change during the long-running benchmark cannot produce a
            # stale-provenance final-test artifact.
            validate_final_test_gate(manifest, config=config)
            if current_git_state() != (commit, False):
                raise RetrievalEvaluationError("Frozen source changed during final-test benchmark")
        summary = {"schema_version": 1, "run_id": run_id, "status": "COMPLETED", "split": args.split,
                   "source_git_commit": commit, "source_git_dirty": dirty, "config": raw_config,
                   "effective_config": effective_config_identity(config), "retrieval": retrieval,
                   "split_identity": inputs.split_identity, "dataset": inputs.dataset_identity,
                   "index": index, "metric_policy": metric_policy_identity(), "runtime": hardware,
                   "generation_timing": generation_timing, "failure_candidate_count": len(candidates),
                   **aggregate(outcomes, inputs.gold)}
        output = config.paths.artifact_dir / run_id
        completed = publish_run(output, retrieval_rows(outcomes, run_id), candidates, summary)
        if args.update_manifest:
            if current_git_state() != (commit, dirty):
                raise RetrievalEvaluationError("Source changed during benchmark")
            merged = merge_e7_retrieval_manifest(manifest, config=config,
                                                verification={"metrics": artifact_record(output / "metrics.json")})
            json.dumps(merged, allow_nan=False)
            write_manifest_atomic(manifest_path, merged)
        logger.info("E7 split=%s denominator=%d failures=%d metrics=%s artifact=%s",
                    args.split, completed["denominator"], completed["failure_count"], completed["metrics"], output)
        print(json.dumps({"artifact_dir": str(output), "manifest_updated": args.update_manifest, **completed}, allow_nan=False))
        return 2 if completed["failure_count"] else 0
    except Exception as exc:
        if logger:
            logger.exception("E7 setup/provenance failure")
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
