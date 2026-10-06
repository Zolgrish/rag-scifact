"""Build, persist and verify the canonical SciFact dense index (E3)."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import logging
from pathlib import Path
import sys
import time

from app.chunker import MiniLMChunker
from app.config import REPO_ROOT
from app.data_audit import EXPECTED_CORPUS_COUNT, EXPECTED_FILE_SHA256, file_sha256
from app.embedder import MiniLMEmbedder
from app.indexer import build_index, load_bundle, save_bundle
from app.loader import DatasetMismatchError, load_corpus
from app.manifest import write_manifest_atomic
from app.retriever import validate_retrieval_settings
from scripts._common import bootstrap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--bundle", type=Path, help="Default: config index_dir/scifact")
    parser.add_argument("--cache-folder", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


@contextmanager
def chunk_logging(logger):
    """Route E2 truncation events through this build's existing run handlers."""
    child = logging.getLogger("app.chunker")
    old_level, old_propagate = child.level, child.propagate
    child.setLevel(logger.level)
    child.propagate = False
    added = [handler for handler in logger.handlers if handler not in child.handlers]
    for handler in added:
        child.addHandler(handler)
    try:
        yield
    finally:
        for handler in added:
            child.removeHandler(handler)
        child.setLevel(old_level)
        child.propagate = old_propagate


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = None
    started = time.perf_counter()
    try:
        config, logger, run_id = bootstrap(args.config)
        validate_retrieval_settings(
            config.retrieval.mode,
            config.retrieval.max_top_k,
        )
        if args.batch_size <= 0:
            raise ValueError("batch-size must be positive")
        digest = file_sha256(config.paths.corpus)
        if digest != EXPECTED_FILE_SHA256["corpus"]:
            raise DatasetMismatchError("Canonical corpus SHA256 mismatch")
        corpus = load_corpus(config.paths.corpus)
        if len(corpus) != EXPECTED_CORPUS_COUNT:
            raise DatasetMismatchError(f"Canonical corpus document count mismatch: {len(corpus)}")
        # No queries/qrels are loaded anywhere in this build path.
        embedder = MiniLMEmbedder(
            config.embedding, cache_folder=args.cache_folder,
            local_files_only=args.local_files_only, device=args.device,
        )
        chunker = MiniLMChunker.from_embedder(embedder)
        logger.info("corpus_sha256=%s documents=%d embedding=%s chunking=%s batch_size=%d",
                    digest, len(corpus), embedder.runtime_metadata(), chunker.runtime_metadata(), args.batch_size)
        chunks = []
        with chunk_logging(logger):
            # load_corpus preserves canonical JSONL order; preserve E2 output
            # order within each document for deterministic vector positions.
            for document in corpus.values():
                chunks.extend(chunker.chunk_document(document))
        logger.info("documents=%d chunks=%d; encoding in batches of %d", len(corpus), len(chunks), args.batch_size)
        vectors = embedder.encode_chunks(chunks, batch_size=args.batch_size)
        bundle = build_index(
            vectors, chunks, run_id=run_id,
            corpus={"name": "scifact_beir", "sha256": digest, "documents": len(corpus)},
            embedding=embedder.runtime_metadata(), chunking=chunker.runtime_metadata(),
        )
        directory = args.bundle if args.bundle is not None else config.paths.index_dir / "scifact"
        manifest = save_bundle(bundle, directory)
        reloaded = load_bundle(directory, expected_corpus_sha256=digest)
        if reloaded.chunks != bundle.chunks:
            raise RuntimeError("Post-write mapping verification differs from build mapping")
        import numpy as np

        # Reconstruct existing vectors; this is never another model encoding.
        np.testing.assert_array_equal(reloaded.index.reconstruct_n(0, len(chunks)), vectors)
        norms = np.linalg.norm(vectors, axis=1)
        duration = time.perf_counter() - started
        report = {
            "status": "PASS", "run_id": run_id, "bundle_path": str(directory.resolve()),
            "documents": len(corpus), "chunks": len(chunks), "vector_shape": list(vectors.shape),
            "dtype": str(vectors.dtype), "norm_min": float(norms.min()), "norm_max": float(norms.max()),
            "index_type": manifest["index_type"], "dimension": reloaded.index.d,
            "ntotal": reloaded.index.ntotal, "mapping_count": len(reloaded.chunks),
            "duplicate_chunk_ids": len(chunks) - len({c.chunk_id for c in chunks}),
            "model_id": manifest["embedding"]["model_id"], "model_revision": manifest["embedding"]["revision"],
            "device": manifest["embedding"]["device"], "batch_size": args.batch_size,
            "duration_seconds": duration, "files": manifest["files"],
            "reload_vectors_and_mapping_equal": True, "index_manifest": manifest,
        }
        artifact = config.paths.artifact_dir / run_id
        write_manifest_atomic(artifact / "index_build.json", report)
        config_source = Path(args.config)
        if not config_source.is_absolute():
            config_source = REPO_ROOT / config_source
        (artifact / "config.snapshot.yaml").write_bytes(config_source.read_bytes())
        logger.info("build PASS index_type=%s dimension=%d ntotal=%d mapping=%d duration_s=%.3f bundle=%s",
                    report["index_type"], report["dimension"], report["ntotal"], report["mapping_count"], duration, directory)
        print(json.dumps({key: value for key, value in report.items() if key != "index_manifest"}, allow_nan=False))
        return 0
    except Exception as exc:
        if logger is not None:
            logger.exception("index build failed")
        else:
            logging.exception("index build failed")
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

