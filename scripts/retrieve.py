"""Retrieve Top-K SciFact documents (implemented in E3)."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys
import time

from app.data_audit import EXPECTED_FILE_SHA256
from app.embedder import MiniLMEmbedder
from app.indexer import load_bundle
from app.loader import load_queries
from app.retriever import (
    DenseRetriever,
    QueryValidationError,
    validate_query,
    validate_retrieval_settings,
)
from scripts._common import bootstrap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query")
    source.add_argument("--query-id")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override retrieval.top_k from config",
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--bundle", type=Path, help="Default: config index_dir/scifact")
    parser.add_argument("--cache-folder", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device")
    return parser


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
        top_k = config.retrieval.top_k if args.top_k is None else args.top_k
        query = args.query
        if args.query_id is not None:
            queries = load_queries(config.paths.queries)
            if args.query_id not in queries:
                raise QueryValidationError(f"Query ID not found: {args.query_id}")
            query = queries[args.query_id].text
        query = validate_query(
            query,
            top_k,
            max_top_k=config.retrieval.max_top_k,
        )
        directory = args.bundle if args.bundle is not None else config.paths.index_dir / "scifact"
        bundle = load_bundle(directory, expected_corpus_sha256=EXPECTED_FILE_SHA256["corpus"])
        embedder = MiniLMEmbedder(config.embedding, cache_folder=args.cache_folder,
                                  local_files_only=args.local_files_only, device=args.device)
        results = DenseRetriever(
            bundle,
            embedder,
            max_top_k=config.retrieval.max_top_k,
        ).retrieve(query, top_k)
        logger.info(
            "dense retrieval mode=%s top_k=%d max_top_k=%d model=%s results=%s duration_s=%.3f bundle=%s",
            config.retrieval.mode,
            top_k,
            config.retrieval.max_top_k,
            embedder.runtime_metadata(),
            [(r.doc_id, r.chunk_id) for r in results],
            time.perf_counter() - started,
            directory,
        )
        print(json.dumps({"run_id": run_id, "query_id": args.query_id, "query": query,
                          "top_k": top_k, "results": [asdict(r) for r in results]},
                         ensure_ascii=True, allow_nan=False))
        return 0
    except Exception as exc:
        if logger is not None:
            logger.exception("retrieval failed")
        else:
            logging.exception("retrieval failed")
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 2 if isinstance(exc, QueryValidationError) else 1


if __name__ == "__main__":
    raise SystemExit(main())

