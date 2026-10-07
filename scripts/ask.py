"""Run grounded RAG with the existing dense index and locked local Generator."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from uuid import uuid4

from app.citations import RAGOutputValidationError
from app.generator import GeneratorInfrastructureError
from app.rag import build_rag_pipeline, response_payload
from app.retriever import QueryValidationError, validate_query
from scripts._common import bootstrap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--config", default="config.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request_id = uuid4().hex
    logger = logging.getLogger(__name__)
    try:
        config, logger, _run_id = bootstrap(args.config)
        top_k = config.retrieval.top_k if args.top_k is None else args.top_k
        query = validate_query(args.query, top_k, max_top_k=config.retrieval.max_top_k)
        pipeline = build_rag_pipeline(config, logger=logger)
        execution = pipeline.ask(query, top_k, request_id=request_id)
        print(json.dumps(response_payload(execution.response), ensure_ascii=True, allow_nan=False))
        return 0
    except Exception as exc:
        logger.exception("request_id=%s ask failed error=%s", request_id, type(exc).__name__)
        error = {"request_id": request_id, "error": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, GeneratorInfrastructureError):
            error["details"] = exc.to_dict()
        print(json.dumps(error, ensure_ascii=True), file=sys.stderr)
        if isinstance(exc, QueryValidationError):
            return 2
        return 3 if isinstance(exc, RAGOutputValidationError) else 1


if __name__ == "__main__":
    raise SystemExit(main())

