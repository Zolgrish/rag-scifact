"""Reusable retrieve-before-generate composition with structural grounding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import time
from typing import Protocol
from uuid import uuid4

from app.citations import parse_output, validate_output
from app.config import AppConfig, load_config
from app.context import build_context
from app.data_audit import EXPECTED_FILE_SHA256
from app.embedder import MiniLMEmbedder
from app.generator import Generator, build_generator
from app.indexer import load_bundle
from app.manifest import load_manifest, validate_e5_manifest, validate_runtime_profile_lock
from app.models import RAGResponse, RetrievedDocument
from app.prompt import build_messages, prompt_identity
from app.retriever import DenseRetriever, validate_query, validate_retrieval_settings


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]: ...


@dataclass(frozen=True)
class RAGTrace:
    request_id: str
    retrieved_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    excluded_context_ids: tuple[str, ...]
    context_truncated: bool
    input_token_count: int
    prompt_version: str
    prompt_sha256: str
    model_id: str


@dataclass(frozen=True)
class RAGExecution:
    response: RAGResponse
    trace: RAGTrace


def response_payload(response: RAGResponse) -> dict[str, object]:
    """Public response omits internal evidence text and chunk metadata."""
    return {"request_id": response.request_id, "answer": response.answer,
            "status": response.status.value,
            "citations": [asdict(c) for c in response.citations],
            "retrieved": [{"doc_id": r.doc_id, "rank": r.rank, "score": r.score}
                          for r in response.retrieved],
            "timing_ms": dict(response.timing_ms), "model_id": response.model_id}


class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: Generator, *,
                 context_length: int = 32768, max_new_tokens: int = 512,
                 max_top_k: int = 10, logger: logging.Logger | None = None) -> None:
        self.retriever = retriever
        self.generator = generator
        self.context_length = context_length
        self.max_new_tokens = max_new_tokens
        self.max_top_k = max_top_k
        self.logger = logger or logging.getLogger(__name__)

    def ask(self, query: str, top_k: int, *, request_id: str | None = None) -> RAGExecution:
        started = time.perf_counter()
        request_id = request_id or uuid4().hex
        stage = "validate"
        identity = prompt_identity()
        try:
            query = validate_query(query, top_k, max_top_k=self.max_top_k)
            stage = "retrieve"
            retrieval_start = time.perf_counter()
            retrieved = self.retriever.retrieve(query, top_k)
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
            self.logger.info("request_id=%s retrieved=%s", request_id,
                             [(r.doc_id, r.chunk_id) for r in retrieved])
            stage = "context_budget"
            budget_start = time.perf_counter()
            context = build_context(query, retrieved,
                                    count_prompt_tokens=self.generator.count_prompt_tokens,
                                    context_length=self.context_length,
                                    max_new_tokens=self.max_new_tokens)
            budget_ms = (time.perf_counter() - budget_start) * 1000
            self.logger.info(
                "request_id=%s included=%s excluded=%s truncated=%s input_tokens=%d prompt=%s sha256=%s",
                request_id, [(i.doc_id, i.chunk_id) for i in context.included],
                [(i.doc_id, i.chunk_id) for i in context.excluded], context.truncated,
                context.input_token_count, identity["version"], identity["sha256"])
            stage = "generate"
            messages = build_messages(query, context.included)
            generation_start = time.perf_counter()
            result = self.generator.generate(messages, response_format={"type": "json_object"})
            generation_ms = (result.generation_ms if result.generation_ms is not None
                             else (time.perf_counter() - generation_start) * 1000)
            stage = "output_validation"
            output = parse_output(result.text)
            validate_output(output, context)
            response = RAGResponse(request_id, output.answer, output.status,
                                   list(output.citations), list(retrieved),
                                   {"retrieval": retrieval_ms, "context_budget": budget_ms,
                                    "generation": generation_ms}, result.model_id)
            trace = RAGTrace(request_id, tuple(r.chunk_id for r in retrieved),
                             tuple(i.chunk_id for i in context.included),
                             tuple(i.chunk_id for i in context.excluded), context.truncated,
                             context.input_token_count, str(identity["version"]),
                             str(identity["sha256"]), result.model_id)
            execution = RAGExecution(response, trace)
            response.timing_ms["total"] = (time.perf_counter() - started) * 1000
            self.logger.info("request_id=%s status=%s model=%s timing_ms=%s", request_id,
                             output.status.value, result.model_id, response.timing_ms)
            return execution
        except Exception as exc:
            # Keep original E3/E4 exceptions and diagnostics; never assign a semantic status.
            self.logger.error("request_id=%s stage=%s error=%s prompt=%s sha256=%s",
                              request_id, stage, type(exc).__name__,
                              identity["version"], identity["sha256"])
            raise


def build_rag_pipeline(config: AppConfig | str | Path = "config.yaml", *,
                       logger: logging.Logger | None = None,
                       cache_folder: str | Path | None = None,
                       device: str | None = None) -> RAGPipeline:
    """Single composition path; validate canonical locks before runtime construction."""
    config = config if isinstance(config, AppConfig) else load_config(config)
    return build_rag_pipeline_from_bundle(
        config, bundle_path=config.paths.index_dir / "scifact",
        expected_corpus_sha256=EXPECTED_FILE_SHA256["corpus"],
        logger=logger, cache_folder=cache_folder, device=device,
    )


def build_rag_pipeline_from_bundle(
    config: AppConfig, *, bundle_path: str | Path, expected_corpus_sha256: str,
    logger: logging.Logger | None = None, cache_folder: str | Path | None = None,
    device: str | None = None,
) -> RAGPipeline:
    """Compose the same production pipeline against an explicitly identified corpus."""
    manifest = load_manifest(config.paths.artifact_dir / "manifest.json")
    validate_runtime_profile_lock(manifest, config)
    validate_e5_manifest(manifest, config=config)
    validate_retrieval_settings(config.retrieval.mode, config.retrieval.max_top_k)
    bundle = load_bundle(bundle_path, expected_corpus_sha256=expected_corpus_sha256)
    # Reuse the verified repository cache if present; all normal asks stay offline.
    cached = config.paths.artifact_dir / "e2-validation" / "model-cache"
    cache_folder = cache_folder if cache_folder is not None else (cached if cached.is_dir() else None)
    embedder = MiniLMEmbedder(config.embedding, cache_folder=cache_folder,
                              local_files_only=True, device=device)
    retriever = DenseRetriever(bundle, embedder, max_top_k=config.retrieval.max_top_k)
    generator = build_generator(config.llm)
    return RAGPipeline(retriever, generator, context_length=config.llm.context_length,
                       max_new_tokens=config.llm.max_new_tokens,
                       max_top_k=config.retrieval.max_top_k, logger=logger)

