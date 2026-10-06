# 00 - Project contract

## Goal

Build a reproducible RAG system over SciFact BEIR that:

1. Retrieves evidence from a real index before generation.
2. Sends only retrieved evidence to the local LLM.
3. Produces answers that can be traced to doc_id and chunk_id.
4. Rejects unsupported or conflicting conclusions.
5. Measures retrieval quality quantitatively.
6. Preserves logs and artifacts so the result can be audited and reproduced.

Fine-tuning is outside the exercise.

## Core/Fresher scope - 24 hours

Core includes:

- project skeleton, configuration, logging
- SciFact loader, validation, dataset hashes
- deterministic dev/practice split
- embedding-token-aware chunking
- MiniLM embeddings
- FAISS IndexFlatIP persistence and reload
- dense retrieval with doc-level dedup
- local Ministral generation
- context/prompt guardrails
- citation extraction and validation
- semantic status decision
- generation fixture F01-F06
- Recall@5, Recall@10, MRR@10, nDCG@10
- at least five real failure analyses
- README, manifest, raw run artifacts, report and demo

## Junior scope - additional 16 hours

Junior begins after Core is frozen and adds:

- BM25 lexical retrieval
- dense + BM25 RRF fusion
- cross-encoder reranking
- four comparable retrieval configurations
- FastAPI POST /ask
- API validation, readiness and timeout handling
- p50/p95 latency
- token/runtime accounting
- at least ten failure analyses
- comparison report for the four modes

## Final machine decision

The system is developed, benchmarked and demonstrated on the final Windows 11 `Admin` profile with an NVIDIA GeForce RTX 5070. The repository `.venv` reports Python 3.11.9; PyTorch 2.11.0+cu128 sees CUDA 12.8 and 11.94 GiB VRAM. Ollama 0.24.0 serves the verified `ministral-3:3b-instruct-2512-q8_0` profile (`c269e5748d11`, Q8_0) at `http://127.0.0.1:11434/v1`.

The backlog's reference Ministral checkpoint is BF16 and is described as potentially requiring around 16 GB VRAM. Therefore 12 GB VRAM is a hard design constraint for the demo profile.

Implementation decision:

1. Probe GPU, driver/runtime, CUDA/PyTorch visibility and free VRAM.
2. Verify the exact local Ministral profile that fits the machine.
3. Prefer a verified quantized variant/configuration of the same Ministral 3 3B Instruct family when BF16 does not fit.
4. Pin exact model ID/revision/dtype/quantization/runtime.
5. Record the actual maximum model length if it must be reduced.
6. Keep the generator behind an interface so runtime choice does not leak into RAG business logic.

The spec intentionally does not invent a quantized model identifier before local verification.

## Non-goals

The baseline does not require:

- model fine-tuning
- public cloud LLM usage
- web search during benchmark answers
- public production deployment
- multi-user authentication
- managed vector database
- LangChain/LlamaIndex core pipeline
- query-specific benchmark tricks

## Global invariants

- Corpus is the only answer evidence source at runtime.
- Qrels and gold answers never enter runtime retrieval/generation.
- Retrieval precedes generation.
- Document dedup happens before final Top-K and metrics.
- Context preserves source IDs required for citation validation.
- Invalid citations never count as grounded.
- Infrastructure errors remain distinct from evidence insufficiency.
- Failed queries remain in metric denominators.
- Tuning uses dev only.
- Final test runs happen only after freeze.
- Sources/libraries and AI-assisted code must be disclosed in the final handoff/report, including what was subsequently changed by the author.

## Core Definition of Done

Core is done when all of the following are true:

- Dataset counts and hashes are audited.
- Dev/practice split is deterministic and stored in manifest.
- Chunk limit and overlap are covered by tests.
- Embeddings are 384-dimensional normalized float32.
- FAISS index and metadata persist and reload without re-embedding.
- retrieve returns document IDs, rank, score and evidence chunk.
- ask runs retrieval -> context -> local Ministral -> citation validation end to end.
- Three semantic statuses work.
- F01-F06 run from a separate fixture index.
- Dev metrics cover all 100 dev IDs.
- Retrieval/generation artifacts are parseable and traceable to a manifest.
- README supports setup and demo from a clean environment.
- Prompt injection and leakage rules remain enforced.
