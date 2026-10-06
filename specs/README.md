# RAG SciFact specs

This folder is a derived implementation guide for the two canonical project documents dated 05/10/2026:

- DFM-ENGINEERING “Đề thực hành xây dựng hệ thống RAG”, version 1.0.
- “Backlog RAG SciFact | Local Ministral 3 3B Instruct”, version 1.0.

The original exercise's Qwen endpoint is replaced by a local Ministral 3 3B Instruct runtime. Dataset rules, retrieval rules, chunking, metrics, citation requirements, integrity rules, and delivery requirements remain aligned with the exercise/backlog. If a spec conflicts with the canonical documents, update the spec; do not treat the derived text as permission to change the benchmark contract.

## Final environment

- Windows 11 x64; project user/profile: `Admin`.
- NVIDIA GeForce RTX 5070; PyTorch reports 11.94 GiB VRAM.
- Python `.venv`: 3.11.9.
- PyTorch: 2.11.0+cu128 with CUDA 12.8; CUDA visibility verified.
- Ollama: 0.24.0.
- Verified model: `ministral-3:3b-instruct-2512-q8_0` (`c269e5748d11`, Q8_0).
- Local API: `http://127.0.0.1:11434/v1`.
- Effective context observed: 32768 tokens; model ran 100% GPU in `ollama ps` and completed an OpenAI-compatible chat request.

Environment and runtime records refer to this final Admin machine. E4 application integration is still pending even though the hardware/runtime smoke test is complete.

## Spec map

| File | Scope | Backlog mapping |
| --- | --- | --- |
| 00-project-contract.md | Scope, milestones, invariants, Definition of Done, hardware profile | Global / E0 |
| 01-repository-and-config.md | Repository structure, config, manifest, logging, artifacts | E0, E4, E8 |
| 02-data-retrieval-baseline.md | SciFact data, split, chunking, embeddings, FAISS, dense retrieval | E1-E3 |
| 03-local-ministral-runtime.md | Local Ministral runtime and RTX 5070 12 GB decisions | E4 |
| 04-rag-generation-and-citations.md | Prompt/context, response schema, citations, semantic statuses | E5 |
| 05-evaluation-and-fixtures.md | F01-F06, metrics, benchmark artifacts, error analysis | E6-E7 |
| 06-junior-extensions.md | BM25, RRF, reranker, four modes, FastAPI, performance | E9-E14 |
| 07-delivery-and-demo.md | Work plan, freeze rules, deliverables, report, demo | E8, E14 |

## Usage rule

For every implementation task:

1. Identify the spec that owns the behavior.
2. Implement against that spec's acceptance criteria.
3. Keep evaluator-only information out of runtime code and prompts.
4. Record deviations explicitly rather than silently changing the benchmark.

Core/Fresher scope is E0-E8 and is budgeted at 24 hours. Junior adds E9-E14 for another 16 hours, for a total of 40 hours. Junior optimization starts only after the dense Core baseline has been frozen.
