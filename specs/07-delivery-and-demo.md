# 07 - Delivery, freeze and demo

## Suggested execution plan

The backlog budgets the work as follows.

### Day 1 - Core

Scope:

- E0 project/environment
- E1 dataset/audit/split
- first part of E2 tokenizer/chunker

Exit:

- deterministic data pipeline
- Python 3.11 environment confirmed
- dataset counts/hashes available
- dev/practice split reproducible

### Day 2 - Core

Scope:

- finish E2
- E3 dense index/retrieval
- E4 local Ministral runtime

Exit:

- index builds/reloads
- dense retrieve works
- one independent local Ministral completion works on the RTX 5070 12 GB profile

### Day 3 - Core

Scope:

- E5 RAG generation/citations
- E6 generation fixture
- E7 metrics/error analysis
- E8 README/report/demo

Exit:

- end-to-end ask works
- F01-F06 run
- dev metrics produced
- at least five failures analyzed
- Core baseline frozen

### Day 4 - Junior

Scope:

- E9 BM25/RRF
- E10 reranker
- first part of E11 four-mode comparison

Exit:

- all four retrieval modes runnable on dev

### Day 5 - Junior

Scope:

- finish E11
- E12 FastAPI
- E13 performance/robustness
- E14 comparison/error analysis/final freeze

Exit:

- API demo works
- p50/p95 comparison available
- at least ten failures analyzed
- final package frozen

## Required delivery artifacts

Submit:

- source code
- frozen git commit hash
- requirements/lockfile
- .env.example
- final config
- manifest.json
- reloadable FAISS index + metadata mapping
- retrieval_run.jsonl
- metrics.json
- generation_run.jsonl
- README
- report
- demo procedure
- disclosure of sources/libraries used and AI-assisted code, including the parts subsequently changed by the author

manifest.json may be a generated, gitignored artifact rather than a tracked source file. It is still mandatory in the submitted delivery bundle. Generate it after the source/config freeze commit so it can record the frozen commit hash without making that source commit self-referential.

Junior also submits the four-mode comparison artifacts and API instructions.

## manifest.json minimum

Must identify:

- dataset hashes/counts
- dev/practice split IDs
- seed
- chunk config
- embedding model/revision where available
- reranker model/revision for Junior
- local Ministral exact model plus upstream revision when applicable
- runtime model ID/digest (for Ollama, separate from upstream revision)
- dtype/quantization
- runtime/library versions
- hardware
- inference parameters
- prompt version
- freeze commit/config identity

## README requirements

README must let a reviewer reproduce the main path from a clean machine/environment:

1. install Python 3.11 environment
2. install locked dependencies
3. obtain/place SciFact data
4. start the verified local Ministral runtime
5. build the index
6. reload/reuse the index
7. run retrieve
8. run ask
9. run dev evaluate
10. run generation fixture tests
11. run pytest
12. for Junior, start FastAPI and exercise POST /ask

Use copy/paste-ready Windows PowerShell commands for the primary demo machine. If WSL2 is required for the selected LLM runtime, make the Windows/WSL boundary explicit.

## Report

Maximum report length: 5 pages.

Core report covers:

- architecture
- baseline configuration
- dataset/split
- retrieval metrics
- generation/citation behavior
- at least five failure cases
- local runtime/latency
- limitations
- explicit note that Qwen was replaced by local Ministral 3 3B Instruct
- sources/libraries used, AI-assisted code and what the author changed afterward
- run attempts/cache behavior needed to explain reproducibility or failed runs

Junior adds:

- four-mode comparison
- p50/p95
- token/runtime trade-offs
- at least ten failure cases
- rationale for final mode selection

Reported numbers must match generated artifacts.

## Freeze rule

Before final test evaluation:

- commit code
- snapshot config
- freeze prompt version
- freeze retrieval settings and thresholds
- freeze model ID/revision/quantization/runtime
- write the freeze point into manifest

Once a manifest records final_test_frozen=true, rerunning an audit/manifest merge is allowed only from the same recorded Git commit with a clean working tree. A dirty tree, a different commit, a missing frozen commit identity, or a frozen manifest that itself records git_dirty=true must fail rather than silently preserving the frozen claim while changing source identity.

After viewing final test results, do not change:

- K
- threshold
- prompt
- model
- chunk configuration
- retrieval mode

The purpose of the final test is evaluation, not tuning.

## Demo script - 5 to 10 minutes

Prepare a short deterministic demo that does not rebuild the index live.

Recommended order:

1. Show local model health/readiness.
2. Reload an existing index.
3. Run one answered query with valid evidence/citation.
4. Run one insufficient-evidence fixture case.
5. Run the F04/F05 conflicting-evidence case.
6. Run F06 prompt-injection case.
7. Show a dev metrics artifact.
8. For Junior, show POST /ask and optionally switch/compare retrieval modes.

Have a short explanation ready if local model startup is slow. Do not replace a failed live call with fabricated output.

## Minimum final test suite

Core:

- loader/ID handling
- chunk token limit
- overlap
- embedding shape/norm
- retrieval dedup/order
- index reload
- citation validation
- local LLM-down error mapping
- F06 prompt injection
- metric toy tests

Junior:

- BM25/RRF
- reranker
- invalid API inputs
- timeout
- missing/corrupt index
- local LLM down
- F06 through API/retrieval modes

Tests that validate application logic should run without public web/cloud LLM dependency.

## Final acceptance checklist

- Clean environment uses Python 3.11; this final machine currently uses `.venv` Python 3.11.9.
- Verified Ollama `ministral-3:3b-instruct-2512-q8_0` profile fits the RTX 5070 12 GB Admin machine and is documented; application-level E4 integration must also pass before final freeze.
- Core pipeline retrieves before generation.
- Index reloads without re-embedding.
- Citations are source-valid.
- Three semantic statuses work.
- Infrastructure errors remain errors.
- All dev failures stay in denominator.
- Required metrics/artifacts exist.
- F01-F06 run from a separate fixture index.
- Integrity rules are satisfied.
- README/report/demo are consistent with the frozen manifest.
