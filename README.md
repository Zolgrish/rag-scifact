# RAG SciFact — Local Ministral Core

This project implements the DFM-ENGINEERING SciFact BEIR RAG exercise. The original Qwen endpoint was intentionally replaced by local Ministral 3 3B Instruct under the Local Ministral backlog. Retrieval precedes generation; answers use retrieved context and citations are validated against that context.

## Current status and source identity

**Core E0–E8 completed.** E0–E7 implement the benchmark; E8 supplies delivery documentation, software validation, the report and demo. Junior E9–E14 remains outside Core scope.

- **Benchmark commit:** `c7711284c066c6ff4a3052a3366cbd826e819b63`.
- **Freeze identity SHA256:** `57299e5bcf5e590698ee6689d3e446ee21c19eddd8a523062bc3b1abafaef6ac`.
- `final_test_frozen=true` and `runtime_profile_locked=true`.
- Official held-out retrieval evaluation completed: **300 queries, denominator 300, zero execution failures**.
- **Delivery commit:** the later commit containing E8 documentation. Git assigns its hash when the author commits; it is intentionally distinct from the benchmark commit. See [delivery provenance](delivery/E8-provenance.json).

The official evidence belongs to the benchmark commit above. Do not update the frozen manifest, refreeze, rebuild the sealed index, rerun the official test or tune settings after viewing its results. Exact historical reproduction requires the reviewer to prepare a clean checkout of the **benchmark commit**, the saved frozen manifest/evidence and identical assets/runtime. A later delivery checkout supports documentation review and software tests; frozen `ask`, generation and DEV evaluation reject dirty/different source identity. This is expected provenance protection, not a reason to change the seal.

## Architecture and locked baseline

```text
corpus -> MiniLM token chunks -> normalized embeddings -> persisted FAISS
query -> same MiniLM -> dense search -> document dedup -> Top-K context
      -> local Ministral JSON -> citation validation -> response and trace
```

| Component | Core contract |
| --- | --- |
| Corpus / queries | SciFact BEIR: 5,183 documents / 1,109 queries |
| Train / test | 809 / 300 query IDs; 919 / 339 qrel rows |
| Internal split | Numerically sorted train IDs, `random.Random(42)` shuffle; first 100 DEV, remaining 709 practice |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, L2 normalized `float32` |
| Embedding revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Chunking | Maximum 220 standalone MiniLM body tokens; exactly 30 token overlap at stable WordPiece boundaries |
| Embedding input | Actual title + body input checked against observed 256-token limit including special tokens |
| Index / retrieval | FAISS `IndexFlatIP`; dense document ranking; highest scoring evidence chunk per document |
| Serving / evaluation | Top-K 5; one Top-10 retrieval per evaluated query; maximum Top-K 10 |
| Prompt | `rag-grounded-json-v4` |
| Generation | Temperature 0, output limit 512 new tokens, seed 42 |

Chunk IDs are `<doc_id>:<token_start>-<token_end>` with half-open source spans. Evidence is the exact original character substring, never reconstructed by decoding. Overlong embedding titles are reduced to a source token prefix until actual `embedding_title + "\n\n" + body` tokenization fits; the original title/body remain available and truncation is logged. Empty bodies produce zero chunks. The saved index contains 10,359 chunks/vectors; that count is observed, not hardcoded.

Deduplication precedes Top-K; ranking uses score descending, then vector position for ties. Recall counts positive judgments, MRR uses the first positive hit, and graded nDCG uses `2**rel - 1` with `log2(rank + 1)` discount. All queries remain in the denominator, including zero-positive judgments and per-query errors. Global setup failure aborts rather than publishing a fake score.

Qrels/gold labels stay in evaluator code. Runtime queries contain ID and text only; gold data never reaches embeddings, index, retrieval, prompts or LLM. Documents are untrusted data. Citations identify a document/chunk actually sent in context and contain an exact or normalized source substring. Semantic statuses are `ANSWERED`, `INSUFFICIENT_EVIDENCE`, `CONFLICTING_EVIDENCE`. Connection errors, timeouts, OOM, corrupt index and invalid model output use separate error paths. Similarity is not correctness probability.

## Verified machine and model choice

Windows 11 x64, Admin; Python 3.11.9; RTX 5070 (11.94 GiB reported VRAM); PyTorch `2.11.0+cu128`; CUDA 12.8; Ollama 0.24.0. Model: `ministral-3:3b-instruct-2512-q8_0`, digest prefix `c269e5748d11`, Q8_0, observed context 32768, 100% GPU placement. Canonical E5/E6 record the full observed digest.

The backlog reference is `mistralai/Ministral-3-3B-Instruct-2512-BF16`. Q8_0 fits the verified approximately 12 GB GPU where reference BF16 may exceed VRAM. Quality equivalence to BF16 or the original Qwen service was not measured. `LLM_REVISION` and compute dtype remain unset; an Ollama digest is recorded separately, not as an upstream revision. The local API is `http://127.0.0.1:11434/v1`; services bind to loopback.

## Reviewer setup: Windows PowerShell

These commands document fresh-environment preparation; do not create assets over the existing sealed delivery. Work in `D:\rag-scifact` on Admin. Start every repository command block with the guard below. Use benchmark source plus supplied evidence for frozen runtime operations; do not change tracked files in that checkout.

```powershell
if ($env:USERNAME -ne 'Admin') { Write-Output "SKIP_PROFILE=$env:USERNAME"; exit 7 }
Set-Location D:\rag-scifact
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock.txt
python --version
python -m scripts.check_environment
python -m pip check
```

The lockfile includes the CUDA 12.8 wheel index. `requirements.txt` and `requirements-dev.txt` are development ranges. Portable configuration uses repository-relative paths. `.env.example` contains the non-secret profile; optional overrides belong in ignored `.env` and must match the frozen effective config. Never commit keys.

### Dataset and delivery assets

Use the exercise's canonical [BEIR SciFact archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip). Download/extract only in a fresh environment. The following copies an already obtained archive:

```powershell
if ($env:USERNAME -ne 'Admin') { Write-Output "SKIP_PROFILE=$env:USERNAME"; exit 7 }
Set-Location D:\rag-scifact
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\scifact.zip" -DestinationPath "$env:TEMP\scifact-e8-setup"
New-Item -ItemType Directory -Force data\scifact\qrels | Out-Null
Copy-Item "$env:TEMP\scifact-e8-setup\scifact\corpus.jsonl" data\scifact\corpus.jsonl
Copy-Item "$env:TEMP\scifact-e8-setup\scifact\queries.jsonl" data\scifact\queries.jsonl
Copy-Item "$env:TEMP\scifact-e8-setup\scifact\qrels\train.tsv" data\scifact\qrels\train.tsv
Copy-Item "$env:TEMP\scifact-e8-setup\scifact\qrels\test.tsv" data\scifact\qrels\test.tsv
Get-FileHash data\scifact\corpus.jsonl,data\scifact\queries.jsonl,data\scifact\qrels\train.tsv,data\scifact\qrels\test.tsv -Algorithm SHA256
```

Verify byte identity before use; counts alone are insufficient:

| File | SHA256 |
| --- | --- |
| `corpus.jsonl` | `dec31c8182f3d744c7d2c09423756fd1d17cbef75808db13ba01cc0aab4d1ac6` |
| `queries.jsonl` | `8ff84a7c903f722981cd8d595c022660140c51867b27608a6d4910db86080313` |
| `qrels/train.tsv` | `a53f2114831916c096b6c37d9e54da68cef4efdcdbd5ed46533601af972acf1d` |
| `qrels/test.tsv` | `0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c` |

Supply ignored dataset, indexes, artifacts and model caches separately: a source clone alone is insufficient. Include frozen `artifacts/manifest.json`, referenced E4 verification, canonical E5/E6/DEV/TEST directories (raw runs and config snapshots), `indexes/scifact/`, `indexes/fixtures/atlas/`, `data/fixtures/atlas.jsonl` and the pinned MiniLM cache. Preserve layout and verify hashes using [E8 provenance](delivery/E8-provenance.json).

The E1 overlap audit reports 2 canonical exact and 98 near duplicate cross-split pairs (25 practice/DEV, 66 practice/TEST, 7 DEV/TEST near pairs). This is integrity disclosure only; overlaps did not change splits or denominators.

### Ollama startup and health

Use recorded Ollama 0.24.0. If no server is running, leave this in a separate terminal:

```powershell
if ($env:USERNAME -ne 'Admin') { Write-Output "SKIP_PROFILE=$env:USERNAME"; exit 7 }
$env:OLLAMA_HOST = '127.0.0.1:11434'
ollama serve
```

Pull only on a fresh machine lacking the exact model; a mutable tag alone does not establish digest identity:

```powershell
if ($env:USERNAME -ne 'Admin') { Write-Output "SKIP_PROFILE=$env:USERNAME"; exit 7 }
Set-Location D:\rag-scifact
ollama --version
ollama pull ministral-3:3b-instruct-2512-q8_0
ollama list
Invoke-RestMethod http://127.0.0.1:11434/api/tags | ConvertTo-Json -Depth 6
.\.venv\Scripts\python.exe -m scripts.check_generator --config config.yaml
ollama ps
```

`check_generator` performs local readiness and runtime-metadata validation and writes a new diagnostic artifact without updating the canonical manifest. Add `--smoke` only when a bounded deterministic generation probe is required. Check Q8_0, digest `c269e5748d11`, context 32768 and GPU placement. There is no cloud/model fallback. Slow startup is an infrastructure issue; retain the error and label any recorded demonstration as historical.

## Build on a fresh environment; reload for normal use

**Build is setup documentation only. Never rebuild the sealed index for E8 or during the demo.** On a fresh reproduction environment without an index, this supported command prepares the pinned MiniLM cache and a new bundle:

```powershell
if ($env:USERNAME -ne 'Admin') { Write-Output "SKIP_PROFILE=$env:USERNAME"; exit 7 }
Set-Location D:\rag-scifact
.\.venv\Scripts\python.exe -m scripts.build_index --config config.yaml --cache-folder artifacts/e2-validation/model-cache --device cuda --batch-size 32
```

The first exact-model fetch needs network access; later operations reuse the local cache. A fresh rebuild is separate reproduction output, not a replacement for historical sealed assets. Frozen runtime commands require the original matching bundle. Its files are `indexes/scifact/index.faiss`, `chunks.jsonl`, `index_manifest.json`.

Reload without corpus embedding or model construction:

```powershell
if ($env:USERNAME -ne 'Admin') { Write-Output "SKIP_PROFILE=$env:USERNAME"; exit 7 }
Set-Location D:\rag-scifact
.\.venv\Scripts\python.exe -c "from app.indexer import load_bundle; b=load_bundle('indexes/scifact'); print(b.index.ntotal, len(b.chunks))"
```

The saved bundle has 10,359 vectors and aligned rows. Reload checks hashes, FAISS type/dimension, normalized vectors, mapping alignment and identities; missing/corrupt/incompatible assets fail explicitly.

## Retrieve and ask with the existing bundle

Use clean benchmark source and frozen assets for `ask`:

```powershell
if ($env:USERNAME -ne 'Admin') { Write-Output "SKIP_PROFILE=$env:USERNAME"; exit 7 }
Set-Location D:\rag-scifact
.\.venv\Scripts\python.exe -m scripts.retrieve --query-id 0 --top-k 5 --cache-folder artifacts/e2-validation/model-cache --local-files-only --device cuda
.\.venv\Scripts\python.exe -m scripts.retrieve --query "What effect did exercise training have on self-reported health status in patients with chronic heart failure?" --top-k 5 --cache-folder artifacts/e2-validation/model-cache --local-files-only --device cuda
.\.venv\Scripts\python.exe -m scripts.ask --query "What effect did exercise training have on self-reported health status in patients with chronic heart failure?" --config config.yaml
```

ID lookup reads original query text without gold metadata. Validation requires 1–2000 trimmed characters and integer Top-K 1–10. `ask` uses only the local pinned cache, counts full chat input through Ollama, reserves 512 output tokens, and retains the largest fitting whole-chunk rank prefix with exclusions logged. JSON contains request ID, answer, semantic status, citations, retrieved IDs/ranks/scores, timings and model ID. Stdout is JSON; stderr holds diagnostics. Exit codes: 0 success, 2 invalid input, 3 invalid model output, 1 infrastructure/configuration error. Normal asks do not alter the seal.

## DEV and generation fixture reproduction

Optional reviewer runs below require clean benchmark source and frozen assets and create new diagnostic artifacts. They are not required to inspect delivered results and were not run during E8. Do not append manifest updates, freeze, index build or official TEST commands.

```powershell
if ($env:USERNAME -ne 'Admin') { Write-Output "SKIP_PROFILE=$env:USERNAME"; exit 7 }
Set-Location D:\rag-scifact
.\.venv\Scripts\python.exe -m scripts.evaluate --split dev --config config.yaml --cache-folder artifacts/e2-validation/model-cache --device cuda
.\.venv\Scripts\python.exe -m scripts.evaluate_generation --fixture data/fixtures/atlas.jsonl --config config.yaml --cache-folder artifacts/e2-validation/model-cache --device cuda
```

DEV uses exactly 100 `manifest.split.dev_ids`, train qrels and one Top-10 retrieval/query. Atlas uses its separate existing six-document index at `indexes/fixtures/atlas`; it never mixes fixtures into SciFact. Q01–Q06 cover permissions, expiry, approval, insufficiency, conflict and F06 injection. Expected answers/grading stay outside runtime. If the fixture bundle is absent, obtain the supplied bundle before the demo.

The explicit freeze already occurred. Test access checks the entire seal before qrels access and again before publication: clean frozen HEAD, config, prompt, runtime/model, retrieval, index, data and canonical provenance. It never auto-freezes. Normal reproduction does not run the official held-out test. Historical evidence remains bound to benchmark source, never to a refrozen delivery commit.

## Recorded results and artifacts

| Measure | Canonical DEV | Official held-out TEST |
| --- | ---: | ---: |
| Queries / denominator | 100 / 100 | 300 / 300 |
| Execution failures | 0 | 0 |
| Recall@5 | 0.7217 | 0.7360 |
| Recall@10 | 0.8017 | 0.8212 |
| MRR@10 | 0.6056 | 0.6134 |
| nDCG@10 | 0.6446 | 0.6624 |
| Retrieval p50, ms | 4.6822 | 4.7941 |
| Quality candidates | 56 | 154 |

The held-out test metrics were slightly higher than the internal dev metrics under the same frozen baseline. No post-test tuning was performed. TEST's 154 quality candidates comprise 51 retrieval misses, 6 partial recalls and 97 ranking failures, **not execution failures**. DEV has 19 misses, 2 partial recalls and 35 ranking failures. The report contains five deterministic DEV cases.

Canonical Atlas F01–F06: **6 passed / 0 failed**, six timing samples. Its p50 retrieval/generation/total are **4.6186 / 954.8444 / 1085.6832 ms**. These are **Atlas fixture timings**, not generation latency over the 300-query SciFact test. Local external LLM API monetary spend: **$0**; electricity/GPU operating cost: **not measured**.

| Evidence | Saved artifact |
| --- | --- |
| Frozen manifest | [manifest.json](artifacts/manifest.json) |
| E5 grounded smoke | [rag_check.json](artifacts/20261008T020100Z-635c5b5b/rag_check.json) |
| E6 Atlas summary | [generation_summary.json](artifacts/20261008T020110Z-08ab60d0/generation_summary.json) |
| E6 answers/citations/context IDs | [generation_run.jsonl](artifacts/20261008T020110Z-08ab60d0/generation_run.jsonl) |
| Canonical DEV | [metrics.json](artifacts/20261008T020122Z-bb3c17a1/metrics.json) |
| DEV failures | [failure_analysis.jsonl](artifacts/20261008T020122Z-bb3c17a1/failure_analysis.jsonl) |
| Official TEST | [metrics.json](artifacts/20261008T020950Z-627514bd/metrics.json) |
| TEST raw retrieval | [retrieval_run.jsonl](artifacts/20261008T020950Z-627514bd/retrieval_run.jsonl) |
| TEST quality candidates | [failure_analysis.jsonl](artifacts/20261008T020950Z-627514bd/failure_analysis.jsonl) |

Exact unrounded measurements remain in these artifacts. Retrieval JSONL contains query-status rows followed by ranked results; errors have explicit status without fake document rows. Metrics bind source/config/split/index/data hashes, denominator, timings and raw-run SHA256. Failure analysis is ordered by Recall@10, MRR@10, then numeric query ID. Strict JSON rejects NaN/Infinity; immutable runs publish summary last. Canonical provenance was checked by independently reconstructing metrics from raw results and evaluator judgments before freeze.

## Software acceptance evidence

Application tests use synthetic data/injected backends and mocks, without public web access or a cloud LLM. Real-data integrity tests require the canonical local corpus. Run in the delivery checkout:

```powershell
if ($env:USERNAME -ne 'Admin') { Write-Output "SKIP_PROFILE=$env:USERNAME"; exit 7 }
Set-Location D:\rag-scifact
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -m compileall -q app scripts tests
.\.venv\Scripts\python.exe -m pip check
```

E8 observed validation on 2026-10-08: **pytest: 243 passed and 300 subtests passed; unittest: 243 tests, OK; targeted Core acceptance: 123 passed and 196 subtests passed; compileall: exit 0; pip check: no broken requirements**.

| Acceptance requirement | Existing tests |
| --- | --- |
| Loader IDs / malformed records / leakage | `test_e1_loader.py`, `test_e1_data_audit.py` |
| Chunk token limit / overlap / title / exact spans | `test_e2_chunker.py` |
| Embedding 384 shape / float32 / norm / finite | `test_e2_embedder.py` |
| Persistence/reload / document dedup/order | `test_e3_indexer.py`, `test_e3_retriever.py` |
| LLM-down / connection / timeout / OOM | `test_e4_generator.py` |
| Citation authority / statuses / separate errors | `test_e5_citations.py`, `test_e5_rag.py` |
| Atlas isolation / F06 / six-case runner | `test_e6_fixture.py`, `test_e6_grader.py`, `test_e6_runner.py` |
| Metric math / denominator / provenance / seal | `test_e7_metrics.py`, `test_e7_runner.py`, `test_e7_manifest.py`, `test_e7_freeze.py` |

Validation ran in the normal Admin environment. The first sandboxed pytest attempt stopped with nine collection errors while the native FAISS DLL failed to initialize (Windows invalid-handle exception); the identical command passed outside that sandbox. No application changes were made to address the sandbox issue.

Earlier canonicalization with Ollama stopped caused E5/E6 connection refusal; E7/freeze then refused stale provenance. Restarting Ollama and completing E5 → E6 → E7 DEV → freeze succeeded, demonstrating fail-closed provenance. Prompt/grader corrections occurred during development before freeze; canonical runs are identified above. MiniLM and persisted indexes are reused locally; reload does not re-embed documents. Readiness/context probes and cold startup can add latency, so p50 scope matters.

## Report, demo, limitations and disclosure

Read the [Core report](reports/E8-core-report.md) and follow the [approximately 8.5-minute demo](docs/E8-demo.md). The demo reloads the existing index, runs a live grounded query on clean benchmark source, and shows recorded Atlas insufficiency/conflict/F06 results and saved DEV/TEST metrics. It never rebuilds or reruns the official benchmark.

Core uses dense retrieval only; no BM25/RRF/reranker comparison or FastAPI. Official TEST measures document retrieval, not generated answer accuracy or scientific support/contradiction labels. F01–F06 is a small targeted fixture, not universal prompt-injection safety proof. Zero execution errors does not mean zero retrieval misses. Failure modes do not prove deep embedding/chunking/lexical causes. Latency varies with hardware/load/cache; cross-backend byte-identical generation is not guaranteed. Ignored evidence and model assets must accompany source delivery.

Sources: DFM-ENGINEERING RAG exercise v1.0 (05/10/2026), Local Ministral backlog v1.0 (05/10/2026), BEIR SciFact and pinned MiniLM. Actual components include Python 3.11, PyTorch, sentence-transformers, Transformers/Hugging Face tokenizers/model stack, FAISS, NumPy, Requests, PyYAML, Ollama and pytest/unittest. No LangChain/LlamaIndex or fine-tuning. Codex/ChatGPT assisted architecture investigation, implementation, tests, review/audit and documentation/report preparation. The author reviewed and changed AI-assisted work afterward, including E7 failure-analysis semantics, the final-test provenance race recheck before publication, canonical provenance verification, regression tests and documentation corrections. The work is not presented as wholly manually authored.
