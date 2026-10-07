# RAG SciFact - Local Ministral

This repository implements the DFM-ENGINEERING SciFact RAG exercise with:

- sentence-transformers/all-MiniLM-L6-v2 for embeddings
- FAISS IndexFlatIP for dense retrieval
- local Ministral 3 3B Instruct for generation
- Windows 11 + NVIDIA GeForce RTX 5070 as the final benchmark/demo machine

## Current status

E0–E6 are implemented on the final Admin machine.

E1 now includes:

- strict reusable SciFact corpus/query/qrels loaders
- runtime-safe query objects that expose query ID + text only
- dataset count/reference/duplicate/malformed-record validation
- SHA256 hashes for all four SciFact source files
- deterministic Random(42) dev/practice split
- exact and near-duplicate cross-split integrity audit
- section-preserving reproducibility manifest merge
- per-run E1 artifacts and logging

E2 provides source-preserving MiniLM token windows and a normalized embedding
wrapper, validated with the real model and the full corpus. E3 adds a validated,
reloadable FAISS bundle and dense document retrieval. E4 adds the concrete
local Ollama/OpenAI-compatible Generator, readiness, structured infrastructure
errors, deterministic smoke validation and runtime manifest locking. E5 adds
grounded ask orchestration, exact local context counting, strict model JSON and
context-only citation validation. E6 evaluates the separate Atlas generation
fixtures; SciFact retrieval evaluation remains pending for E7.

## Final benchmark/demo machine

Verified profile:

- Windows 11 x64, project user/profile: Admin
- Python .venv: 3.11.9
- NVIDIA GeForce RTX 5070
- PyTorch: 2.11.0+cu128
- PyTorch CUDA runtime: 12.8
- torch.cuda.is_available(): true
- PyTorch-reported VRAM: 11.94 GiB
- Ollama: 0.24.0
- model: ministral-3:3b-instruct-2512-q8_0
- Ollama model ID/digest: c269e5748d11
- quantization: Q8_0
- local OpenAI-compatible API: http://127.0.0.1:11434/v1
- observed Ollama runtime context: 32768 tokens
- Ollama processor placement: 100% GPU
- local /v1/chat/completions smoke test: PASS

The backlog reference remains mistralai/Ministral-3-3B-Instruct-2512-BF16.
The final demo profile uses the same Ministral 3 3B Instruct 2512 family/model
through Ollama in Q8_0 because the reference BF16 profile may exceed 12 GB VRAM.

LLM_REVISION is intentionally blank because the Ollama digest is not a
Hugging Face/upstream revision. The Ollama digest is recorded separately as
LLM_RUNTIME_MODEL_DIGEST.

runtime_profile_locked is true after E4 verified the concrete Generator client,
model readiness, structured failure mapping, deterministic local completion and
runtime manifest merge on the final Admin machine.

## Repository layout

~~~text
rag-scifact/
  AGENTS.md
  README.md
  .env.example
  .gitignore
  config.yaml
  requirements.txt
  requirements-dev.txt
  requirements.lock.txt
  app/
    config.py
    logging_utils.py
    models.py
    loader.py
    data_audit.py
    manifest.py
    chunker.py
    embedder.py
    indexer.py
    retriever.py
    context.py
    prompt.py
    generator.py
    citations.py
    rag.py
    evaluator.py
  scripts/
    _common.py
    check_environment.py
    audit_dataset.py
    build_index.py
    retrieve.py
    ask.py
    evaluate.py
    evaluate_generation.py
  data/
    scifact/
    fixtures/
  indexes/
  logs/
  artifacts/
  tests/
  specs/
~~~

## Use the existing Admin environment

~~~powershell
cd D:\rag-scifact
.\.venv\Scripts\Activate.ps1
python --version
python -m scripts.check_environment
pytest -q
~~~

Expected baseline:

~~~text
Python 3.11.x
CUDA available: true
GPU: NVIDIA GeForce RTX 5070
~~~

## Reproduce a clean environment

Python 3.11 is required.

~~~powershell
cd D:\rag-scifact
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
~~~

requirements.lock.txt is the exact snapshot of the verified environment and
includes the PyTorch CUDA 12.8 wheel index required to resolve
torch==2.11.0+cu128.

After install:

~~~powershell
python -m scripts.check_environment
python -m unittest discover -s tests -v
pytest -q
python -m pip check
~~~

requirements.txt and requirements-dev.txt remain broader development manifests.
Use requirements.lock.txt for benchmark/final reproduction.

## Ollama / Ministral

Check the local runtime:

~~~powershell
ollama --version
ollama list
ollama ps
~~~

On a fresh compatible machine, obtain the verified profile with:

~~~powershell
ollama pull ministral-3:3b-instruct-2512-q8_0
~~~

Ollama serves the local OpenAI-compatible API at:

~~~text
http://127.0.0.1:11434/v1
~~~

Example smoke request:

~~~powershell
$body = @{
    model = "ministral-3:3b-instruct-2512-q8_0"
    messages = @(@{ role = "user"; content = "Reply with exactly: OK" })
    temperature = 0
    max_tokens = 512
    seed = 42
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://127.0.0.1:11434/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body
~~~

There is no cloud-LLM fallback for benchmark generation.

## Local configuration

.env.example contains the verified non-secret profile. Copy it only when local
overrides are needed:

~~~powershell
Copy-Item .env.example .env
~~~

.env is gitignored and must not be committed.

Current LLM profile:

~~~dotenv
LLM_PROVIDER=local_openai_compatible
LLM_RUNTIME=ollama
LLM_RUNTIME_VERSION=0.24.0
LLM_REFERENCE_MODEL=mistralai/Ministral-3-3B-Instruct-2512-BF16
LLM_MODEL=ministral-3:3b-instruct-2512-q8_0
LLM_REVISION=
LLM_RUNTIME_MODEL_DIGEST=c269e5748d11
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_CONTEXT_LENGTH=32768
LLM_TEMPERATURE=0
LLM_MAX_NEW_TOKENS=512
LLM_SEED=42
LLM_DTYPE=
LLM_QUANTIZATION=Q8_0
LLM_DEVICE=NVIDIA GeForce RTX 5070
LLM_RUNTIME_PROFILE_LOCKED=true
~~~

## Benchmark baseline

- seed = 42
- embedding = sentence-transformers/all-MiniLM-L6-v2
- embedding dimension = 384
- embedding vectors = normalized float32
- chunk body maximum = 220 MiniLM tokenizer tokens
- overlap = 30 MiniLM tokenizer tokens
- dense index = FAISS IndexFlatIP
- default Top-K = 5
- generation temperature = 0
- generation output limit = 512 new tokens
- local generator family = Ministral 3 3B Instruct 2512

## E1 - Dataset audit and deterministic split

Run:

~~~powershell
python -m scripts.audit_dataset --config config.yaml
~~~

The E1 command:

- strictly validates corpus.jsonl, queries.jsonl, qrels/train.tsv and qrels/test.tsv
- preserves IDs as strings internally
- reports malformed data with file/line context
- rejects duplicate IDs and duplicate qrel query/document pairs
- validates every qrel query/document reference
- preserves multiple relevant documents per query
- never exposes source query metadata to the runtime query object
- validates the canonical SciFact counts
- requires qrels headers to be exactly query-id / corpus-id / score in that order
- rejects blank qrels rows and any qrels data row that does not contain exactly three tab-separated fields
- validates the four canonical raw-byte SHA256 hashes, not just counts
- creates the required Random(42) split from every unique query ID present in train qrels: 100 dev + 709 practice
- rejects project/config seed values other than 42
- audits exact/near duplicate query text across practice/dev/test
- writes artifacts/manifest.json
- writes per-run artifacts under artifacts/<run_id>/

Canonical current counts:

~~~text
corpus documents        5183
queries                 1109
train query IDs          809
test query IDs           300
train qrel rows          919
test qrel rows           339
dev                      100
practice                 709
~~~

Current file SHA256 values:

~~~text
corpus.jsonl
dec31c8182f3d744c7d2c09423756fd1d17cbef75808db13ba01cc0aab4d1ac6

queries.jsonl
8ff84a7c903f722981cd8d595c022660140c51867b27608a6d4910db86080313

qrels/train.tsv
a53f2114831916c096b6c37d9e54da68cef4efdcdbd5ed46533601af972acf1d

qrels/test.tsv
0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c
~~~

### Exact / near-duplicate audit policy

Policy version: query_overlap_v1.

Exact canonical normalization:

- Unicode NFKC
- casefold
- whitespace collapse
- punctuation and numbers remain significant

Near-duplicate policy:

- cross-split pairs only
- canonical-exact pairs are excluded from near count
- Unicode NFKC + casefold word tokens
- maximum token count must be at least 5
- token-level Levenshtein distance
- maximum distance = max(1, floor(0.10 * max_token_count))
- no stemming
- no stopword removal
- no semantic embeddings

The policy was fixed before benchmark tuning. Final-test overlaps are reporting
only and must never be used to tune retrieval, prompts, thresholds, model
selection, query removal, split membership, or evaluation denominators.

Current aggregate result:

~~~text
pairs compared            313600
raw exact pairs                2
canonical exact pairs          2
near duplicate pairs          98

practice vs dev near          25
practice vs test near         66
dev vs test near               7
~~~

The artifact intentionally reports aggregate overlap counts only. It does not
publish final-test query text or IDs.

## E1 artifacts

Canonical manifest:

~~~text
artifacts/manifest.json
~~~

Each run also writes:

~~~text
artifacts/<run_id>/
  dataset_audit.json
  manifest.json
  config.snapshot.yaml
~~~

The manifest stores:

- git commit and dirty state
- dataset counts and SHA256 hashes
- seed 42
- all 100 dev IDs
- all 709 practice IDs
- aggregate query-overlap audit policy/results
- known E0 runtime/hardware metadata
- placeholders for later E2-E7 sections

E1 manifest updates are conflict-safe: an existing manifest with different
dataset hashes or split identity is rejected instead of silently overwritten.
Unrelated later-stage manifest sections are preserved.

The artifacts/ directory is intentionally gitignored. artifacts/manifest.json
is therefore a generated delivery artifact, not a tracked source file. For a
freeze, commit source/config first, make sure the working tree is clean, then
rerun python -m scripts.audit_dataset --config config.yaml. The generated
manifest should then record the frozen source commit with git_dirty=false.
Include that manifest separately in the final delivery bundle.

If a manifest already records final_test_frozen=true, a later audit may only merge it when the repository is still on the same recorded Git commit and the working tree is clean. Dirty state, a different commit, or an invalid frozen identity is rejected instead of silently retaining the frozen flag.

## E2 - MiniLM chunking and embedding

Construct one shared runtime for corpus and queries:

~~~python
from app.config import load_config
from app.loader import load_corpus
from app.embedder import MiniLMEmbedder
from app.chunker import MiniLMChunker

config = load_config()
embedder = MiniLMEmbedder(config.embedding, device="cpu")
chunker = MiniLMChunker.from_embedder(embedder)
document = next(iter(load_corpus(config.paths.corpus).values()))
chunks = chunker.chunk_document(document)
vectors = embedder.encode_chunks(chunks)  # (N, 384), normalized float32
query_vector = embedder.encode_query("A scientific question")  # (384,)
~~~

Use `local_files_only=True` and, if needed, `cache_folder=...` to reuse a local
exact-model cache. Construction loads the model; module imports do not.
Routine tests use injected runtimes and need no public network access.

Body windows are chosen only at stable MiniLM WordPiece word boundaries. Each
exact source substring is retokenized and verified to match the original document
token slice, so the standalone body is never more than 220 MiniLM tokens. Adjacent
chunks are selected so the last 30 standalone token IDs of the left chunk exactly
equal the first 30 token IDs of the right chunk.

IDs are `<doc_id>:<token_start>-<token_end>`, with half-open token and character
spans. Evidence is exactly `document.text[char_start:char_end]`; no tokenizer
decoding reconstructs it. Empty/whitespace-only bodies produce zero chunks.

Embedding text is `embedding_title + "\n\n" + chunk.text`, or only the body
when the embedding title is empty. The original title remains in metadata.
If the actual combined input exceeds the runtime limit, the chunker retains a
source title prefix ending at a title-token offset, reducing it until the
actual combined tokenization (including special tokens) fits. The body source
span is preserved. Each truncation logs document ID, original/retained title
token counts, input limit and body token count. Oversized embedding inputs are
rejected instead of being silently truncated by SentenceTransformer.

Observed validation on 2026-10-06 with the exact pinned all-MiniLM-L6-v2 runtime:

| Full-corpus check | Result |
| --- | ---: |
| Documents / chunks | 5,183 / 10,359 |
| Standalone body tokens, min / max | 31 / 220 |
| Chunks over 220 | 0 |
| Actual 30-token overlap violations | 0 |
| Source-slice token identity violations | 0 |
| Duplicate IDs / empty bodies | 0 / 0 |
| Title truncations (per chunk) | 546 |
| Maximum actual embedding input / effective runtime limit | 256 / 256 |

The pinned Hugging Face revision is:

```text
1110a243fdf4706b3f48f1d95db1a4f5529b4d41
```

The real CPU smoke encoded 16 corpus chunks into `(16, 384)` finite float32
vectors; norms ranged from 0.9999999404 to 1.0. Single versus batch maximum
absolute difference was 5.96e-8. The cached validation rerun uses
local_files_only=True with the pinned revision. Observed counts are validation
results, not constants in application logic. Generated reports/logs and the
exact-model cache are under `artifacts/e2-validation/` (gitignored).

Both runtime objects expose `runtime_metadata()` for E3 manifest recording.
E2 does not modify the E1 manifest, its freeze guards, or index status.
FAISS/index building belongs to E3 below.

E2 and E3 code and tests were produced with Codex assistance. They use existing
SentenceTransformer, Transformers/tokenizers, PyTorch and NumPy libraries;
no model was trained or fine-tuned. Future human changes should be disclosed
in the delivery report alongside this assistance. E2 was subsequently changed
to enforce stable WordPiece boundaries, standalone token identity/overlap checks,
and the verified pinned embedding revision. E3 uses direct FAISS APIs.

## E3 - Persistent dense index and document retrieval

Build from the full canonical corpus (no queries or qrels enter the build):

~~~powershell
python -m scripts.build_index --config config.yaml
~~~

The verified offline build on this machine used the existing E2 cache:

~~~powershell
python -m scripts.build_index --config config.yaml --cache-folder artifacts/e2-validation/model-cache --local-files-only --device cuda --batch-size 32
~~~

The model remains `sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Batch size/device/cache flags are
runtime controls; they do not change the benchmark chunk/model contract.
Canonical corpus JSONL order followed by E2 chunk order determines vector positions.

The default bundle is:

~~~text
indexes/scifact/
  index.faiss
  chunks.jsonl
  index_manifest.json
~~~

`chunks.jsonl` contains one UTF-8 row per vector, with contiguous position,
document/chunk IDs, exact evidence, original/embedding titles, token/character
spans, body count and title-truncation state. The index manifest records schema,
UTC timestamp, run ID, IndexFlatIP/inner-product identity, dimension/counts,
corpus SHA256/count, pinned embedding identity, E2 policies/input limit, build
device, library versions and both data file hashes/sizes.

Files are staged first and the manifest is published last. Reload checks hashes,
FAISS type/dimension/count/vectors, mapping alignment, unique IDs and runtime
compatibility. A missing, partial, corrupt or incompatible bundle fails explicitly.
The reusable application APIs accept a bundle path; both CLIs offer `--bundle`.
E3 writes its own bundle manifest and `artifacts/<run_id>/index_build.json` plus
config snapshot. It leaves `artifacts/manifest.json`, E1 sections and freeze guards
unchanged; the index bundle manifest is the E3 build record.

Reload directly without constructing MiniLM or reading the corpus:

~~~python
from app.indexer import load_bundle
bundle = load_bundle("indexes/scifact")
print(bundle.index.ntotal, len(bundle.chunks))
~~~

Retrieve from the existing persisted bundle; only the query is embedded:

~~~powershell
python -m scripts.retrieve --query "Does physical activity affect cardiovascular health?" --top-k 5
python -m scripts.retrieve --query-id 0 --top-k 5
~~~

To reuse the verified cache offline, append
`--cache-folder artifacts/e2-validation/model-cache --local-files-only --device cuda`
to either retrieval command. `--query` and `--query-id` are mutually exclusive.
ID lookup uses the original query text through the runtime-safe loader, without
qrels or gold metadata. JSON results go to stdout; diagnostics/logs go to stderr.

E3 supports `retrieval.mode: dense` only; both build and retrieval CLIs reject
other configured modes instead of silently running dense behavior. When
`--top-k` is omitted, retrieval uses `retrieval.top_k` from the loaded config.
An explicit `--top-k` overrides that default but must remain within
`1..retrieval.max_top_k`; `max_top_k` itself is validated against the benchmark
hard cap of 10. The trimmed query must be 1..2000 characters.

The retriever begins with `min(ntotal, max(32, 4 * top_k))` chunk candidates,
doubles until enough unique documents are available, and completes ties at the
document cutoff. Ordering is score descending, then vector position ascending.
Each document keeps its highest-scoring chunk, with ranks 1..K. A small valid
index returns all available unique documents; an empty index is an infrastructure
error. Scores are retrieval similarities, not correctness probabilities.

Observed full-corpus build on 2026-10-06:

| Check | Result |
| --- | ---: |
| Documents / chunks / vectors / FAISS ntotal / mapping rows | 5,183 / 10,359 / 10,359 / 10,359 / 10,359 |
| Dimension / dtype | 384 / float32 |
| L2 norm range | 0.9999999404–1.0 |
| Build device / batch size | cuda:0 (RTX 5070) / 32 |
| Build duration, including load/chunk/embed/persist/reload | 23.29 seconds |
| Duplicate chunk IDs | 0 |
| index.faiss bytes / chunks.jsonl bytes | 15,911,469 / 12,534,746 |

Data SHA256 values for this build:

~~~text
index.faiss
cc96c582d39b324a78e6fd7f1ce972bbe7f49b6cae8f062020834fc921d556c7
chunks.jsonl
877dd3a648c8b791487ca2a373953e0c4479bf2e2df8440c8d234afb4b583af1
~~~

Reload reproduced vectors and mapping exactly. Both query ID 0 and the arbitrary
query above returned five unique documents, continuous ranks, descending scores
and exact stored evidence. In-memory and reload retrieval results matched.
Guarded validation confirmed zero corpus loads/embeddings during reload/retrieval.
Reports/stdout from these checks are under `artifacts/e3-validation/`; build run
`20261006T083645Z-c85c7abf` has the full report. These are structural checks;
retrieval metrics and generation remain later work.

## E4 - Local Ministral runtime

E4 keeps generation behind `app.generator.Generator` and supports only the
verified local profile for this benchmark: `local_openai_compatible` + Ollama.
The concrete client rejects non-loopback base URLs and never falls back to a
cloud provider or a different model. The response model ID must exactly match
the configured model. Readiness first verifies model listing and then performs
a bounded local generation probe, so a merely installed-but-unusable model does
not count as ready.

Readiness only:

~~~powershell
python -m scripts.check_generator --config config.yaml
~~~

Acceptance smoke plus E4 runtime-manifest merge:

~~~powershell
python -m scripts.check_generator --config config.yaml --smoke --update-manifest --output-dir artifacts\e4-validation
~~~

The smoke sends `Reply with exactly: OK` through the same Generator used by the
application with temperature 0, max 512 new tokens and seed 42. On the verified
Admin machine it returned exactly `OK`; Ollama then reported the configured
Ministral model at 100% GPU with a 32768-token context.

Generator infrastructure failures remain explicit and separate from future RAG
semantic statuses: connection refused, connect timeout, read timeout, missing
model, GPU OOM, generic runtime HTTP failure, malformed response, and unexpected
model identity all raise structured Generator errors.

The runtime lock is manifest-based rather than source-hardcoded. The manifest
stores the exact locked profile plus a deterministic profile SHA256. With
`runtime_profile_locked=true`, the current config must match that manifest
profile before the Generator is used. This prevents silent drift without
permanently tying the codebase to one model.

To experiment with another local Ollama model/profile, set
`runtime_profile_locked=false`, change the LLM config, then verify it without
changing the canonical lock:

~~~powershell
python -m scripts.check_generator --config config.yaml --smoke --output-dir artifacts\e4-candidate
~~~

When that candidate is intentionally chosen as the new benchmark profile, keep
the config unlocked and explicitly replace the manifest lock:

~~~powershell
python -m scripts.check_generator --config config.yaml --smoke --relock-runtime-profile --output-dir artifacts\e4-validation
~~~

After relocking, set `runtime_profile_locked=true` again for normal benchmark
runs. An ordinary `--update-manifest` only refreshes verification for the same
already-locked profile and cannot silently replace it. Relocking is forbidden
after `final_test_frozen=true`.

E4 also reads Ollama native metadata (`/api/version`, `/api/tags`, `/api/ps`) and
checks the observed runtime version, model digest, quantization and effective
context against the declared profile before provenance can be updated.

`runtime_profile_locked` is separate from the final test freeze. The final test
remains unfrozen until `final_test_frozen=true`. Because the global manifest
records Git identity, rerun the E4 smoke/manifest command after the E4 commit
when a clean-commit manifest snapshot is required.

## E5 - Grounded ask and prompt provenance

Reuse the persisted SciFact bundle and locally cached pinned MiniLM model:

~~~powershell
python -m scripts.ask --query "What evidence links physical activity and cardiovascular health?" --config config.yaml
~~~

Omitting `--top-k` uses `retrieval.top_k`; an explicit flag overrides it within
the configured maximum. JSON goes to stdout, errors/logs to stderr and run logs.
Exit codes: 0 success, 2 invalid request, 3 invalid model output, 1 infrastructure
or configuration failure. Public retrieved entries contain document ID, rank and
similarity score; they do not expose full source text. Citations contain quotes.
Normal asks never mutate the canonical manifest or download model/tokenizer assets.
The factory uses `local_files_only=True` and reuses
`artifacts/e2-validation/model-cache` when present, otherwise the standard local
Hugging Face cache. Prepare the exact pinned MiniLM cache explicitly before asking
on a clean machine. The existing E2/E3 cache controls can be used for that setup.

`app.rag.build_rag_pipeline` validates the canonical E4 runtime lock before index,
embedder, retriever or Generator construction. Reuse this factory for future E6/API
callers. Retrieval runs once, before token counting and final answer generation.
The immutable `ContextBundle.included` snapshot is the only citation authority.

Budgeting counts the full rendered system/user chat with Ollama `/api/chat`,
using `prompt_eval_count` from a deterministic one-token probe. This is a local
runtime call after retrieval, not a MiniLM/character estimate. The 512-token output
reserve must fit with input inside the configured 32768-token window. Full Top-K
is counted once; on overflow every remaining rank prefix is checked (Top-K is
bounded at 10) and the largest fitting whole-chunk prefix is selected without
assuming tokenizer counts are monotonic. Lower-ranked chunks cannot replace an
excluded earlier chunk. Even an oversized base prompt fails explicitly.

The explicit system prompt treats document titles/text as untrusted data. User
question and evidence are deterministic JSON; scores are excluded. Prompt version
is `rag-grounded-json-v4`; `app.prompt.prompt_identity()` exposes its static SHA256,
schema/serialization versions and context policy. The hash excludes request data.
The status decision first filters to evidence directly relevant to the requested
fact, then gives unresolved conflicts precedence over insufficiency and answered
results. Unrelated contradictions elsewhere in retrieved context do not change status.

E5 requests Ollama JSON mode (`response_format: {"type": "json_object"}`);
model output must still be exactly one object with `status`, `answer`, `citations`.
The parser rejects extra/missing keys, duplicate keys, non-standard constants,
fences, surrounding prose and wrong/empty values. JSON mode is a generation
constraint, not output repair. Any invalid citation rejects
the entire output; there is no repair/retry. Matching permits only whitespace
normalization, preserving case, punctuation and numbers. Identical duplicate
citations are rejected. ANSWERED needs a valid citation; INSUFFICIENT_EVIDENCE
allows zero; CONFLICTING_EVIDENCE needs valid citations from two distinct included
documents. These checks establish structural grounding; semantic correctness and
the canonical F01–F06 benchmark belong to E6.

Verify real RAG, observed runtime identity, and equality between native prompt
counting and OpenAI-compatible usage for the same fixed chat:

~~~powershell
python -m scripts.check_rag --config config.yaml
python -m scripts.check_rag --config config.yaml --update-manifest
~~~

Only the explicit update writes E5 prompt/verification provenance to
`artifacts/manifest.json`; it preserves every E1–E4 section and runtime lock.
The report is `artifacts/<run_id>/rag_check.json`. Refresh with `--update-manifest`
after committing E5 on a clean tree to record that source commit. Final-test frozen
manifests reject prompt/config/source drift; E5 never sets the final freeze.
Canonical updates use and bind one fixed SciFact smoke question about the effect of
exercise training on self-reported health status in chronic heart failure. The
question retrieves a concise top-ranked conclusion and still exercises live
retrieval, context budgeting, generation and verbatim citation validation. Custom
`--query` values remain available for diagnostics, but cannot update the canonical
manifest. The verification artifact records the exact query, `source_git_commit`
and `source_git_dirty`; manifest merge rejects a different query or source state.
Ordinary traces/logs contain IDs, versions, counts, truncation and timings, not
full prompts or evidence texts. Check artifacts include the public smoke response.

E4/E5 code and tests were produced with Codex assistance using direct Requests,
standard-library JSON/dataclasses/hashlib, and the existing E2/E3 components.
No model was trained or fine-tuned, and no generation framework was introduced.

An earlier E5 development smoke on 2026-10-07 reused the existing 10,359-vector
bundle unchanged: five documents in context, 1,693 input tokens, no exclusions,
two valid citations, ANSWERED, and the real Ministral model ID. Native and
OpenAI-compatible prompt counts matched at 490 for the fixed check chat. Normal
ask preserved the canonical manifest and all bundle file hashes. The explicit
update preserved dataset/split/embedding/chunking/index/retrieval/llm/freeze
sections, including every E4 lock field, while recording E5 prompt provenance.

This smoke validates structure, not answer entailment. The live answer misstated
which activity group had 20% lower mortality despite quoting the source correctly;
E6 must assess this semantic error. Earlier development probes also demonstrated
that fences, shortened IDs and altered quote punctuation fail explicitly. Baseline
E5 does not repair them. A native count probe adds local inference latency, and
overflow may require additional probes.

The prompt-v4 canonical smoke now uses the fixed chronic-heart-failure question
described above. Development run `20261007T134036Z-22090835` retrieved the concise
top-ranked conclusion in `40817021:380-463`, used five context chunks / 1,713 input
tokens, returned one exact citation with ANSWERED, and matched native/completion
prompt counts at 790/790. This remains a structural smoke, not an entailment
benchmark; E6 owns semantic fixture grading.

## E6 - Atlas generation fixtures

`data/fixtures/atlas.jsonl` contains the six exercise-prescribed fictional Atlas
documents F01–F06. These are test documents, not real company policies. The separate
bundle is `indexes/fixtures/atlas`; SciFact's bundle is never rebuilt or modified.
The fixture helper validates exact corpus content/order/schema, uses the existing
loader/MiniLM chunker/embedder/FAISS primitives, and checks SciFact hashes before
and after the explicit fixture build. Model downloads are prohibited when
`--local-files-only` is selected. Normal evaluation uses an existing bundle and
the E5 factory's offline query embedder; missing/corrupt fixture bundles fail.

~~~powershell
python -m scripts.evaluate_generation --config config.yaml --build-index --cache-folder artifacts/e2-validation/model-cache --local-files-only --device cuda
python -m scripts.evaluate_generation --config config.yaml
~~~

Options include `--fixture`, `--bundle`, `--batch-size`, `--cache-folder`,
`--local-files-only`, `--device`, `--build-index`, and `--update-manifest`.
Omitting `--build-index` never rebuilds implicitly. Top-K comes from config.
The shared `build_rag_pipeline_from_bundle` preserves E4 lock/E5 freeze checks
and composes the same production retriever and Generator against Atlas.

`app.evaluator` owns six ordered questions and all expected statuses/facts/forbidden
claims. Only each question string enters RAG; grading labels never enter corpus,
index, prompt, retrieval or generation inputs. Suite identity is
`atlas-generation-v1` with grading policy `atlas-en-vi-predicates-v4`; its SHA256
binds static cases, EN/VI predicate rules and grading implementation. Run IDs,
timestamps, answers, timings and retrieval results are excluded.

The deterministic grader checks creator/member/admin-transfer permissions, 30-day
creation-based expiry and owner shortening, team-lead approval and Operations'
two-working-day activation/rejection behavior, unspecified password requirements,
unresolved 50/100 MB notice conflict, and 08:00–17:00 Monday–Friday support. It checks
both answer facts and aggregate source-valid citation coverage. Q06 rejects
affirmative 24/7 or secret-disclosure claims while allowing safe discussion of
the malicious note. The fixture contains no API key. These are explicit bilingual
predicates, not a general entailment model; unfamiliar paraphrases can be rejected.
Failures retain distinct retrieval/context, status, semantic, unsupported-claim,
citation-coverage, conflict, injection, invalid-output and infrastructure codes.
Unexpected evaluator/programming faults use a separate evaluator-error code rather
than being mislabeled as infrastructure. One failed case never removes later cases,
and a case is never retried within a suite.

Each run writes UTF-8/LF `artifacts/<run_id>/generation_run.jsonl` (Q01→Q06) and
`generation_summary.json`, published atomically per file with the summary last.
Strict JSON rejects NaN/Infinity. Summary records source commit/dirty state captured
before evaluation, raw config hash, effective retrieval settings, prompt/runtime/
suite/corpus/index identities, a deterministic snapshot/hash of the resolved
effective AppConfig, observed local runtime and SciFact integrity hashes.
Latency p50 includes all returned responses, even semantic failures; exceptions
have no fabricated timing and remain in the six-case failure denominator.
Exit codes are 0 all-pass, 2 failed fixture cases, and 1 setup/provenance failure.

Normal evaluation never changes `artifacts/manifest.json`. Canonical E6 updates
require all six cases to pass, a clean committed source tree, unchanged source
state, and E5 verification from that same clean commit. Existing E1–E5 sections,
runtime lock and freeze fields are preserved. A stale E5 record permits normal
development runs but blocks canonical E6 updates. E6 never sets the final freeze.

After committing under your own local-ignore policy, finalize in this order:

~~~powershell
python -m scripts.check_rag --config config.yaml --update-manifest
python -m scripts.evaluate_generation --config config.yaml --cache-folder artifacts/e2-validation/model-cache --device cuda --update-manifest
~~~

The canonical E6 command reuses the already validated separate Atlas fixture index.
Use `--build-index --local-files-only` only when that fixture bundle is absent or
intentionally being rebuilt; canonicalization does not require an unnecessary
rebuild and never rebuilds the SciFact index.

E6 source/tests were produced with Codex assistance using existing production
primitives and standard-library deterministic grading. No LLM judge, web service,
model training, fixture-specific runtime shortcuts or E7 metrics were introduced.

Observed development validation on 2026-10-07 built six Atlas chunks using cached
MiniLM on CUDA and exercised Q01–Q06 through Ollama Ministral Q8_0. The immutable
original run `artifacts/20261007T100159Z-a59064c7` scored 2/6 under grader v1.
Offline regrades preserved those model outputs while fixing evaluator false negatives:
v2 scored 3/6 and v3 scored 4/6 with zero generator calls. The audit then added
independent artifact reconstruction/regrading, effective-config provenance and the
v4 error/grading policy.

Prompt v2 fixed the genuine Q04 insufficiency and Q05 conflict behavior, but a new
development suite exposed over-broad conflict selection when unrelated retrieved
documents disagreed. Prompt v3 added a relevance gate before status selection and
produced a 6/6 diagnostic run. Prompt v4 then aligned the conflict citation wording
with grader v4: both incompatible source assertions require exact citation coverage,
while the explanation that no supplied rule resolves them is not given a stricter
fixture-only quote requirement. A fresh E5 verification for prompt v4 passed with
native/completion prompt counts 790/790. The subsequent live run
`artifacts/20261007T132021Z-4a226142` passed all six cases under grader v4 using
prompt SHA256 `1b9b14f2563f9cfe88ebefec75c85cb69ff3480304c0d6058546e0df2fff6702`
and suite SHA256 `7837ab88900fe677db247675e81ef0f1fb2e5107072855f0c636aaaea49f5361`.
Each development suite performs exactly one generation per case; a new suite is run
only after changing the prompt contract. SciFact bundle hashes remain unchanged.
The canonical manifest is intentionally unchanged while the current E6 source is
dirty/uncommitted; canonical E5/E6 provenance must be regenerated on the final clean
commit before E6 is recorded.

## Tests

~~~powershell
python -m unittest discover -s tests -v
pytest -q
python -m compileall -q app scripts tests
python -m pip check
~~~

E1 tests cover strict loading, malformed records, duplicate IDs, qrel reference
validation, multi-document relevance, metadata isolation, deterministic split,
real-data counts/hashes, overlap policy, and manifest merge/conflict behavior.
E4 tests cover flexible unlocked profiles, manifest-authoritative locking,
explicit relock transitions, canonical profile hashing, loopback-only routing,
observed Ollama metadata, deterministic request parameters, readiness,
transport/timeouts, model unavailable/OOM, malformed responses and model
mismatch.

## Commands not implemented yet

This remains a scaffold command until E7 is implemented:

~~~powershell
python -m scripts.evaluate --split dev --config config.yaml
~~~

Do not treat scaffold output as a benchmark result.
