# RAG SciFact - Local Ministral

This repository implements the DFM-ENGINEERING SciFact RAG exercise with:

- sentence-transformers/all-MiniLM-L6-v2 for embeddings
- FAISS IndexFlatIP for dense retrieval
- local Ministral 3 3B Instruct for generation
- Windows 11 + NVIDIA GeForce RTX 5070 as the final benchmark/demo machine

## Current status

E0 and E1 are implemented on the final Admin machine.

E1 now includes:

- strict reusable SciFact corpus/query/qrels loaders
- runtime-safe query objects that expose query ID + text only
- dataset count/reference/duplicate/malformed-record validation
- SHA256 hashes for all four SciFact source files
- deterministic Random(42) dev/practice split
- exact and near-duplicate cross-split integrity audit
- section-preserving reproducibility manifest merge
- per-run E1 artifacts and logging

E2-E7 are not implemented yet. build-index, retrieve, ask, evaluate, and
evaluate-generation remain scaffold commands until their owning epics are done.

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

runtime_profile_locked remains false until E4 implements and verifies the
concrete Generator client, readiness/error mapping, and final runtime manifest.

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
LLM_RUNTIME_PROFILE_LOCKED=false
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

## Commands not implemented yet

These remain scaffold commands until E2-E7 are implemented:

~~~powershell
python -m scripts.build_index --config config.yaml
python -m scripts.retrieve --query "..." --top-k 5
python -m scripts.ask --query "..." --top-k 5
python -m scripts.evaluate --split dev --config config.yaml
python -m scripts.evaluate_generation --fixture data/fixtures/atlas.jsonl
~~~

Do not treat scaffold output as a benchmark result.
