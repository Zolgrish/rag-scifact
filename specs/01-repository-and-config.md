# 01 - Repository, configuration and observability

## Repository structure

Target structure:

~~~text
rag-scifact/
  AGENTS.md
  README.md
  requirements.txt
  .env.example
  config.yaml
  app/
    __init__.py
    config.py
    models.py
    loader.py
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
    api.py          # Junior
    bm25.py         # Junior
    reranker.py     # Junior
  scripts/
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

The exact package layout may evolve, but module boundaries must remain explicit. Benchmark logic must not exist only inside a notebook or CLI.

E0 structure acceptance additionally requires:

- all scaffolded core modules import without error in the intended E0 environment
- README documents the repository structure and environment/setup commands
- benchmark/application logic remains in reusable modules rather than only in notebooks or CLI entry points

## Python baseline

The exercise requires Python 3.11.

The current repository `.venv` on the final Admin machine reports Python 3.11.9 and satisfies the Python baseline. Dependencies are installed and the exact environment snapshot is stored in `requirements.lock.txt`.

~~~powershell
.\.venv\Scripts\Activate.ps1
python --version
python -m scripts.check_environment
python -m pip check
~~~

Acceptance: the benchmark environment reports Python 3.11.x. PyTorch/CUDA must see the RTX 5070. For a fresh machine/environment, README retains `py -3.11 -m venv .venv` and installs `requirements.lock.txt`; the lockfile includes the CUDA 12.8 PyTorch index required to resolve the verified `torch==2.11.0+cu128` wheel. `requirements.txt` remains the broader development install manifest.

Use repo-relative paths with pathlib.Path where possible. Do not bake user-specific absolute paths into portable config.

## Central configuration

Benchmark settings must come from one config layer.

Recommended config.yaml shape:

~~~yaml
project:
  seed: 42

paths:
  corpus: data/scifact/corpus.jsonl
  queries: data/scifact/queries.jsonl
  qrels_train: data/scifact/qrels/train.tsv
  qrels_test: data/scifact/qrels/test.tsv
  index_dir: indexes
  log_dir: logs
  artifact_dir: artifacts

embedding:
  model_id: sentence-transformers/all-MiniLM-L6-v2
  chunk_size_tokens: 220
  chunk_overlap_tokens: 30
  normalize: true
  dtype: float32

retrieval:
  mode: dense
  top_k: 5

llm:
  provider: local_openai_compatible
  runtime: ollama
  runtime_version: "0.24.0"
  reference_model_id: mistralai/Ministral-3-3B-Instruct-2512-BF16
  model_id: ministral-3:3b-instruct-2512-q8_0
  revision: ""
  runtime_model_digest: c269e5748d11
  base_url: http://127.0.0.1:11434/v1
  context_length: 32768
  temperature: 0
  max_new_tokens: 512
  seed: 42
  dtype: ""
  quantization: Q8_0
  device: NVIDIA GeForce RTX 5070
  connect_timeout_s: 10
  read_timeout_s: 120
  runtime_profile_locked: false
~~~

The final model fields must describe the runtime actually used. Do not leave BF16 or quantization=none in the manifest if the demo actually uses a quantized profile.

## Environment overrides

Suggested .env.example:

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
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE_TOKENS=220
CHUNK_OVERLAP_TOKENS=30
TOP_K=5
~~~

Do not commit real secrets.

## Manifest contract

manifest.json is the reproducibility record. It must include at least:

~~~json
{
  "git_commit": "...",
  "created_at": "...",
  "dataset": {
    "corpus_count": 5183,
    "query_count": 1109,
    "train_query_count": 809,
    "test_query_count": 300,
    "hashes": {}
  },
  "split": {
    "seed": 42,
    "dev_ids": [],
    "practice_ids": []
  },
  "chunking": {
    "tokenizer": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size_tokens": 220,
    "overlap_tokens": 30
  },
  "embedding": {},
  "index": {},
  "llm": {},
  "runtime": {},
  "hardware": {},
  "prompt_version": "...",
  "freeze": {}
}
~~~

Hardware metadata should include OS, GPU name, VRAM, Python version, and relevant CUDA/PyTorch/runtime versions. Do not collect unrelated personal information.

artifacts/manifest.json is a generated delivery artifact and is intentionally excluded by the repository artifacts/* gitignore rule. This avoids a self-referential source-commit cycle because the manifest records the source commit it describes. For a freeze/delivery: first commit the source/config changes, ensure the working tree is clean, then rerun the audit/build so the generated manifest records that frozen commit with git_dirty=false. The final delivery bundle must include the generated manifest.json separately even though it is not tracked in the source commit. After final_test_frozen=true, manifest updates are permitted only when the current Git commit matches the recorded frozen commit and the working tree is clean; otherwise the merge must fail.

## Logging contract

Every benchmark/build job has a run_id. Every ask/API request has a request_id.

Log:

- timestamp
- config identity/hash
- retrieval mode
- model ID
- retrieved doc/chunk IDs
- context IDs actually sent to the model
- truncation events
- retrieval/generation/total timing
- semantic status or infrastructure error

Exceptions must be logged with a useful message and stack trace, then propagated or mapped to the explicit infrastructure-error path. Logging must not swallow an exception or convert it into `INSUFFICIENT_EVIDENCE`.

Do not swallow exceptions. Preserve diagnostic information in developer logs, while keeping secrets out of public artifacts.

## Artifact layout

Recommended:

~~~text
artifacts/
  <run_id>/
    manifest.json
    config.snapshot.yaml
    retrieval_run.jsonl
    metrics.json
    generation_run.jsonl
~~~

Index and metadata mapping must be versioned together so a mapping from one index cannot be loaded with another.

## Target commands

~~~powershell
python -m scripts.audit_dataset --config config.yaml
python -m scripts.build_index --config config.yaml
python -m scripts.retrieve --query "..." --top-k 5
python -m scripts.retrieve --query-id 0 --top-k 5
python -m scripts.ask --query "..." --top-k 5
python -m scripts.evaluate --split dev --config config.yaml
python -m scripts.evaluate_generation --fixture data/fixtures/atlas.jsonl
pytest -q
~~~

Junior:

~~~powershell
uvicorn app.api:app --host 127.0.0.1 --port 8080
~~~

The verified local runtime is Ollama on the Admin RTX 5070 machine. README must document `ollama pull ministral-3:3b-instruct-2512-q8_0`, the Ollama local service, model digest `c269e5748d11`, and the OpenAI-compatible endpoint on port 11434.
