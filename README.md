# RAG SciFact - Local Ministral

Project th?c h?nh RAG tr?n SciFact BEIR. Baseline d?ng `sentence-transformers/all-MiniLM-L6-v2`, FAISS `IndexFlatIP`, v? local Ministral 3 3B Instruct.

## Tr?ng th?i hi?n t?i

Repository ?ang ? cu?i E0 v? m?i tr??ng/project skeleton. M?i tr??ng cu?i ?? ???c c?i v? verify tr?n ch?nh m?y benchmark/demo n?y; E1-E7 v?n ch?a ???c tri?n khai v? c?c CLI `build-index`, `retrieve`, `ask`, `evaluate`, `evaluate-generation` hi?n v?n l? scaffold.

### M?y benchmark/demo cu?i

- Windows 11 x64, user ch?y project: `Admin`
- NVIDIA GeForce RTX 5070, PyTorch th?y kho?ng 11.94 GiB VRAM
- Python `.venv`: 3.11.9
- PyTorch: 2.11.0+cu128
- CUDA runtime c?a PyTorch: 12.8
- Ollama: 0.24.0
- Local model: `ministral-3:3b-instruct-2512-q8_0`
- Ollama model ID/digest: `c269e5748d11`
- Quantization: `Q8_0`
- Ollama API: `http://127.0.0.1:11434/v1`
- Runtime context quan s?t b?ng `ollama ps`: 32768 tokens
- `ollama ps` ?? x?c nh?n model ch?y `100% GPU`; API `/v1/chat/completions` ?? tr? response h?p l?

Model backlog tham chi?u v?n l? `mistralai/Ministral-3-3B-Instruct-2512-BF16`. V? BF16 c? th? v??t VRAM 12 GB, profile demo ?? verify d?ng c?ng family/model 3B Instruct 2512 nh?ng ? Q8_0. ??y l? thay ??i runtime/quantization, kh?ng ??i model family.

## C?u tr?c project

```text
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
```

## D?ng m?i tr??ng hi?n t?i tr?n m?y Admin

```powershell
cd D:\rag-scifact
.\.venv\Scripts\Activate.ps1
python --version
python -m scripts.check_environment
pytest -q
```

Python ph?i l? 3.11.x v? `torch.cuda.is_available()` ph?i l? `true`.

## T?o l?i m?i tr??ng s?ch

Python 3.11 l? baseline b?t bu?c:

```powershell
cd D:\rag-scifact
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
```

`requirements.lock.txt` l? snapshot ch?nh x?c c?a m?i tr??ng ?? verify v? c? th?m PyTorch CUDA 12.8 index ?? wheel `torch==2.11.0+cu128` c? th? ???c resolve. `requirements.txt` v? `requirements-dev.txt` v?n l? manifest theo kho?ng version d?nh cho ph?t tri?n; benchmark/final reproduction ?u ti?n lockfile.

Sau khi c?i:

```powershell
python -m scripts.check_environment
python -m unittest discover -s tests -v
pytest -q
python -m pip check
```

## Ollama / Ministral tr?n m?y n?y

M?y Admin ?? c? Ollama v? model. Ki?m tra:

```powershell
ollama --version
ollama list
ollama ps
```

N?u d?ng l?i tr?n m?y s?ch, t?i ??ng profile ?? verify:

```powershell
ollama pull ministral-3:3b-instruct-2512-q8_0
```

Ollama t? ph?c v? local API tr?n port 11434. Project d?ng OpenAI-compatible endpoint:

```text
http://127.0.0.1:11434/v1
```

Test tr?c ti?p:

```powershell
$body = @{
    model = "ministral-3:3b-instruct-2512-q8_0"
    messages = @(@{ role = "user"; content = "Reply with exactly: OK" })
    temperature = 0
    max_tokens = 512
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
    -Uri "http://127.0.0.1:11434/v1/chat/completions" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

Kh?ng d?ng cloud LLM fallback cho benchmark.

## C?u h?nh local

`.env.example` ch?a profile ?? verify nh?ng kh?ng ch?a secret. T?o `.env` khi c?n override local:

```powershell
Copy-Item .env.example .env
```

`.env` b? gitignore v? kh?ng ???c commit.

C?c gi? tr? runtime hi?n t?i:

```dotenv
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
```

`LLM_REVISION` ?? tr?ng v? Ollama digest kh?ng ph?i Hugging Face revision. `LLM_DTYPE` c?ng ?? tr?ng cho ??n khi runtime cung c?p m?t compute dtype ?? r? ?? ghi ch?nh x?c. `runtime_profile_locked=false` cho ??n khi E4 concrete generator, health/error mapping v? manifest ???c ho?n t?t; hardware/runtime smoke test th? ?? pass.

## Baseline benchmark

- seed = 42
- embedding = `sentence-transformers/all-MiniLM-L6-v2`
- embedding dimension = 384
- normalized float32
- chunk body = t?i ?a 220 MiniLM tokenizer tokens
- overlap = 30 tokens
- dense index = FAISS `IndexFlatIP`
- Top-K m?c ??nh = 5
- generation temperature = 0
- generation output limit = 512 new tokens
- local model = Ministral 3 3B Instruct 2512 Q8_0 qua Ollama

## Dataset hi?n c?

C?c file SciFact ?? c? trong `data/scifact/`:

```text
corpus.jsonl                5,183 documents
queries.jsonl               1,109 queries
qrels/train.tsv             809 unique query IDs
qrels/test.tsv              300 unique query IDs
```

E1 s? th?c hi?n audit/hash, deterministic dev/practice split, duplicate/near-duplicate report v? t?o manifest. Kh?ng tune theo final test.

## Test E0

Hi?n c? c?c smoke test cho import/config/logging. Tr?n m?i tr??ng Admin ?? verify:

```powershell
python -m unittest discover -s tests -v
pytest -q
python -m compileall -q app scripts tests
python -m pip check
```

## C?c CLI ch?a tri?n khai

C?c command sau ?? c? parser/boundary nh?ng ch?a ch?y pipeline th?t cho t?i khi E1-E7 ???c tri?n khai:

```powershell
python -m scripts.build_index --config config.yaml
python -m scripts.retrieve --query "..." --top-k 5
python -m scripts.ask --query "..." --top-k 5
python -m scripts.evaluate --split dev --config config.yaml
python -m scripts.evaluate_generation --fixture data/fixtures/atlas.jsonl
```

Kh?ng coi output scaffold l? benchmark result.
