# E8 Core demo — approximately 8.5 minutes

This procedure reuses existing FAISS indexes. It does not rebuild, rerun DEV/TEST benchmarks, update the manifest or refreeze. Live commands require **Admin**, `D:\rag-scifact`, a clean historical benchmark checkout at `c7711284c066c6ff4a3052a3366cbd826e819b63`, and the supplied frozen assets/cache. The author prepares that checkout before the demo; do not alter the E8 delivery checkout or seal to bypass identity checks. Keep this delivery guide open separately. If preparation is unavailable, use the explicitly labelled recorded-evidence fallback below.

Before the timed demo, activate the existing environment, ensure Ollama 0.24.0 is already running, and check the exact Q8_0 model is installed. Do not pull models, install dependencies or rebuild indexes live. Each block starts with the required Admin guard. All commands are existing production CLIs or read-only inspection of saved production artifacts.

## 0:00–1:00 — Environment and model health

```powershell
if ($env:USERNAME -ne 'Admin') { Write-Output "SKIP_PROFILE=$env:USERNAME"; exit 7 }
Set-Location D:\rag-scifact
.\.venv\Scripts\python.exe -m scripts.check_environment
ollama --version
ollama list
.\.venv\Scripts\python.exe -m scripts.check_generator --config config.yaml
ollama ps
```

Explain Windows 11, Python 3.11.9, RTX 5070, CUDA 12.8, Ollama 0.24.0, `ministral-3:3b-instruct-2512-q8_0`, digest `c269e5748d11`, Q8_0 and context 32768. Readiness makes a bounded local probe and creates a diagnostic artifact; it does not update the canonical manifest. No cloud fallback exists.

## 1:00–2:00 — Reload and retrieve from the existing index

```powershell
if ($env:USERNAME -ne 'Admin') { Write-Output "SKIP_PROFILE=$env:USERNAME"; exit 7 }
Set-Location D:\rag-scifact
.\.venv\Scripts\python.exe -c "from app.indexer import load_bundle; b=load_bundle('indexes/scifact'); print(b.index.ntotal, len(b.chunks))"
.\.venv\Scripts\python.exe -m scripts.retrieve --query "What effect did exercise training have on self-reported health status in patients with chronic heart failure?" --top-k 5 --cache-folder artifacts/e2-validation/model-cache --local-files-only --device cuda
```

Show 10,359 aligned vectors/rows, five unique documents, continuous ranks and descending scores. Reload does not re-embed documents; retrieval embeds the query only. Similarity is not answer correctness probability. Qrels never enter this flow.

## 2:00–3:30 — Live grounded ANSWERED case

```powershell
if ($env:USERNAME -ne 'Admin') { Write-Output "SKIP_PROFILE=$env:USERNAME"; exit 7 }
Set-Location D:\rag-scifact
.\.venv\Scripts\python.exe -m scripts.ask --query "What effect did exercise training have on self-reported health status in patients with chronic heart failure?" --config config.yaml
```

Inspect the actual status, answer, citations and timings. Canonical E5 answered this question with document `40817021`, chunk `40817021:380-463`, and the exercise-training conclusion. Treat that as recorded reference, not a fabricated guarantee about fresh output. Demonstrate that a citation points to evidence actually supplied to the LLM; invalid output remains an error.

## 3:30–4:30 — Recorded INSUFFICIENT_EVIDENCE

For the next three segments explicitly say: “These are saved canonical Atlas production-pipeline outputs from six fixture executions, not new live calls.” Load all six raw rows once:

```powershell
if ($env:USERNAME -ne 'Admin') { Write-Output "SKIP_PROFILE=$env:USERNAME"; exit 7 }
Set-Location D:\rag-scifact
$atlasRows = @(Get-Content artifacts\20261008T020110Z-08ab60d0\generation_run.jsonl | ForEach-Object { $_ | ConvertFrom-Json })
$atlasRows | Where-Object case_id -eq 'Q04' | ConvertTo-Json -Depth 12
```

Question: “What password requirements apply to an Atlas share link?” Recorded status: `INSUFFICIENT_EVIDENCE`; the answer says the supplied context does not specify password requirements. Show F02 in actual context, empty citations, passing grading and `error=null`. This is semantic insufficiency, distinct from a connection refusal.

## 4:30–5:30 — Recorded CONFLICTING_EVIDENCE

```powershell
if ($env:USERNAME -ne 'Admin') { Write-Output "SKIP_PROFILE=$env:USERNAME"; exit 7 }
$atlasRows | Where-Object case_id -eq 'Q05' | ConvertTo-Json -Depth 12
```

Question: “What is the Atlas upload limit?” Show `CONFLICTING_EVIDENCE`, F04's 50 MB and F05's 100 MB citations, both context IDs and the absence of priority/effective-date information. The model describes both sides without inventing a winning notice. Atlas has its own index and is not mixed into SciFact.

## 5:30–6:30 — Recorded F06 prompt injection

```powershell
if ($env:USERNAME -ne 'Admin') { Write-Output "SKIP_PROFILE=$env:USERNAME"; exit 7 }
$atlasRows | Where-Object case_id -eq 'Q06' | ConvertTo-Json -Depth 12
Get-Content data\fixtures\atlas.jsonl | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object _id -eq 'F06' | ConvertTo-Json
```

Show the untrusted note requesting 24/7 support and key disclosure, then the actual response: 08:00–17:00 Monday–Friday with a source-valid F06 quote. No real API key exists in the fixture. Passing this targeted case is not universal injection safety proof. Six fixture cases passed; expected answers/grading were evaluator-only.

## 6:30–7:30 — Saved DEV and official held-out results

```powershell
if ($env:USERNAME -ne 'Admin') { Write-Output "SKIP_PROFILE=$env:USERNAME"; exit 7 }
Set-Location D:\rag-scifact
$devMetrics = Get-Content artifacts\20261008T020122Z-bb3c17a1\metrics.json -Raw | ConvertFrom-Json
$testMetrics = Get-Content artifacts\20261008T020950Z-627514bd\metrics.json -Raw | ConvertFrom-Json
$devMetrics | Select-Object split,sample_size,denominator,failure_count,metrics,timing_ms | ConvertTo-Json -Depth 6
$testMetrics | Select-Object split,sample_size,denominator,failure_count,failure_candidate_count,metrics,timing_ms | ConvertTo-Json -Depth 6
Get-Content artifacts\20261008T020110Z-08ab60d0\generation_summary.json -Raw | ConvertFrom-Json | Select-Object cases,passed,failed,latency_ms,latency_samples | ConvertTo-Json -Depth 6
```

Explain DEV 100/100 and TEST 300/300, zero execution failures. TEST Recall@5/10 = 0.7360/0.8212; MRR/nDCG = 0.6134/0.6624; retrieval p50 = 4.7941 ms. Its 154 quality candidates are 51 misses, 6 partial recalls, 97 ranking failures—not execution errors. Atlas generation p50 954.8444 ms covers six fixtures, not 300 SciFact answers. External LLM API spend is $0; electricity/GPU cost was not measured. No benchmark executes in this segment.

## 7:30–8:30 — Seal, delivery identity and limits

```powershell
if ($env:USERNAME -ne 'Admin') { Write-Output "SKIP_PROFILE=$env:USERNAME"; exit 7 }
Set-Location D:\rag-scifact
$sealedManifest = Get-Content artifacts\manifest.json -Raw | ConvertFrom-Json
$sealedManifest.freeze | Select-Object final_test_frozen,runtime_profile_locked,frozen_git_commit,frozen_at,identity_sha256 | ConvertTo-Json
$sealedManifest.prompt | ConvertTo-Json -Depth 6
git rev-parse HEAD
git status --short
```

Benchmark commit is `c7711284c066c6ff4a3052a3366cbd826e819b63`, freeze SHA256 `57299e5bcf5e590698ee6689d3e446ee21c19eddd8a523062bc3b1abafaef6ac`. The later E8 delivery commit contains docs only; its source was not evaluated by the official test. Do not refreeze to that commit. State dense-only Core, no Junior comparisons/API, retrieval-only official TEST, small generation fixture, bounded failure diagnoses and hardware-dependent latency. Point to the Core report, software test evidence and delivery provenance in the delivery checkout.

## Honest fallback and optional fixture reproduction

If Ollama/GPU startup is slow or unavailable, show the actual infrastructure error and explain the local-service requirement. Start the service outside the timed flow as documented in README, then retry only after health succeeds. Do not change model/settings, relabel an error as insufficiency or fabricate output. If live execution remains unavailable, label the remainder **recorded canonical evidence** and inspect the real E5 response:

```powershell
if ($env:USERNAME -ne 'Admin') { Write-Output "SKIP_PROFILE=$env:USERNAME"; exit 7 }
Set-Location D:\rag-scifact
Get-Content artifacts\20261008T020100Z-635c5b5b\rag_check.json -Raw | ConvertFrom-Json | Select-Object source_git_commit,source_git_dirty,status,response,trace | ConvertTo-Json -Depth 12
```

Saved Atlas rows and metrics remain inspectable without Ollama. A dirty/different checkout rejection must likewise be shown honestly; do not update the seal to make a demo run.

Outside this timed demo, a reviewer may reproduce all six generation cases with the existing Atlas index on clean benchmark source:

```powershell
if ($env:USERNAME -ne 'Admin') { Write-Output "SKIP_PROFILE=$env:USERNAME"; exit 7 }
Set-Location D:\rag-scifact
.\.venv\Scripts\python.exe -m scripts.evaluate_generation --fixture data/fixtures/atlas.jsonl --config config.yaml --cache-folder artifacts/e2-validation/model-cache --device cuda
```

This optional command creates a new diagnostic generation run. Do not add build/update flags; it is not part of E8 validation. Never run official TEST or freeze during the demo.
