# Core delivery report — SciFact RAG

## 1. Architecture and model decision

The Core pipeline loads the canonical SciFact corpus, preserves exact source spans in MiniLM token-aware chunks, creates normalized embeddings, and persists FAISS `IndexFlatIP` with aligned evidence metadata. A query uses the same embedder; dense search deduplicates documents before Top-K, retaining each document's highest-scoring evidence chunk. Ranked context goes to local Ministral, followed by strict JSON parsing and context-only citation validation. Evidence quotes must map to a chunk actually sent to the LLM. Retrieved text is untrusted data, never executable instruction.

Responses distinguish `ANSWERED`, `INSUFFICIENT_EVIDENCE` and `CONFLICTING_EVIDENCE`. Missing/corrupt index, connection refusal, timeout, OOM and malformed output take explicit error paths; they are not evidence statuses. Whole-chunk context budgeting counts the rendered local chat and reserves output capacity. Qrels, reference answers and grading rules are evaluator-only and never enter runtime retrieval or generation.

**The original Qwen endpoint was intentionally replaced by a local Ministral 3 3B Instruct deployment**, as authorized by the Local Ministral backlog. The reference checkpoint is `mistralai/Ministral-3-3B-Instruct-2512-BF16`. Q8_0 was selected because it fits the verified approximately 12 GB GPU while reference BF16 may exceed VRAM. This choice does not establish quality equivalence to BF16 or Qwen.

| Verified runtime | Value |
| --- | --- |
| OS / profile / Python | Windows 11 x64 / Admin / 3.11.9 |
| GPU | NVIDIA GeForce RTX 5070; 11.94 GiB reported VRAM |
| PyTorch / CUDA | `2.11.0+cu128` / 12.8 |
| Ollama / placement | 0.24.0 / observed 100% GPU |
| Model | `ministral-3:3b-instruct-2512-q8_0` |
| Digest / quantization | `c269e5748d11` prefix / Q8_0 |
| Context / temperature | 32768 tokens / 0 |
| Output / seed | 512 new tokens / 42 |

Full observed digest: `c269e5748d11ad36a4848e2dc6e36c238bef82683889ad5be24a22e57fe9f887`. The upstream revision and compute dtype are unset; the runtime digest is not a Hugging Face revision. The client uses loopback `http://127.0.0.1:11434/v1`, with no cloud/model fallback.

<div style="page-break-after: always;"></div>

## 2. Dataset, baseline and frozen provenance

| Dataset quantity | Count |
| --- | ---: |
| Corpus documents / queries | 5,183 / 1,109 |
| Train query IDs / qrel rows | 809 / 919 |
| Test query IDs / qrel rows | 300 / 339 |
| DEV / practice / test | 100 / 709 / 300 |

All train IDs are numerically sorted, shuffled with `random.Random(42)`, then split into the first 100 DEV and remaining 709 practice IDs. The manifest retains membership/order and raw dataset hashes; malformed records and identity mismatch fail closed. The integrity audit found 2 canonical exact and 98 near duplicate cross-split pairs, reported without changing membership or denominators.

| Frozen baseline | Value |
| --- | --- |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| Pinned embedding revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Vectors | 384-dimensional, L2 normalized `float32` |
| Body budget / overlap | Maximum 220 / exactly 30 MiniLM tokens |
| Actual embedding input limit | 256 tokens including special tokens; truncate embedding title, preserve original evidence |
| Index / retrieval | FAISS `IndexFlatIP` / dense; 10,359 saved vectors |
| Serving / evaluation / maximum K | 5 / 10 / 10 |
| Prompt | `rag-grounded-json-v4` |

Chunk IDs are `<doc_id>:<token_start>-<token_end>`. Stable WordPiece boundaries preserve standalone token identity and exact overlap. Half-open character spans recover the original body substring. Title truncation is observable; original titles remain metadata. Empty bodies produce zero chunks.

**Benchmark commit:** `c7711284c066c6ff4a3052a3366cbd826e819b63`.

**Freeze identity SHA256:** `57299e5bcf5e590698ee6689d3e446ee21c19eddd8a523062bc3b1abafaef6ac`.

**Prompt SHA256:** `1b9b14f2563f9cfe88ebefec75c85cb69ff3480304c0d6058546e0df2fff6702`.

The baseline was sealed before official testing with `final_test_frozen=true` and `runtime_profile_locked=true`. Config, prompt, model/runtime, retrieval, index, dataset, split and canonical E5/E6/DEV artifacts are bound in the seal. Test gates recheck identity before judgments and before publication. No benchmark, freeze or settings were changed for E8.

**Delivery commit:** the later documentation/report/demo commit, assigned by the author after review. It is intentionally different from the evaluated benchmark commit; its hash is not invented here. Historical evidence remains attached to benchmark source. [E8 delivery provenance](../delivery/E8-provenance.json) records freshly computed evidence hashes without modifying the benchmark manifest.

<div style="page-break-after: always;"></div>

## 3. Recorded results, timing and cost

| Measure | Canonical DEV | Official held-out TEST |
| --- | ---: | ---: |
| Sample / denominator | 100 / 100 | 300 / 300 |
| Execution failures | 0 | 0 |
| Recall@5 | 0.7217 | 0.7360 |
| Recall@10 | 0.8017 | 0.8212 |
| MRR@10 | 0.6056 | 0.6134 |
| nDCG@10 | 0.6446 | 0.6624 |
| Retrieval p50, ms | 4.6822 | 4.7941 |

The held-out test metrics were slightly higher than the internal dev metrics under the same frozen baseline. No settings were tuned from test outcomes. Both runs use one Top-10 retrieval per query; Top-5/Top-10 measures derive from that result. Recall averages per query, MRR uses the first positive relevant rank, and nDCG preserves graded relevance with gain `2^rel − 1` and discount `log2(rank + 1)`. Failed and zero-positive queries remain in the denominator. Wall-clock retrieval timings include attempted query calls and exclude setup. MiniLM used CUDA (`cuda:0`); FAISS ran on CPU with 14 threads.

Canonical generation/citation fixture **F01–F06: 6 passed / 0 failed**. These six Atlas responses exercise permissions, expiry, approval, insufficiency, unresolved conflict and injection handling.

| Atlas F01–F06 timing only | p50, ms |
| --- | ---: |
| Retrieval | 4.6186 |
| Generation | 954.8444 |
| Total, including context budgeting | 1085.6832 |

There are six timing samples. **These are Atlas fixture timings, not generation latency over the 300-query SciFact test.** The official test measures retrieval; generated answer accuracy on those 300 queries was not measured.

External LLM API monetary spend: **$0**, because generation is local. Electricity/GPU operating cost was **not measured**. Cold loading, readiness and context-counting probes add time outside the E7 retrieval p50 scope. Deterministic inference settings do not promise byte-identical responses across machines/backends.

Exact, unrounded measurements and source/config/runtime identities are retained in:

- [E5 grounded smoke](../artifacts/20261008T020100Z-635c5b5b/rag_check.json).
- [E6 Atlas summary](../artifacts/20261008T020110Z-08ab60d0/generation_summary.json) and adjacent `generation_run.jsonl`.
- [Canonical DEV metrics](../artifacts/20261008T020122Z-bb3c17a1/metrics.json), adjacent raw retrieval and failure analysis.
- [Official TEST metrics](../artifacts/20261008T020950Z-627514bd/metrics.json), adjacent raw retrieval and failure analysis.

Run artifacts use strict JSON/JSONL, reject non-finite values, preserve explicit errors and publish summaries last. Canonical verification independently checked raw rankings, metrics, timing and hashes before sealing.

<div style="page-break-after: always;"></div>

## 4. Five measured DEV failure cases

Selection is deterministic: the first two misses, first partial recall and first two ranking failures in the saved DEV order (Recall@10, MRR@10, numeric query ID). This provides category coverage without choosing held-out cases. All positive relevance grades below are 1.0. `doc@rank` lists observed documents; full Top-10 ranks/scores remain in [DEV failure analysis](../artifacts/20261008T020122Z-bb3c17a1/failure_analysis.jsonl).

**14 — retrieval miss.** “5'-nucleotidase metabolizes 6MP.” Expected: `641786`. Retrieved head: `393001@1`, `6421792@2`, `34498325@3`; the positive document is absent from Top-10. Observed Recall@5/10, MRR and nDCG: all 0. Immediate measured failure mode: relevant document absent from Top-10. Deeper cause is unestablished. Future DEV-only idea on a separate experimental baseline: inspect the positive document's best chunk rank/score beyond rank 10 before considering representation changes.

**15 — retrieval miss.** “50% of patients exposed to radiation have activated markers of mesenchymal stem cells.” Expected: `22080671`. Retrieved head: `25238950@1`, `37437064@2`, `18953920@3`; the positive is absent from Top-10. All four metrics are 0. Immediate failure mode: relevant evidence did not reach the cutoff; no embedding or lexical mechanism is proven. Future idea: compare the positive source/chunk with higher-ranked evidence on DEV, then test a representation alternative only if that inspection supports it.

**1407 — partial recall.** “β1/Ketel is able to bind microtubules.” Expected: `29863668`, `8087082`. Retrieved head: `1546650@1`, `21598000@2`, `8570478@3`; `8087082@9` is present and `29863668` is absent. Recall@5=0, Recall@10=0.5, MRR=0.1111, nDCG=0.1846. Immediate failure mode: incomplete coverage plus a low first positive rank; deep cause is unestablished. Future idea: inspect omitted-positive ranks and score margins against the retrieved positive before evaluating ranking changes on DEV.

**383 — ranking failure.** “Epidemiological disease burden from noncommunicable diseases is more prevalent in high economic settings.” Expected: `13770184`. Retrieved head: `25691878@1`, `581832@2`, `51972698@3`; positive `13770184@9`. Recall@5=0, Recall@10=1, MRR=0.1111, nDCG=0.3010. Immediate failure mode: relevant evidence retrieved but ranked below serving Top-5; deeper cause is unestablished. Future idea: inspect the eight higher-ranked chunks and score margins before testing a ranking alternative on DEV.

**1300 — ranking failure.** “Thiopurine active metabolites can be catabolized through dephosphorylation of thioguanine nucleotides.” Expected: `6421792`. Retrieved head: `32975424@1`, `1818578@2`, `2890952@3`; positive `6421792@8`. Recall@5=0, Recall@10=1, MRR=0.1250, nDCG=0.3155. Immediate failure mode: relevant evidence ranked below serving Top-5; deeper cause is unestablished. Future idea: audit higher-ranked chunk relevance and margins, then compare an independent ranking experiment using DEV only.

Official TEST summary: **154 quality candidates = 51 misses + 6 partial recalls + 97 ranking failures; 0 execution errors.** Thus 154 does not mean 154 failed executions. These observations describe retrieval quality, not generation errors. Future ideas above use DEV cases only and do not authorize changes to the frozen Core or tuning from held-out failures.

<div style="page-break-after: always;"></div>

## 5. Validation, limitations and disclosure

E8 software validation on 2026-10-08: **pytest: 243 passed and 300 subtests passed; unittest: 243 tests, OK; targeted Core acceptance: 123 passed and 196 subtests passed; compileall: exit 0; pip check: no broken requirements**. Existing suites cover strict IDs/loading, 220-token budget/30-token overlap, 384-dimensional normalized embeddings, persistence/reload, document dedup/order, citation authority, local connection errors, F06 injection, metric toy math and infrastructure/status separation. Application tests use injected local backends/mocks; no public web/cloud LLM is required. Commands and coverage map are in [README](../README.md); this validation did not rerun the official benchmark.

Core uses dense retrieval only; no BM25/RRF/reranker comparison or Junior API is claimed. TEST measures retrieval, not end-to-end answer accuracy or scientific support/contradiction labels. F01–F06 is a small targeted generation/citation fixture, not universal safety proof. Zero execution errors still permits retrieval misses. The five diagnoses establish immediate failure modes, not causal proof about embeddings, chunking, lexical mismatch or generation. Latency varies with load/cache/hardware; electricity/GPU cost is not measured. Q8_0 versus BF16/Qwen quality was not compared.

Reproduction requires the pinned MiniLM cache, saved SciFact and separate Atlas indexes, raw corpus files, locked dependencies and frozen manifest/run evidence, supplied separately where ignored. Index reload never re-embeds corpus; fresh rebuilds are separate outputs, not replacements for sealed bytes. Earlier pre-test development included prompt/grader corrections; the report uses only identified canonical runs. During a canonicalization attempt with Ollama stopped, E5/E6 failed closed on connection refusal and E7/freeze refused stale provenance. After restart, E5 → E6 → E7 DEV → freeze succeeded. No E8 runtime/benchmark changes or additional model calls were made.

Sources: DFM-ENGINEERING RAG exercise v1.0 and “RAG SciFact | Local Ministral 3 3B Instruct” backlog v1.0, both dated 05/10/2026; BEIR SciFact and the pinned MiniLM model. Used components: Python 3.11, PyTorch, sentence-transformers, Hugging Face Transformers/tokenizers/model stack, FAISS, NumPy, Requests, PyYAML, Ollama, pytest and unittest. No model was trained/fine-tuned; no LangChain/LlamaIndex core framework was used.

**AI assistance:** Codex/ChatGPT assisted architecture investigation, implementation, tests, code review/audit and documentation/report preparation. The author reviewed and changed AI-assisted work afterward. Concrete corrections include E7 failure-analysis semantics, final-test freeze/provenance race rechecking before artifact publication, canonical provenance verification, regression-test additions and documentation corrections. The project is not claimed to be fully manually authored.

Benchmark evidence belongs to `c7711284c066c6ff4a3052a3366cbd826e819b63`; the later E8 delivery commit contains documentation/report/demo only. Do not repair that distinction by refreezing or changing manifest source identity. The [8.5-minute demo](../docs/E8-demo.md) reuses indexes and recorded results, with an honest infrastructure fallback.

**Export:** Markdown is the canonical editable source. Render its tables and HTML page breaks; print to PDF on A4 portrait, 11-point text, 20 mm margins, headers/footers off. The five sections are intended as five pages; inspect preview and keep the report at at most five pages. No runtime dependency installation is needed for document export.
