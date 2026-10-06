# 06 - Junior extensions: hybrid retrieval, reranking, API and performance

Junior work starts after the Core dense baseline has been frozen.

## BM25

Build lexical retrieval over the same corpus.

Rules:

- do not index qrels, queries or gold answers
- document the tokenizer/preprocessing used
- preserve doc/chunk mapping
- BM25 independently returns a ranked list
- candidate cap for the standard hybrid experiment: Top 50

Tests cover:

- empty query
- Unicode handling
- duplicate docs
- Top-50 cap
- stable ordering/tie behavior

## Dense + BM25 RRF

For the standard hybrid configuration:

- dense candidate list: Top 50
- BM25 candidate list: Top 50
- fuse with Reciprocal Rank Fusion
- RRF constant k = 60
- each list contributes 1 / (60 + rank)
- sum contributions per document
- keep the Top 50 fused documents before downstream reranking/final Top-K

Tie-breaking must be deterministic and covered by a toy test.

The same evaluator is used for dense and RRF modes.

## Cross-encoder reranker

Model:

- cross-encoder/ms-marco-MiniLM-L6-v2

Pin model revision/version used.

Canonical reranker input per document:

- document title
- the passage/chunk with the highest dense score for that document

Use this same canonical dense-evidence passage when reranking candidates that came from RRF, so the reranker input remains aligned with the original exercise and backlog E10-02.

Check the reranker tokenizer limit and log truncation counts.

Do not silently hide reranker failures or truncation.

## Four comparison modes

Run the same dev sample and evaluator for:

1. dense
2. dense + reranker
3. BM25 + dense RRF
4. RRF + reranker

Each mode gets separate:

- config/run identity
- retrieval_run artifact
- metrics
- timing data

Keep the comparison denominator and metric implementation identical.

## Dev experiment matrix

Junior may experiment with:

- chunk size
- K
- relevance threshold

Rules:

- experiments run on dev only
- keep seed/model/LLM/evaluator fixed unless the experiment explicitly studies one of them
- assign each experiment an ID and config snapshot
- avoid changing multiple uncontrolled variables in one experiment
- choose final config using measured dev quality and latency trade-offs
- do not assume reranking/hybrid always improves results

Freeze final config before final test.

## FastAPI POST /ask

Provide:

- POST /ask
- production-shaped request/response schema aligned with Core

Request validation:

- query after trim: 1..2000 characters
- top_k: integer 1..10

Invalid requests should return an appropriate HTTP 400/422 response with machine-readable error information.

Do not call retriever or LLM for invalid input.

## Readiness and infrastructure errors

If required components are unavailable, for example:

- index missing/corrupt
- local LLM unavailable
- model not ready

return an explicit service error and HTTP 503 where appropriate.

Do not return INSUFFICIENT_EVIDENCE to hide service failure.

## Timeout handling

Set explicit retrieval/generation timeouts appropriate to the implementation.

Map timeout exceptions into a structured error response. Requests must not hang indefinitely.

Test timeout behavior using mocks where appropriate.

## API logging

Each API request logs:

- request_id
- timestamp
- retrieval mode
- model ID
- semantic status or error
- retrieval/generation/total timing
- run/config identity where applicable

It must be possible to trace an API request to its retrieval/generation artifacts when artifacts are enabled.

## Performance

Warm up the relevant models/runtime before benchmark timing unless reporting cold-start separately.

For the four retrieval modes, report:

- p50 latency
- p95 latency
- retrieval time
- reranker time where applicable
- generation time
- total time

Record hardware and runtime context.

If the local generation backend exposes input/output token usage, record it. If usage is unavailable, report N/A rather than inventing zero.

Local API monetary cost can be reported as zero external API spend, but the report should still discuss compute/runtime trade-offs.

## Robustness/soak checks

Run full dev batches and keep:

- OOM events
- timeouts
- runtime errors
- failure count

Do not drop failures from metric denominators.

Reloading/restarting the index must not change retrieval semantics unexpectedly.

## Security regression

Run F06 through:

- API
- relevant retrieval modes

No mode may obey instructions embedded in the document or expose secrets in user-facing output/public logs.

## Junior error analysis

Analyze at least ten cases, separating:

- retrieval miss
- ranking failure
- chunking issue
- RRF issue
- reranker issue
- generation issue
- citation issue

Each conclusion should point to evidence from run artifacts.

## Final comparison table

Compare all four modes with:

- Recall@5
- Recall@10
- MRR@10
- nDCG@10
- p50
- p95
- token usage or N/A
- failure count
- qualitative trade-off

Every row must link to a reproducible run/config identity.

## Acceptance criteria

- BM25 and RRF implement the specified Top-50 and k=60 behavior.
- Reranker uses the pinned cross-encoder and logs truncation.
- Four modes run on the same dev sample.
- FastAPI /ask validates input and preserves Core response/error semantics.
- p50/p95 and failures are measured.
- At least ten Junior failure cases are analyzed.
- Final mode selection is justified by measured dev evidence.
