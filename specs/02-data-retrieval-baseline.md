# 02 - Data, chunking, embedding and dense retrieval baseline

## SciFact package contract

Expected files:

~~~text
data/scifact/corpus.jsonl
data/scifact/queries.jsonl
data/scifact/qrels/train.tsv
data/scifact/qrels/test.tsv
~~~

Expected counts:

| Item | Expected |
| --- | ---: |
| corpus documents | 5,183 |
| queries | 1,109 |
| unique query IDs present in train qrels | 809 |
| unique query IDs present in test qrels | 300 |

All judgments in the canonical staged files currently have score 1, so these counts are also the query IDs with positive relevance. Split membership is nevertheless defined by presence in the qrels file, not by filtering on score.

If counts or file structure differ, stop the benchmark and report a dataset/version mismatch. Do not edit corpus/qrels to force the expected numbers.

## Loader

The loader must:

- read corpus, queries and qrels through reusable modules
- normalize ID types for internal use while preserving source values where useful
- detect duplicate IDs and missing required fields
- report malformed records with file/line or equivalent record identity
- never silently skip malformed input
- require the qrels header to be exactly query-id, corpus-id, score in that order; extra, missing or reordered columns are a dataset/version mismatch
- validate every physical qrels data row before field parsing: it must contain exactly three tab-separated fields; blank rows, rows with extra fields and rows with missing fields are malformed and must fail with file/line context rather than being skipped

Create SHA256 hashes for the main dataset files and record them in the audit output/manifest. For this staged SciFact package the canonical raw-byte hashes are:

| File | SHA256 |
| --- | --- |
| corpus.jsonl | dec31c8182f3d744c7d2c09423756fd1d17cbef75808db13ba01cc0aab4d1ac6 |
| queries.jsonl | 8ff84a7c903f722981cd8d595c022660140c51867b27608a6d4910db86080313 |
| qrels/train.tsv | a53f2114831916c096b6c37d9e54da68cef4efdcdbd5ed46533601af972acf1d |
| qrels/test.tsv | 0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c |

A hash mismatch must fail E1 even when record counts still match.

## Deterministic dev/practice split

Do not assume qrels/dev.tsv exists.

Required algorithm:

~~~python
ids = sorted(all_unique_query_ids_present_in_train_qrels, key=int)
rng = random.Random(42)
rng.shuffle(ids)
dev_ids = ids[:100]
practice_ids = ids[100:]
~~~

Expected sizes:

- dev: 100
- practice: 709

Store both ID lists and seed 42 in manifest.json. Seed 42 is fixed by the benchmark contract: E1 must reject a project/config seed other than 42 rather than silently generating a different split. Query IDs with score 0, if present in a future/repackaged qrels file, still participate because split membership is based on file presence. Do not use test results for tuning.

Audit exact and near-duplicate query text across practice/dev/test splits and publish the counts. This is a data-integrity audit only; do not use final-test query content or overlap information to tune retrieval, prompts, thresholds or model settings.

The E1 implementation freezes `query_overlap_v1` before benchmark tuning:

- raw exact: decoded source strings are equal
- canonical exact: Unicode NFKC + casefold + whitespace collapse
- near duplicate: canonical-exact pairs excluded; Unicode word tokens; maximum token count at least 5; token-level Levenshtein distance <= `max(1, floor(0.10 * max_token_count))`
- no stemming, stopword removal, semantic embeddings, query merging or query removal
- publish aggregate cross-split counts only; do not publish final-test query text/IDs in the audit artifact
- a later policy bug fix requires a version bump plus synthetic regression test; never change the policy in response to final-test overlap or benchmark scores

## Chunking contract

Chunk budget is defined by the tokenizer of sentence-transformers/all-MiniLM-L6-v2, not word count and not the LLM tokenizer.

Rules:

- Combine title and chunk content for embedding.
- Chunk body maximum: 220 embedding-model tokens, following backlog E2-02.
- The final `title + chunk` input must still fit the MiniLM model/tokenizer input limit. Truncate an overlong title to fit the input budget and log the truncation.
- Body overlap: 30 embedding-model tokens.
- Short abstracts may be a single chunk.
- Preserve original text needed for citation.

Required chunk metadata:

- doc_id
- stable unique chunk_id
- title
- source text
- token start/end or an equivalent source position

Given chunk_id, the system must be able to recover the exact source document and evidence text.

### E2 implementation details

`MiniLMChunker.from_embedder` uses the same loaded SentenceTransformer tokenizer\nas `MiniLMEmbedder`. A fast tokenizer with reliable offset mappings and\nWordPiece word identities is required. Body tokenization disables special tokens\nand truncation. Chunk boundaries are chosen only at stable tokenizer word boundaries.\nFor a non-final chunk, the end boundary must also make end - 30 a stable boundary;\nthis guarantees an exact 30-token standalone overlap. The largest valid end not\nexceeding the 220-token body budget is selected.\n\nEach exact source substring is retokenized before acceptance and must reproduce the\noriginal document token IDs for its half-open token span. Therefore standalone body\ntoken count is at most 220 and adjacent chunks satisfy actual token-ID overlap, not\nonly arithmetic overlap in the original document stream.\n\n`Chunk` retains the original title, exact source substring, half-open token and\ncharacter spans, standalone body token count, separate embedding title and truncation\nflag. IDs follow `<doc_id>:<token_start>-<token_end>`. Empty/whitespace-only\nbodies return zero chunks; a non-empty body with unusable offsets/word identities\nfails explicitly. Interior source whitespace, Unicode and punctuation are preserved.\n\nThe deterministic embedding separator is two newlines, omitted with an empty embedding\ntitle. The effective input limit is the loaded SentenceTransformer max_seq_length,\nreduced by a smaller finite tokenizer limit when present. It includes special tokens;\nthe observed exact-model runtime limit is 256. Long titles are reduced by source-token\nprefix until the actual combined input fits; body evidence is never silently truncated.\nPer-chunk truncation logging includes document ID, original/retained title counts,\nlimit and body token count.\n\nE2 pins the verified Hugging Face revision\n`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` and exposes it in runtime metadata.\nE2 does not write the E1 manifest or produce an index; E3 must apply the existing\nmanifest freeze/identity guards when recording these values.\n\n## Embedding contract

Model: sentence-transformers/all-MiniLM-L6-v2.

Rules:

- Encode title + chunk.
- Batch encoding is supported.
- Output dimension is 384.
- Convert to float32.
- L2-normalize corpus and query vectors.
- Batch and single-item calls use compatible semantics.

Tests must prove dimension and norm expectations.

`MiniLMEmbedder` exposes `encode_texts`, `encode_chunks` and `encode_query` with
one shared model/preprocessing path. Empty batches return `(0, 384)` float32.
Inputs exceeding the effective input limit fail rather than relying on hidden
SentenceTransformer truncation. The wrapper requests normalized embeddings,
then casts, checks shape/finiteness, explicitly normalizes, and validates unit
norms; zero/invalid vectors fail. E2 runtime construction rejects changes to
the locked model, 220/30 windows, float32 dtype or normalization flag. Runtime
metadata includes observed device, output contracts and the pinned verified revision. Heavy libraries/model loading remain lazy, and offline unit tests
inject deterministic backends.

## FAISS index

Use FAISS IndexFlatIP over normalized float32 vectors.

Index all corpus chunks. The build log must include:

- embedding model identity
- chunk configuration
- total chunks
- index.ntotal
- build duration

Persist:

- FAISS index
- vector-position -> chunk/document metadata mapping
- enough version/config identity to detect mismatches

Reload must not re-embed the corpus.

## Dense retrieval

For a query:

1. Validate/normalize input.
2. Embed with the same embedding model.
3. Normalize the query vector.
4. Search more chunks than the final document Top-K when necessary.
5. Sort by score descending.
6. Group by doc_id.
7. Keep the highest scoring chunk as the canonical evidence chunk per document.
8. Take final Top-K unique documents.
9. Reassign continuous ranks 1..K.

The retriever must support top_k from 1 to 10.

The cosine/IP score is a retrieval similarity score. It is not answer correctness probability.

## CLI retrieve

Support both:

~~~powershell
python -m scripts.retrieve --query "..." --top-k 5
python -m scripts.retrieve --query-id 0 --top-k 5
~~~

Output at minimum:

- doc_id
- rank
- score
- chunk_id

The R01-R20 demo queries must be read in their original English form from queries.jsonl by query ID. The Vietnamese descriptions in the exercise are explanatory only and must not drive benchmark optimization.

## Acceptance criteria

- Dataset audit matches expected counts before benchmarking.
- Split is deterministic across runs.
- Exact/near-duplicate query counts across splits are reported.
- No chunk exceeds the embedding token contract.
- 30-token overlap is verified.
- Embeddings are 384-D normalized float32.
- index.ntotal equals the number of indexed chunks.
- Reloaded index returns equivalent retrieval semantics.
- Final Top-K has no duplicate doc_id.
- Missing/corrupt index is reported as an infrastructure error, not as no evidence.
