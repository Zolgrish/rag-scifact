# 03 - Local Ministral 3 3B Instruct runtime

## Purpose

Replace the original organizer-provided Qwen endpoint with a local Ministral 3 3B Instruct generation service while keeping retrieval, prompt, citation, evaluation, and benchmark semantics unchanged.

Reference model from the backlog:

- mistralai/Ministral-3-3B-Instruct-2512-BF16

Reference local API shape from the backlog:

- Backlog reference endpoint shape: OpenAI-compatible local API (the verified Ollama implementation uses `http://127.0.0.1:11434/v1`)
- temperature 0 or greedy
- max_new_tokens 512
- seed 42 where supported

The final implementation may use another local runtime adapter if the default vLLM path is not suitable for the verified Windows environment. The benchmark model family remains Ministral 3 3B Instruct.

## Final-machine hardware constraint

Final benchmark/demo machine:

- Windows 11, profile `Admin`
- NVIDIA GeForce RTX 5070; PyTorch reports 11.94 GiB VRAM
- repository `.venv`: Python 3.11.9
- PyTorch 2.11.0+cu128, CUDA 12.8, `torch.cuda.is_available() == True`
- Ollama 0.24.0

Hardware/runtime visibility is verified on the final Admin machine.

The backlog notes that the BF16 profile can require around 16 GB VRAM. Therefore BF16 is a reference profile, not a guaranteed runnable profile on this machine.

Do not wait until the end of the project to test the model. The runtime spike belongs near the beginning of implementation.

## Verified runtime decision on the Admin machine

The runtime spike is complete for hardware/model serving:

- runtime: Ollama 0.24.0
- model: `ministral-3:3b-instruct-2512-q8_0`
- Ollama model ID/digest: `c269e5748d11`
- quantization: Q8_0
- local OpenAI-compatible API: `http://127.0.0.1:11434/v1`
- model-declared maximum context: 262144 tokens
- effective context observed in `ollama ps`: 32768 tokens
- processor: 100% GPU
- API smoke test: successful `/v1/chat/completions` response
- benchmark generation settings remain temperature 0, max 512 new tokens, seed 42 where the backend supports it

The Ollama digest is runtime identity and must not be mislabeled as an upstream Hugging Face revision. Compute dtype is intentionally blank until it can be stated accurately. `runtime_profile_locked` stays false until the application Generator, health/readiness and failure mapping are implemented and tested.


## Runtime decision procedure

Run these checks before pinning the LLM config:

1. Record Windows version.
2. Confirm the RTX 5070 is visible to the selected CUDA/PyTorch stack.
3. Record total/free VRAM before model load.
4. Verify model files can be downloaded/cached.
5. Attempt the intended model/runtime profile.
6. Measure whether the model loads without OOM.
7. Run one deterministic chat completion.
8. Record runtime/library versions and effective model context limit.

If the BF16 reference profile does not fit:

1. Select a verified quantized profile of the same Ministral 3 3B Instruct model family.
2. Record exact model ID and revision.
3. Record quantization format and dtype.
4. Reduce maximum model length only when required by the verified runtime.
5. Re-run the health and one-completion checks.

Do not guess a quantized checkpoint name in code/config before it has been verified.

## Windows runtime policy

The backlog proposes vLLM as the default serving layer but explicitly allows Transformers or another local OpenAI-compatible runtime when vLLM is unsuitable for the platform.

For this Windows machine:

- Do not assume native Windows vLLM compatibility; verify the actual installation path.
- If vLLM is used through WSL2 or another supported environment, document that clearly in README.
- If a Transformers-based local runtime is used instead, expose it through the same Generator abstraction.
- Retrieval and generation application code must not depend directly on a particular serving implementation.

The selected path for this final machine is Ollama. The Generator remains runtime-independent at the application boundary so the benchmark logic is not coupled to Ollama-specific calls.

## Generator interface

The application generator abstraction accepts:

- model identity/config
- messages or prompt
- timeout options
- generation parameters

It returns:

- generated text
- model ID
- usage information when the runtime exposes it
- generation timing
- structured runtime error on failure

CLI and FastAPI code call this abstraction; they must not duplicate model-client logic.

## Deterministic settings

For benchmark runs:

- temperature = 0 or greedy
- max_new_tokens = 512
- seed = 42 if the backend supports a seed
- keep model revision/config fixed after freeze

Do not promise byte-identical output across every GPU/backend. Record enough environment information to explain runtime differences.

## Local-only rule

During benchmark/demo generation:

- the RAG client calls only the configured local runtime
- no cloud LLM fallback
- no external web lookup
- no hidden provider fallback when the local model is unavailable

If the local model is down, the request returns an infrastructure error.

## Health and error contract

Provide a health/readiness check appropriate to the runtime.

Handle at least:

- connection refused
- connect timeout
- read/generation timeout
- model unavailable/not loaded
- GPU OOM
- malformed runtime response

These are infrastructure errors. They must not become:

- ANSWERED
- INSUFFICIENT_EVIDENCE
- CONFLICTING_EVIDENCE

For Junior API mode, unavailable required services should map to HTTP 503 where appropriate.

## Model cache/offline behavior

After model download is complete, the demo should be able to load from local cache/storage without needing a cloud generation API.

Record:

- model ID
- upstream revision when applicable
- runtime model ID/digest (for Ollama, keep this separate from upstream revision)
- local runtime/library version
- dtype
- quantization
- device
- effective max model length
- temperature
- max new tokens
- seed support

These fields belong in manifest.json.

## Context budget

The context builder must know the effective runtime/model context budget.

If retrieved context must be truncated:

- preserve retrieval order unless the chosen policy is explicitly documented
- never silently remove cited evidence
- log which chunks were included/excluded
- record truncation in the generation artifact

Chunking for embeddings remains governed by the MiniLM tokenizer and the 220/30 rule. LLM context budgeting is a separate concern.

## Acceptance criteria

- A verified local Ministral 3 3B Instruct profile runs on the RTX 5070 12 GB machine.
- The exact model/revision/quantization/runtime is recorded.
- A deterministic completion works locally without a cloud API key.
- The generator is independent from CLI/API layers.
- Model-down, timeout and OOM are explicit infrastructure errors.
- The RAG client does not silently fall back to a cloud LLM.
- README contains the exact verified startup procedure used on this machine.
