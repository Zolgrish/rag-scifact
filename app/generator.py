"""Runtime-independent local LLM generation through an OpenAI-compatible API."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import TYPE_CHECKING, Mapping, Protocol, Sequence
from urllib.parse import urlparse

import requests

if TYPE_CHECKING:
    from app.config import LLMConfig


@dataclass(frozen=True)
class GeneratorResult:
    text: str
    model_id: str
    usage: Mapping[str, int] | None = None
    generation_ms: float | None = None


@dataclass(frozen=True)
class GeneratorReadiness:
    ready: bool
    model_id: str
    runtime: str
    response_ms: float


@dataclass(frozen=True)
class RuntimeMetadata:
    runtime: str
    runtime_version: str
    model_id: str
    runtime_model_digest: str
    quantization: str
    context_length: int
    size_vram_bytes: int | None = None


class Generator(Protocol):
    """Runtime-independent local generation interface."""

    def check_readiness(self) -> GeneratorReadiness:
        """Verify that the configured local model is available or raise."""

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *, response_format: Mapping[str, object] | None = None,
    ) -> GeneratorResult:
        """Generate text from messages or raise a structured infrastructure error."""

    def inspect_runtime_metadata(self) -> RuntimeMetadata:
        """Return observed local runtime/model identity for reproducibility checks."""

    def count_prompt_tokens(self, messages: Sequence[Mapping[str, str]]) -> int:
        """Count the exact rendered chat input; no heuristic fallback."""


class GeneratorError(RuntimeError):
    """Base class for generator failures."""


class GeneratorConfigurationError(GeneratorError):
    """Raised when the configured local generator profile is unsupported/unsafe."""


class GeneratorInputError(ValueError):
    """Raised for invalid messages passed by application code."""


class GeneratorInfrastructureError(GeneratorError):
    """Base class for local runtime/service failures."""

    code = "generator_infrastructure_error"
    retryable = True

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "code": self.code,
            "message": str(self),
            "retryable": self.retryable,
        }
        if self.status_code is not None:
            payload["status_code"] = self.status_code
        return payload


class GeneratorConnectionError(GeneratorInfrastructureError):
    code = "generator_connection_error"


class GeneratorConnectTimeoutError(GeneratorInfrastructureError):
    code = "generator_connect_timeout"


class GeneratorReadTimeoutError(GeneratorInfrastructureError):
    code = "generator_read_timeout"


class GeneratorModelUnavailableError(GeneratorInfrastructureError):
    code = "generator_model_unavailable"
    retryable = False


class GeneratorOutOfMemoryError(GeneratorInfrastructureError):
    code = "generator_out_of_memory"


class GeneratorMalformedResponseError(GeneratorInfrastructureError):
    code = "generator_malformed_response"
    retryable = False


class GeneratorModelMismatchError(GeneratorMalformedResponseError):
    code = "generator_model_mismatch"


class GeneratorHTTPError(GeneratorInfrastructureError):
    code = "generator_http_error"


_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_OOM_MARKERS = (
    "out of memory",
    "cuda oom",
    "cuda out of memory",
    "gpu memory",
    "not enough memory",
)
_MODEL_UNAVAILABLE_MARKERS = (
    "not found",
    "not available",
    "not loaded",
    "does not exist",
    "pull it first",
    "try pulling",
)
_USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _normalize_local_base_url(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise GeneratorConfigurationError("llm.base_url must be a non-empty local URL")
    value = base_url.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise GeneratorConfigurationError("llm.base_url must use http or https")
    if parsed.hostname not in _LOOPBACK_HOSTS:
        raise GeneratorConfigurationError(
            "E4 local generation requires a loopback-only llm.base_url"
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise GeneratorConfigurationError(
            "llm.base_url must not contain credentials, query, or fragment"
        )
    return value


def _validate_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise GeneratorInputError("messages must be a non-empty sequence of mappings")
    if not messages:
        raise GeneratorInputError("messages must not be empty")

    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise GeneratorInputError(f"messages[{index}] must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise GeneratorInputError(f"messages[{index}].role must be non-empty text")
        if not isinstance(content, str):
            raise GeneratorInputError(f"messages[{index}].content must be text")
        normalized.append({"role": role.strip(), "content": content})
    return normalized


def _error_text(response: requests.Response) -> str:
    text = getattr(response, "text", "")
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())[:500]


def _raise_for_runtime_error(response: requests.Response) -> None:
    status = int(response.status_code)
    if status < 400:
        return

    detail = _error_text(response)
    lowered = detail.casefold()
    if any(marker in lowered for marker in _OOM_MARKERS):
        raise GeneratorOutOfMemoryError(
            f"local generator reported an out-of-memory failure (HTTP {status})",
            status_code=status,
        )
    if "model" in lowered and any(
        marker in lowered for marker in _MODEL_UNAVAILABLE_MARKERS
    ):
        raise GeneratorModelUnavailableError(
            f"configured local model is unavailable (HTTP {status})",
            status_code=status,
        )
    raise GeneratorHTTPError(
        f"local generator returned HTTP {status}"
        + (f": {detail}" if detail else ""),
        status_code=status,
    )


def _parse_usage(value: object) -> Mapping[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise GeneratorMalformedResponseError("runtime usage must be an object")
    result: dict[str, int] = {}
    for field in _USAGE_FIELDS:
        if field not in value:
            continue
        count = value[field]
        if type(count) is not int or count < 0:
            raise GeneratorMalformedResponseError(
                f"runtime usage.{field} must be a non-negative integer"
            )
        result[field] = count
    return result or None


def _parse_completion_payload(
    payload: Mapping[str, object],
    *,
    expected_model_id: str,
) -> tuple[str, str, Mapping[str, int] | None]:
    returned_model = payload.get("model")
    if not isinstance(returned_model, str) or not returned_model:
        raise GeneratorMalformedResponseError(
            "chat completion response is missing model identity"
        )
    if returned_model != expected_model_id:
        raise GeneratorModelMismatchError(
            f"runtime returned model {returned_model!r}, expected {expected_model_id!r}"
        )

    choices = payload.get("choices")
    if (
        not isinstance(choices, Sequence)
        or isinstance(choices, (str, bytes))
        or not choices
    ):
        raise GeneratorMalformedResponseError(
            "chat completion response must contain at least one choice"
        )
    first = choices[0]
    if not isinstance(first, Mapping):
        raise GeneratorMalformedResponseError(
            "chat completion choice must be an object"
        )
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise GeneratorMalformedResponseError(
            "chat completion choice is missing message"
        )
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise GeneratorMalformedResponseError(
            "chat completion message content must be non-empty text"
        )
    return content, returned_model, _parse_usage(payload.get("usage"))


class OpenAICompatibleGenerator:
    """Local OpenAI-compatible chat-completions client with explicit failures."""

    def __init__(
        self,
        *,
        model_id: str,
        base_url: str,
        runtime: str = "openai_compatible",
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        seed: int = 42,
        connect_timeout_s: float = 10.0,
        read_timeout_s: float = 120.0,
        session: requests.Session | None = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise GeneratorConfigurationError("llm.model_id must be non-empty")
        if not isinstance(runtime, str) or not runtime.strip():
            raise GeneratorConfigurationError("llm.runtime must be non-empty")
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise GeneratorConfigurationError("llm.max_new_tokens must be > 0")
        if type(seed) is not int:
            raise GeneratorConfigurationError("llm.seed must be an integer")
        if (
            not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or temperature < 0
        ):
            raise GeneratorConfigurationError(
                "llm.temperature must be finite and >= 0"
            )
        if (
            not isinstance(connect_timeout_s, (int, float))
            or not math.isfinite(float(connect_timeout_s))
            or connect_timeout_s <= 0
        ):
            raise GeneratorConfigurationError(
                "llm.connect_timeout_s must be finite and > 0"
            )
        if (
            not isinstance(read_timeout_s, (int, float))
            or not math.isfinite(float(read_timeout_s))
            or read_timeout_s <= 0
        ):
            raise GeneratorConfigurationError(
                "llm.read_timeout_s must be finite and > 0"
            )

        self.model_id = model_id.strip()
        self.base_url = _normalize_local_base_url(base_url)
        self.runtime = runtime.strip()
        self.temperature = float(temperature)
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.connect_timeout_s = float(connect_timeout_s)
        self.read_timeout_s = float(read_timeout_s)
        self._session = session if session is not None else requests.Session()

        parsed = urlparse(self.base_url)
        if self.runtime == "ollama" and parsed.path.rstrip("/") != "/v1":
            raise GeneratorConfigurationError(
                "Ollama OpenAI-compatible llm.base_url must end with /v1"
            )
        self._native_runtime_root = f"{parsed.scheme}://{parsed.netloc}"

    @property
    def timeout(self) -> tuple[float, float]:
        return self.connect_timeout_s, self.read_timeout_s

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        return self._request_json_url(method, url, json_body=json_body)

    def _request_json_url(
        self,
        method: str,
        url: str,
        *,
        json_body: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        try:
            response = self._session.request(
                method,
                url,
                json=json_body,
                headers={"Accept": "application/json"},
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectTimeout as exc:
            raise GeneratorConnectTimeoutError(
                f"timed out connecting to local generator at {self.base_url}"
            ) from exc
        except requests.exceptions.ReadTimeout as exc:
            raise GeneratorReadTimeoutError(
                f"timed out waiting for local generator at {self.base_url}"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise GeneratorConnectionError(
                f"cannot connect to local generator at {self.base_url}"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise GeneratorReadTimeoutError(
                f"local generator request timed out at {self.base_url}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise GeneratorConnectionError(
                f"local generator transport failed at {self.base_url}: {exc.__class__.__name__}"
            ) from exc

        _raise_for_runtime_error(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise GeneratorMalformedResponseError(
                "local generator returned non-JSON data"
            ) from exc
        if not isinstance(payload, Mapping):
            raise GeneratorMalformedResponseError(
                "local generator JSON root must be an object"
            )
        return payload

    def inspect_runtime_metadata(self) -> RuntimeMetadata:
        """Inspect the selected Ollama runtime/model through local native APIs."""

        if self.runtime != "ollama":
            raise GeneratorConfigurationError(
                "runtime metadata inspection is implemented for Ollama only"
            )

        version_payload = self._request_json_url(
            "GET",
            f"{self._native_runtime_root}/api/version",
        )
        runtime_version = version_payload.get("version")
        if not isinstance(runtime_version, str) or not runtime_version:
            raise GeneratorMalformedResponseError(
                "Ollama /api/version response is missing version"
            )

        tags_payload = self._request_json_url(
            "GET",
            f"{self._native_runtime_root}/api/tags",
        )
        models = tags_payload.get("models")
        if not isinstance(models, Sequence) or isinstance(models, (str, bytes)):
            raise GeneratorMalformedResponseError(
                "Ollama /api/tags response must contain a models array"
            )
        selected: Mapping[str, object] | None = None
        for item in models:
            if not isinstance(item, Mapping):
                raise GeneratorMalformedResponseError(
                    "Ollama /api/tags model entries must be objects"
                )
            if item.get("name") == self.model_id or item.get("model") == self.model_id:
                selected = item
                break
        if selected is None:
            raise GeneratorModelUnavailableError(
                f"configured model {self.model_id!r} is not present in Ollama /api/tags"
            )

        digest = selected.get("digest")
        details = selected.get("details")
        if not isinstance(digest, str) or not digest:
            raise GeneratorMalformedResponseError(
                "Ollama model metadata is missing digest"
            )
        if not isinstance(details, Mapping):
            raise GeneratorMalformedResponseError(
                "Ollama model metadata is missing details"
            )
        quantization = details.get("quantization_level")
        if not isinstance(quantization, str) or not quantization:
            raise GeneratorMalformedResponseError(
                "Ollama model metadata is missing quantization_level"
            )

        ps_payload = self._request_json_url(
            "GET",
            f"{self._native_runtime_root}/api/ps",
        )
        loaded_models = ps_payload.get("models")
        if not isinstance(loaded_models, Sequence) or isinstance(
            loaded_models, (str, bytes)
        ):
            raise GeneratorMalformedResponseError(
                "Ollama /api/ps response must contain a models array"
            )
        loaded: Mapping[str, object] | None = None
        for item in loaded_models:
            if not isinstance(item, Mapping):
                raise GeneratorMalformedResponseError(
                    "Ollama /api/ps model entries must be objects"
                )
            if item.get("name") == self.model_id or item.get("model") == self.model_id:
                loaded = item
                break
        if loaded is None:
            raise GeneratorModelUnavailableError(
                f"configured model {self.model_id!r} is not loaded after readiness"
            )
        context_length = loaded.get("context_length")
        if type(context_length) is not int or context_length <= 0:
            raise GeneratorMalformedResponseError(
                "Ollama loaded model metadata has invalid context_length"
            )
        loaded_digest = loaded.get("digest")
        if not isinstance(loaded_digest, str) or loaded_digest != digest:
            raise GeneratorMalformedResponseError(
                "Ollama loaded model digest does not match installed model digest"
            )
        size_vram = loaded.get("size_vram")
        if size_vram is not None and (type(size_vram) is not int or size_vram < 0):
            raise GeneratorMalformedResponseError(
                "Ollama loaded model metadata has invalid size_vram"
            )

        return RuntimeMetadata(
            runtime=self.runtime,
            runtime_version=runtime_version,
            model_id=self.model_id,
            runtime_model_digest=digest,
            quantization=quantization,
            context_length=context_length,
            size_vram_bytes=size_vram,
        )

    def check_readiness(self) -> GeneratorReadiness:
        started = time.perf_counter()
        payload = self._request_json("GET", "/models")

        data = payload.get("data")
        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise GeneratorMalformedResponseError(
                "runtime /models response must contain a data array"
            )
        model_ids: list[str] = []
        for item in data:
            if not isinstance(item, Mapping):
                raise GeneratorMalformedResponseError(
                    "runtime /models data entries must be objects"
                )
            model_id = item.get("id")
            if not isinstance(model_id, str) or not model_id:
                raise GeneratorMalformedResponseError(
                    "runtime /models entry is missing a valid id"
                )
            model_ids.append(model_id)

        if self.model_id not in model_ids:
            raise GeneratorModelUnavailableError(
                f"configured model {self.model_id!r} is not listed by the local runtime"
            )

        probe_payload = self._request_json(
            "POST",
            "/chat/completions",
            json_body={
                "model": self.model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply with a short readiness acknowledgment.",
                    }
                ],
                "temperature": 0.0,
                "max_tokens": min(8, self.max_new_tokens),
                "seed": self.seed,
                "stream": False,
            },
        )
        _parse_completion_payload(
            probe_payload,
            expected_model_id=self.model_id,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return GeneratorReadiness(
            ready=True,
            model_id=self.model_id,
            runtime=self.runtime,
            response_ms=elapsed_ms,
        )

    def count_prompt_tokens(self, messages: Sequence[Mapping[str, str]]) -> int:
        """Use Ollama's native one-token probe on the same chat messages."""
        normalized = _validate_messages(messages)
        if self.runtime != "ollama":
            raise GeneratorConfigurationError("Exact prompt counting requires Ollama")
        payload = self._request_json_url(
            "POST", f"{self._native_runtime_root}/api/chat",
            json_body={"model": self.model_id, "messages": normalized, "stream": False,
                       "options": {"num_predict": 1, "temperature": self.temperature,
                                   "seed": self.seed}},
        )
        if payload.get("model") != self.model_id:
            raise GeneratorModelMismatchError("Prompt counter returned unexpected model identity")
        count = payload.get("prompt_eval_count")
        message = payload.get("message")
        if (type(count) is not int or count <= 0 or payload.get("done") is not True
                or type(payload.get("eval_count")) is not int
                or payload["eval_count"] != 1
                or payload.get("done_reason") not in ("length", "stop")
                or not isinstance(message, Mapping)
                or message.get("role") != "assistant"
                or not isinstance(message.get("content"), str)):
            raise GeneratorMalformedResponseError("Invalid Ollama prompt-count probe response")
        return count

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *, response_format: Mapping[str, object] | None = None,
    ) -> GeneratorResult:
        normalized_messages = _validate_messages(messages)
        request_body: dict[str, object] = {
            "model": self.model_id,
            "messages": normalized_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "seed": self.seed,
            "stream": False,
        }
        if response_format is not None:
            if dict(response_format) != {"type": "json_object"}:
                raise GeneratorInputError("Supported structured response format is json_object")
            request_body["response_format"] = dict(response_format)

        started = time.perf_counter()
        payload = self._request_json(
            "POST",
            "/chat/completions",
            json_body=request_body,
        )
        generation_ms = (time.perf_counter() - started) * 1000.0
        content, returned_model, usage = _parse_completion_payload(
            payload,
            expected_model_id=self.model_id,
        )

        return GeneratorResult(
            text=content,
            model_id=returned_model,
            usage=usage,
            generation_ms=generation_ms,
        )


def build_generator(
    config: "LLMConfig",
    *,
    session: requests.Session | None = None,
) -> OpenAICompatibleGenerator:
    """Build the single supported E4 local runtime profile from central config."""

    if config.provider != "local_openai_compatible":
        raise GeneratorConfigurationError(
            "E4 supports llm.provider='local_openai_compatible' only"
        )
    if config.runtime != "ollama":
        raise GeneratorConfigurationError("E4 supports llm.runtime='ollama' only")
    return OpenAICompatibleGenerator(
        model_id=config.model_id,
        base_url=config.base_url,
        runtime=config.runtime,
        temperature=config.temperature,
        max_new_tokens=config.max_new_tokens,
        seed=config.seed,
        connect_timeout_s=config.connect_timeout_s,
        read_timeout_s=config.read_timeout_s,
        session=session,
    )


def validate_runtime_metadata(config: "LLMConfig", observed: RuntimeMetadata) -> None:
    """Fail if declared reproducibility identity differs from the observed runtime."""

    if observed.runtime != config.runtime:
        raise GeneratorConfigurationError(
            f"runtime mismatch: configured {config.runtime!r}, observed {observed.runtime!r}"
        )
    if observed.runtime_version != config.runtime_version:
        raise GeneratorConfigurationError(
            "runtime version mismatch: "
            f"configured {config.runtime_version!r}, observed {observed.runtime_version!r}"
        )
    if observed.model_id != config.model_id:
        raise GeneratorConfigurationError(
            f"model mismatch: configured {config.model_id!r}, observed {observed.model_id!r}"
        )
    if not config.runtime_model_digest:
        raise GeneratorConfigurationError(
            "llm.runtime_model_digest must be set before reproducibility verification"
        )
    if not observed.runtime_model_digest.startswith(config.runtime_model_digest):
        raise GeneratorConfigurationError(
            "runtime model digest mismatch: configured digest is not a prefix of "
            f"observed {observed.runtime_model_digest!r}"
        )
    if observed.quantization != config.quantization:
        raise GeneratorConfigurationError(
            "quantization mismatch: "
            f"configured {config.quantization!r}, observed {observed.quantization!r}"
        )
    if observed.context_length != config.context_length:
        raise GeneratorConfigurationError(
            "context length mismatch: "
            f"configured {config.context_length}, observed {observed.context_length}"
        )


__all__ = [
    "Generator",
    "GeneratorConfigurationError",
    "GeneratorConnectionError",
    "GeneratorConnectTimeoutError",
    "GeneratorError",
    "GeneratorHTTPError",
    "GeneratorInfrastructureError",
    "GeneratorInputError",
    "GeneratorMalformedResponseError",
    "GeneratorModelMismatchError",
    "GeneratorModelUnavailableError",
    "GeneratorOutOfMemoryError",
    "GeneratorReadTimeoutError",
    "GeneratorReadiness",
    "GeneratorResult",
    "OpenAICompatibleGenerator",
    "RuntimeMetadata",
    "build_generator",
    "validate_runtime_metadata",
]
