"""Centralized configuration loading for the RAG SciFact project.

Third-party imports remain lazy so configuration can still be inspected with minimal startup coupling. The final Admin machine now has the verified E0 dependencies installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]


class ConfigurationError(RuntimeError):
    """Raised when project configuration cannot be loaded or validated."""


@dataclass(frozen=True)
class PathsConfig:
    corpus: Path
    queries: Path
    qrels_train: Path
    qrels_test: Path
    fixture_corpus: Path
    index_dir: Path
    log_dir: Path
    artifact_dir: Path


@dataclass(frozen=True)
class EmbeddingConfig:
    model_id: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    normalize: bool
    dtype: str


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str
    top_k: int
    max_top_k: int


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    runtime: str
    runtime_version: str
    reference_model_id: str
    model_id: str
    revision: str
    runtime_model_digest: str
    base_url: str
    context_length: int
    temperature: float
    max_new_tokens: int
    seed: int
    dtype: str
    quantization: str
    device: str
    connect_timeout_s: float
    read_timeout_s: float
    runtime_profile_locked: bool


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    filename_prefix: str


@dataclass(frozen=True)
class AppConfig:
    name: str
    seed: int
    paths: PathsConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    logging: LoggingConfig


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _env(name: str, default: Any) -> Any:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env", override=False)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigurationError(
            "PyYAML is not installed. Install the project environment with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except OSError as exc:
        raise ConfigurationError(f"Cannot read config file: {path}") from exc

    if not isinstance(data, Mapping):
        raise ConfigurationError(f"Config root must be a mapping: {path}")
    return data


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load config.yaml with optional .env/environment overrides."""

    _load_dotenv_if_available()

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    raw = _load_yaml(config_path.resolve())

    project = raw.get("project", {})
    paths = raw.get("paths", {})
    embedding = raw.get("embedding", {})
    retrieval = raw.get("retrieval", {})
    llm = raw.get("llm", {})
    logging_cfg = raw.get("logging", {})

    top_k = int(_env("TOP_K", retrieval.get("top_k", 5)))
    max_top_k = int(retrieval.get("max_top_k", 10))
    if not 1 <= top_k <= max_top_k:
        raise ConfigurationError(
            f"retrieval.top_k must be in range 1..{max_top_k}, got {top_k}"
        )

    chunk_size = int(
        _env("CHUNK_SIZE_TOKENS", embedding.get("chunk_size_tokens", 220))
    )
    chunk_overlap = int(
        _env("CHUNK_OVERLAP_TOKENS", embedding.get("chunk_overlap_tokens", 30))
    )
    if chunk_size <= 0:
        raise ConfigurationError("embedding.chunk_size_tokens must be > 0")
    if not 0 <= chunk_overlap < chunk_size:
        raise ConfigurationError(
            "embedding.chunk_overlap_tokens must be >= 0 and < chunk_size_tokens"
        )

    llm_context_length = int(
        _env("LLM_CONTEXT_LENGTH", llm.get("context_length", 32768))
    )
    if llm_context_length <= 0:
        raise ConfigurationError("llm.context_length must be > 0")

    return AppConfig(
        name=str(project.get("name", "rag-scifact")),
        seed=int(project.get("seed", 42)),
        paths=PathsConfig(
            corpus=_resolve_repo_path(str(paths["corpus"])),
            queries=_resolve_repo_path(str(paths["queries"])),
            qrels_train=_resolve_repo_path(str(paths["qrels_train"])),
            qrels_test=_resolve_repo_path(str(paths["qrels_test"])),
            fixture_corpus=_resolve_repo_path(
                str(paths.get("fixture_corpus", "data/fixtures/atlas.jsonl"))
            ),
            index_dir=_resolve_repo_path(str(paths.get("index_dir", "indexes"))),
            log_dir=_resolve_repo_path(str(paths.get("log_dir", "logs"))),
            artifact_dir=_resolve_repo_path(
                str(paths.get("artifact_dir", "artifacts"))
            ),
        ),
        embedding=EmbeddingConfig(
            model_id=str(
                _env(
                    "EMBEDDING_MODEL",
                    embedding.get(
                        "model_id", "sentence-transformers/all-MiniLM-L6-v2"
                    ),
                )
            ),
            chunk_size_tokens=chunk_size,
            chunk_overlap_tokens=chunk_overlap,
            normalize=_as_bool(embedding.get("normalize", True)),
            dtype=str(embedding.get("dtype", "float32")),
        ),
        retrieval=RetrievalConfig(
            mode=str(retrieval.get("mode", "dense")),
            top_k=top_k,
            max_top_k=max_top_k,
        ),
        llm=LLMConfig(
            provider=str(_env("LLM_PROVIDER", llm.get("provider", "local"))),
            runtime=str(_env("LLM_RUNTIME", llm.get("runtime", ""))),
            runtime_version=str(
                _env("LLM_RUNTIME_VERSION", llm.get("runtime_version", ""))
            ),
            reference_model_id=str(
                _env(
                    "LLM_REFERENCE_MODEL",
                    llm.get(
                        "reference_model_id",
                        "mistralai/Ministral-3-3B-Instruct-2512-BF16",
                    ),
                )
            ),
            model_id=str(_env("LLM_MODEL", llm.get("model_id", ""))),
            revision=str(_env("LLM_REVISION", llm.get("revision", ""))),
            runtime_model_digest=str(
                _env(
                    "LLM_RUNTIME_MODEL_DIGEST",
                    llm.get("runtime_model_digest", ""),
                )
            ),
            base_url=str(
                _env(
                    "LLM_BASE_URL",
                    llm.get("base_url", "http://127.0.0.1:11434/v1"),
                )
            ),
            context_length=llm_context_length,
            temperature=float(
                _env("LLM_TEMPERATURE", llm.get("temperature", 0))
            ),
            max_new_tokens=int(
                _env("LLM_MAX_NEW_TOKENS", llm.get("max_new_tokens", 512))
            ),
            seed=int(_env("LLM_SEED", llm.get("seed", 42))),
            dtype=str(_env("LLM_DTYPE", llm.get("dtype", ""))),
            quantization=str(
                _env("LLM_QUANTIZATION", llm.get("quantization", ""))
            ),
            device=str(_env("LLM_DEVICE", llm.get("device", ""))),
            connect_timeout_s=float(
                _env(
                    "LLM_CONNECT_TIMEOUT_S",
                    llm.get("connect_timeout_s", 10),
                )
            ),
            read_timeout_s=float(
                _env("LLM_READ_TIMEOUT_S", llm.get("read_timeout_s", 120))
            ),
            runtime_profile_locked=_as_bool(
                _env(
                    "LLM_RUNTIME_PROFILE_LOCKED",
                    llm.get("runtime_profile_locked", False),
                )
            ),
        ),
        logging=LoggingConfig(
            level=str(_env("LOG_LEVEL", logging_cfg.get("level", "INFO"))),
            filename_prefix=str(
                logging_cfg.get("filename_prefix", "rag-scifact")
            ),
        ),
    )


def ensure_output_directories(config: AppConfig) -> None:
    """Create writable runtime output directories if they do not exist."""

    for path in (
        config.paths.index_dir,
        config.paths.log_dir,
        config.paths.artifact_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

