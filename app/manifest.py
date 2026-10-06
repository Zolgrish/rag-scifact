"""Reproducibility manifest helpers with section-preserving E1 updates."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Mapping

from app.config import AppConfig, REPO_ROOT
from app.data_audit import DatasetSplit


class ManifestError(RuntimeError):
    """Raised when a manifest cannot be read or safely updated."""


class ManifestConflictError(ManifestError):
    """Raised when existing frozen E1 identity conflicts with current data."""


def load_manifest(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    if not resolved.exists():
        return {}
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Cannot read manifest {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"Manifest root must be a JSON object: {resolved}")
    return value


def write_manifest_atomic(path: str | Path, payload: Mapping[str, object]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=resolved.name + ".",
        suffix=".tmp",
        dir=resolved.parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, resolved)
    finally:
        tmp = Path(tmp_name)
        if tmp.exists():
            tmp.unlink()


def _git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "", True


def _runtime_hardware() -> tuple[dict[str, object], dict[str, object]]:
    runtime: dict[str, object] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    hardware: dict[str, object] = {}
    try:
        import torch
    except ImportError:
        return runtime, hardware

    runtime.update(
        {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        }
    )
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        hardware = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_vram_gib": round(props.total_memory / (1024**3), 2),
        }
    return runtime, hardware


def _conflict_check(
    existing: Mapping[str, object],
    *,
    dataset: Mapping[str, object],
    split: Mapping[str, object],
) -> None:
    old_dataset = existing.get("dataset")
    if isinstance(old_dataset, Mapping):
        old_hashes = old_dataset.get("hashes")
        new_hashes = dataset.get("hashes")
        if old_hashes is not None and old_hashes != new_hashes:
            raise ManifestConflictError(
                "Existing manifest dataset hashes differ from current E1 audit"
            )

    old_split = existing.get("split")
    if isinstance(old_split, Mapping):
        for key in ("seed", "dev_ids", "practice_ids"):
            if key in old_split and old_split[key] != split[key]:
                raise ManifestConflictError(
                    f"Existing manifest split field {key!r} differs from E1 result"
                )


def _validate_frozen_manifest_identity(
    existing: Mapping[str, object],
    *,
    current_git_commit: str,
    current_git_dirty: bool,
) -> None:
    freeze = existing.get("freeze")
    if not isinstance(freeze, Mapping):
        return
    if not bool(freeze.get("final_test_frozen", False)):
        return

    frozen_commit = existing.get("git_commit")
    frozen_dirty = existing.get("git_dirty")

    if not isinstance(frozen_commit, str) or not frozen_commit:
        raise ManifestConflictError(
            "Frozen manifest is missing a valid git_commit"
        )
    if frozen_dirty is not False:
        raise ManifestConflictError(
            "Frozen manifest must record git_dirty=false"
        )
    if current_git_dirty:
        raise ManifestConflictError(
            "Cannot update a frozen manifest from a dirty working tree"
        )
    if current_git_commit != frozen_commit:
        raise ManifestConflictError(
            "Cannot update a frozen manifest from a different git commit: "
            f"frozen={frozen_commit}, current={current_git_commit}"
        )


def merge_e1_manifest(
    existing: Mapping[str, object],
    *,
    config: AppConfig,
    split: DatasetSplit,
    audit: Mapping[str, object],
) -> dict[str, object]:
    """Merge E1-owned sections and preserve all unrelated sections."""

    merged = deepcopy(dict(existing))
    now = datetime.now(timezone.utc).isoformat()
    git_commit, git_dirty = _git_state()
    runtime, hardware = _runtime_hardware()

    counts = audit["counts"]
    hashes = audit["hashes"]
    assert isinstance(counts, Mapping)
    assert isinstance(hashes, Mapping)

    dataset_section = {
        "corpus_count": counts["corpus_count"],
        "query_count": counts["query_count"],
        "train_query_count": counts["train_query_count"],
        "test_query_count": counts["test_query_count"],
        "train_qrel_row_count": counts["train_qrel_row_count"],
        "test_qrel_row_count": counts["test_qrel_row_count"],
        "hashes": deepcopy(hashes),
    }
    split_section = {
        "seed": split.seed,
        "dev_ids": list(split.dev_ids),
        "practice_ids": list(split.practice_ids),
    }
    _conflict_check(existing, dataset=dataset_section, split=split_section)
    _validate_frozen_manifest_identity(
        existing,
        current_git_commit=git_commit,
        current_git_dirty=git_dirty,
    )

    merged.setdefault("schema_version", 1)
    merged.setdefault("created_at", now)
    merged["updated_at"] = now
    merged["git_commit"] = git_commit
    merged["git_dirty"] = git_dirty

    # Preserve extra later-stage keys nested in these sections.
    old_dataset = merged.get("dataset")
    dataset_merged = dict(old_dataset) if isinstance(old_dataset, Mapping) else {}
    dataset_merged.update(dataset_section)
    merged["dataset"] = dataset_merged

    old_split = merged.get("split")
    split_merged = dict(old_split) if isinstance(old_split, Mapping) else {}
    split_merged.update(split_section)
    merged["split"] = split_merged

    merged["data_integrity"] = {
        "status": audit["status"],
        "validation": deepcopy(audit["validation"]),
        "query_overlap_audit": deepcopy(audit["query_overlap_audit"]),
    }

    merged.setdefault(
        "chunking",
        {
            "tokenizer": config.embedding.model_id,
            "chunk_size_tokens": config.embedding.chunk_size_tokens,
            "overlap_tokens": config.embedding.chunk_overlap_tokens,
        },
    )
    merged.setdefault(
        "embedding",
        {
            "model_id": config.embedding.model_id,
            "dtype": config.embedding.dtype,
            "normalize": config.embedding.normalize,
        },
    )
    merged.setdefault("index", {"status": "not_built_e1"})
    merged.setdefault(
        "llm",
        {
            "provider": config.llm.provider,
            "runtime": config.llm.runtime,
            "runtime_version": config.llm.runtime_version,
            "reference_model_id": config.llm.reference_model_id,
            "model_id": config.llm.model_id,
            "revision": config.llm.revision,
            "runtime_model_digest": config.llm.runtime_model_digest,
            "context_length": config.llm.context_length,
            "temperature": config.llm.temperature,
            "max_new_tokens": config.llm.max_new_tokens,
            "seed": config.llm.seed,
            "dtype": config.llm.dtype,
            "quantization": config.llm.quantization,
            "device": config.llm.device,
        },
    )
    runtime.update(
        {
            "llm_runtime": config.llm.runtime,
            "llm_runtime_version": config.llm.runtime_version,
            "llm_base_url": config.llm.base_url,
        }
    )
    merged.setdefault("runtime", runtime)
    merged.setdefault("hardware", hardware)
    merged.setdefault("prompt_version", "not_implemented_e1")
    merged.setdefault(
        "freeze",
        {
            "final_test_frozen": False,
            "runtime_profile_locked": config.llm.runtime_profile_locked,
        },
    )
    return merged


__all__ = [
    "ManifestConflictError",
    "ManifestError",
    "load_manifest",
    "merge_e1_manifest",
    "write_manifest_atomic",
]
