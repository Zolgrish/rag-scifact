"""Reproducibility manifest helpers with section-preserving E1 updates."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
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


def current_git_state() -> tuple[str, bool]:
    """Return the source commit and dirty flag used to bind verification artifacts."""

    return _git_state()


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
            # E1 cannot establish an E4 runtime lock. Only verified E4 manifest
            # flows may set this true.
            "runtime_profile_locked": False,
        },
    )
    return merged


_RUNTIME_PROFILE_FIELDS = (
    "provider",
    "runtime",
    "runtime_version",
    "reference_model_id",
    "model_id",
    "revision",
    "runtime_model_digest",
    "base_url",
    "context_length",
    "temperature",
    "max_new_tokens",
    "seed",
    "dtype",
    "quantization",
    "device",
    "connect_timeout_s",
    "read_timeout_s",
)


def runtime_profile_from_config(config: AppConfig) -> dict[str, object]:
    """Return the canonical E4 runtime profile candidate from central config."""

    profile: dict[str, object] = {"schema_version": 1}
    for field in _RUNTIME_PROFILE_FIELDS:
        profile[field] = getattr(config.llm, field)
    return profile


def runtime_profile_sha256(profile: Mapping[str, object]) -> str:
    """Hash a normalized runtime profile independently from run artifacts."""

    canonical = json.dumps(
        dict(profile),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _legacy_locked_profile(llm: Mapping[str, object]) -> dict[str, object]:
    profile: dict[str, object] = {"schema_version": 1}
    for field in _RUNTIME_PROFILE_FIELDS:
        if field not in llm:
            raise ManifestConflictError(
                f"Locked runtime manifest is missing profile field {field!r}"
            )
        profile[field] = llm[field]
    return profile


def _locked_runtime_profile(
    existing: Mapping[str, object],
) -> tuple[dict[str, object], str] | None:
    freeze = existing.get("freeze")
    if not isinstance(freeze, Mapping) or freeze.get("runtime_profile_locked") is not True:
        return None
    llm = existing.get("llm")
    if not isinstance(llm, Mapping):
        raise ManifestConflictError("Locked runtime manifest is missing llm section")

    stored_profile = llm.get("locked_profile")
    stored_hash = llm.get("locked_profile_sha256")
    if stored_profile is None and stored_hash is None:
        profile = _legacy_locked_profile(llm)
        return profile, runtime_profile_sha256(profile)
    if not isinstance(stored_profile, Mapping):
        raise ManifestConflictError("llm.locked_profile must be an object")
    profile = dict(stored_profile)
    if profile.get("schema_version") != 1:
        raise ManifestConflictError("Unsupported llm.locked_profile schema_version")
    for field in _RUNTIME_PROFILE_FIELDS:
        if field not in profile:
            raise ManifestConflictError(
                f"llm.locked_profile is missing field {field!r}"
            )
    if not isinstance(stored_hash, str) or len(stored_hash) != 64:
        raise ManifestConflictError("llm.locked_profile_sha256 is invalid")
    actual_hash = runtime_profile_sha256(profile)
    if actual_hash != stored_hash:
        raise ManifestConflictError("llm.locked_profile SHA256 mismatch")
    return profile, stored_hash


def validate_runtime_profile_lock(
    existing: Mapping[str, object],
    config: AppConfig,
) -> str | None:
    """Enforce the manifest profile only when the caller requests locked mode."""

    if not config.llm.runtime_profile_locked:
        return None
    locked = _locked_runtime_profile(existing)
    if locked is None:
        raise ManifestConflictError(
            "runtime_profile_locked=true but the manifest has no locked runtime profile; "
            "verify unlocked and use the explicit relock flow first"
        )
    locked_profile, locked_hash = locked
    current_profile = runtime_profile_from_config(config)
    current_hash = runtime_profile_sha256(current_profile)
    if current_hash != locked_hash:
        differences = [
            field
            for field in _RUNTIME_PROFILE_FIELDS
            if current_profile.get(field) != locked_profile.get(field)
        ]
        raise ManifestConflictError(
            "Current runtime profile does not match the locked manifest profile; "
            "set runtime_profile_locked=false to experiment/reverify. Differing fields: "
            + ", ".join(differences)
        )
    return locked_hash


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_observed_runtime(
    config: AppConfig,
    observed: Mapping[str, object],
) -> None:
    exact = {
        "runtime": config.llm.runtime,
        "runtime_version": config.llm.runtime_version,
        "model_id": config.llm.model_id,
        "quantization": config.llm.quantization,
        "context_length": config.llm.context_length,
    }
    for key, expected in exact.items():
        if observed.get(key) != expected:
            raise ManifestConflictError(
                f"Observed runtime mismatch for {key}: expected {expected!r}, "
                f"got {observed.get(key)!r}"
            )
    digest = observed.get("runtime_model_digest")
    if not isinstance(digest, str) or not digest.startswith(config.llm.runtime_model_digest):
        raise ManifestConflictError(
            "Observed runtime model digest does not match configured digest/prefix"
        )
    size_vram = observed.get("size_vram_bytes")
    if size_vram is not None and (type(size_vram) is not int or size_vram < 0):
        raise ManifestConflictError("Observed runtime size_vram_bytes is invalid")


def _validate_e4_verification(
    config: AppConfig,
    verification: Mapping[str, object],
) -> None:
    for key in ("status", "readiness", "smoke"):
        if verification.get(key) != "PASS":
            raise ManifestConflictError(
                f"Cannot update runtime provenance without {key} PASS"
            )

    profile = runtime_profile_from_config(config)
    profile_hash = runtime_profile_sha256(profile)
    if verification.get("profile") != profile:
        raise ManifestConflictError("E4 verification profile does not match current config")
    if verification.get("profile_sha256") != profile_hash:
        raise ManifestConflictError("E4 verification profile SHA256 mismatch")
    if verification.get("seed_parameter_accepted") is not True:
        raise ManifestConflictError(
            "E4 verification must record that the configured seed parameter was accepted"
        )

    observed = verification.get("observed_runtime")
    if not isinstance(observed, Mapping):
        raise ManifestConflictError("E4 verification observed_runtime is missing")
    _validate_observed_runtime(config, observed)

    artifact = verification.get("artifact")
    artifact_sha256 = verification.get("artifact_sha256")
    if not isinstance(artifact, str) or not artifact:
        raise ManifestConflictError("E4 verification artifact path is missing")
    if not isinstance(artifact_sha256, str) or len(artifact_sha256) != 64:
        raise ManifestConflictError("E4 verification artifact SHA256 is invalid")

    artifact_path = (REPO_ROOT / artifact).resolve()
    try:
        artifact_path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ManifestConflictError(
            "E4 verification artifact must be inside the repository"
        ) from exc
    if not artifact_path.is_file():
        raise ManifestConflictError(
            f"E4 verification artifact does not exist: {artifact}"
        )
    if _sha256_file(artifact_path) != artifact_sha256:
        raise ManifestConflictError("E4 verification artifact SHA256 mismatch")

    try:
        artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestConflictError("Cannot read E4 verification artifact") from exc
    if not isinstance(artifact_payload, Mapping):
        raise ManifestConflictError("E4 verification artifact root must be an object")

    artifact_expected = {
        "status": "PASS",
        "ready": True,
        "profile": profile,
        "profile_sha256": profile_hash,
        "observed_runtime": dict(observed),
        "seed_parameter_accepted": True,
        "config_runtime_profile_locked": config.llm.runtime_profile_locked,
    }
    for key, expected in artifact_expected.items():
        if artifact_payload.get(key) != expected:
            raise ManifestConflictError(
                f"E4 verification artifact mismatch for {key!r}"
            )

    smoke = artifact_payload.get("smoke")
    if not isinstance(smoke, Mapping):
        raise ManifestConflictError("E4 verification artifact is missing smoke data")
    if smoke.get("model_id") != config.llm.model_id:
        raise ManifestConflictError("E4 smoke model identity mismatch")
    if smoke.get("exact_text_match") is not True:
        raise ManifestConflictError("E4 deterministic smoke did not match exactly")


def merge_e4_runtime_manifest(
    existing: Mapping[str, object],
    *,
    config: AppConfig,
    verification: Mapping[str, object],
    relock: bool = False,
) -> dict[str, object]:
    """Refresh or explicitly replace the manifest-authoritative runtime profile."""

    _validate_e4_verification(config, verification)
    current_profile = runtime_profile_from_config(config)
    current_hash = runtime_profile_sha256(current_profile)

    old_freeze = existing.get("freeze")
    final_test_frozen = (
        isinstance(old_freeze, Mapping)
        and old_freeze.get("final_test_frozen") is True
    )
    if relock:
        if config.llm.runtime_profile_locked:
            raise ManifestConflictError(
                "Explicit runtime relock requires runtime_profile_locked=false while "
                "verifying the candidate profile"
            )
        if final_test_frozen:
            raise ManifestConflictError(
                "Cannot replace runtime profile after final_test_frozen=true"
            )
    else:
        if not config.llm.runtime_profile_locked:
            raise ManifestConflictError(
                "Ordinary manifest refresh requires runtime_profile_locked=true; "
                "use --relock-runtime-profile to replace the locked profile"
            )
        validate_runtime_profile_lock(existing, config)

    merged = deepcopy(dict(existing))
    now = datetime.now(timezone.utc).isoformat()
    git_commit, git_dirty = _git_state()
    runtime, hardware = _runtime_hardware()
    _validate_frozen_manifest_identity(
        existing,
        current_git_commit=git_commit,
        current_git_dirty=git_dirty,
    )

    if config.llm.device and hardware.get("gpu_name"):
        if hardware["gpu_name"] != config.llm.device:
            raise ManifestConflictError(
                "Configured llm.device does not match observed GPU: "
                f"{config.llm.device!r} != {hardware['gpu_name']!r}"
            )

    merged.setdefault("schema_version", 1)
    merged.setdefault("created_at", now)
    merged["updated_at"] = now
    merged["git_commit"] = git_commit
    merged["git_dirty"] = git_dirty

    old_llm = merged.get("llm")
    llm_merged = dict(old_llm) if isinstance(old_llm, Mapping) else {}
    previous_locked = _locked_runtime_profile(existing)
    if relock and previous_locked is not None and previous_locked[1] != current_hash:
        history = llm_merged.get("profile_history")
        profile_history = list(history) if isinstance(history, list) else []
        profile_history.append(
            {
                "profile_sha256": previous_locked[1],
                "profile": deepcopy(previous_locked[0]),
                "replaced_at": now,
            }
        )
        llm_merged["profile_history"] = profile_history

    for field in _RUNTIME_PROFILE_FIELDS:
        llm_merged[field] = current_profile[field]
    llm_merged["locked_profile"] = deepcopy(current_profile)
    llm_merged["locked_profile_sha256"] = current_hash
    llm_merged["verification"] = deepcopy(dict(verification))
    merged["llm"] = llm_merged

    runtime.update(
        {
            "llm_runtime": config.llm.runtime,
            "llm_runtime_version": config.llm.runtime_version,
            "llm_base_url": config.llm.base_url,
        }
    )
    old_runtime = merged.get("runtime")
    runtime_merged = dict(old_runtime) if isinstance(old_runtime, Mapping) else {}
    runtime_merged.update(runtime)
    merged["runtime"] = runtime_merged

    if hardware:
        old_hardware = merged.get("hardware")
        hardware_merged = (
            dict(old_hardware) if isinstance(old_hardware, Mapping) else {}
        )
        hardware_merged.update(hardware)
        merged["hardware"] = hardware_merged

    freeze = dict(old_freeze) if isinstance(old_freeze, Mapping) else {}
    freeze.setdefault("final_test_frozen", False)
    freeze["runtime_profile_locked"] = True
    merged["freeze"] = freeze
    return merged

def validate_e5_manifest(existing: Mapping[str, object], *, config: AppConfig) -> None:
    """Enforce prompt/config identity and source identity after final-test freeze."""
    from app.prompt import prompt_identity

    freeze = existing.get("freeze", {})
    if not isinstance(freeze, Mapping) or freeze.get("final_test_frozen") is not True:
        return
    if existing.get("prompt") != prompt_identity():
        raise ManifestConflictError("Frozen prompt identity differs from E5 contract")
    if existing.get("prompt_version") != prompt_identity()["version"]:
        raise ManifestConflictError("Frozen prompt version differs from E5 contract")
    retrieval = {"mode": config.retrieval.mode, "top_k": config.retrieval.top_k,
                 "max_top_k": config.retrieval.max_top_k}
    old_retrieval = existing.get("retrieval")
    if (not isinstance(old_retrieval, Mapping)
            or any(old_retrieval.get(k) != v for k, v in retrieval.items())):
        raise ManifestConflictError("Frozen retrieval config differs from E5 config")
    if not config.llm.runtime_profile_locked:
        raise ManifestConflictError("Frozen E5 requires the E4 runtime profile lock")
    validate_runtime_profile_lock(existing, config)
    commit, dirty = _git_state()
    _validate_frozen_manifest_identity(existing, current_git_commit=commit,
                                       current_git_dirty=dirty)


def merge_e5_prompt_manifest(existing: Mapping[str, object], *, config: AppConfig,
                             verification: Mapping[str, object]) -> dict[str, object]:
    """Record verified E5 provenance without modifying any E4 lock fields."""
    from app.prompt import prompt_identity

    if not config.llm.runtime_profile_locked:
        raise ManifestConflictError("E5 manifest update requires runtime_profile_locked=true")
    profile_hash = validate_runtime_profile_lock(existing, config)
    validate_e5_manifest(existing, config=config)
    identity = dict(prompt_identity())
    artifact_name = verification.get("artifact")
    if not isinstance(artifact_name, str):
        raise ManifestConflictError("E5 verification requires an artifact")
    artifact = (REPO_ROOT / artifact_name).resolve()
    if not artifact.is_relative_to(REPO_ROOT.resolve()) or not artifact.is_file():
        raise ManifestConflictError("E5 verification artifact must exist inside repository")
    data = artifact.read_bytes()
    if hashlib.sha256(data).hexdigest() != verification.get("artifact_sha256"):
        raise ManifestConflictError("E5 verification artifact hash mismatch")
    try:
        payload = json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise ManifestConflictError("Malformed E5 verification artifact") from exc
    if (not isinstance(payload, Mapping) or payload.get("status") != "PASS"
            or payload.get("prompt") != identity
            or payload.get("profile_sha256") != profile_hash):
        raise ManifestConflictError("E5 verification identity mismatch")
    commit, dirty = _git_state()
    if (payload.get("source_git_commit") != commit
            or payload.get("source_git_dirty") is not dirty):
        raise ManifestConflictError(
            "E5 verification source git state differs from current source state"
        )
    observed = payload.get("observed_runtime")
    if not isinstance(observed, Mapping):
        raise ManifestConflictError("E5 verification requires observed runtime")
    _validate_observed_runtime(config, observed)
    counter = payload.get("token_counter_check")
    if (not isinstance(counter, Mapping)
            or type(counter.get("native_prompt_tokens")) is not int
            or counter["native_prompt_tokens"] <= 0
            or type(counter.get("completion_prompt_tokens")) is not int
            or counter["native_prompt_tokens"] != counter["completion_prompt_tokens"]):
        raise ManifestConflictError("E5 exact token-counter verification failed")
    trace = payload.get("trace")
    response = payload.get("response")
    if (not isinstance(trace, Mapping) or not isinstance(response, Mapping)
            or trace.get("model_id") != config.llm.model_id
            or response.get("model_id") != config.llm.model_id
            or type(trace.get("input_token_count")) is not int
            or not 0 < trace["input_token_count"] <= config.llm.context_length - config.llm.max_new_tokens
            or response.get("status") not in {"ANSWERED", "INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"}):
        raise ManifestConflictError("E5 verification requires a successful budgeted RAG smoke")
    merged = deepcopy(dict(existing))
    _validate_frozen_manifest_identity(existing, current_git_commit=commit,
                                       current_git_dirty=dirty)
    merged.update({"prompt": identity, "prompt_version": identity["version"],
                   "updated_at": datetime.now(timezone.utc).isoformat(),
                   "git_commit": commit, "git_dirty": dirty,
                   "rag_verification": deepcopy(dict(verification))})
    return merged


__all__ = [
    "current_git_state",
    "merge_e5_prompt_manifest",
    "validate_e5_manifest",
    "ManifestConflictError",
    "ManifestError",
    "load_manifest",
    "merge_e1_manifest",
    "merge_e4_runtime_manifest",
    "runtime_profile_from_config",
    "runtime_profile_sha256",
    "validate_runtime_profile_lock",
    "write_manifest_atomic",
]
