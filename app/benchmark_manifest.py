"""E7 provenance verification and one-way pre-test sealing, without inference."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Mapping

from app.config import AppConfig, REPO_ROOT, effective_config_identity, load_config
from app.data_audit import (EXPECTED_FILE_SHA256, build_dev_practice_split, file_sha256)
from app.indexer import IndexBundle, load_bundle
from app.loader import (BenchmarkQuery, load_corpus, load_queries, load_qrels,
                        qrel_query_ids, validate_qrel_references)
from app.prompt import prompt_identity
from app.retrieval_evaluator import (RetrievalEvaluationError, aggregate, failure_candidates,
    finite_number, METRIC_NAMES, metric_policy_identity, parse_retrieval_rows, strict_json)


@dataclass(frozen=True)
class EvaluationData:
    queries: tuple[BenchmarkQuery, ...]
    gold: dict[str, dict[str, float]]
    dataset_identity: dict[str, object]
    split_identity: dict[str, object]
    corpus: Mapping[str, object]


def identity(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def repo_path(value: str | Path) -> Path:
    path = (REPO_ROOT / value).resolve()
    if not path.is_relative_to(REPO_ROOT.resolve()):
        raise RetrievalEvaluationError("Benchmark artifacts and inputs must be inside repository")
    return path


def artifact_record(path: str | Path) -> dict[str, str]:
    path = repo_path(path)
    return {"artifact": path.relative_to(REPO_ROOT.resolve()).as_posix(), "sha256": file_sha256(path)}


def verified_artifact(record: object) -> tuple[Path, bytes]:
    if not isinstance(record, Mapping) or not isinstance(record.get("artifact"), str):
        raise RetrievalEvaluationError("Missing benchmark artifact path/hash")
    path = repo_path(record["artifact"])
    if not path.is_file() or file_sha256(path) != record.get("sha256"):
        raise RetrievalEvaluationError("Benchmark artifact missing or SHA256 mismatch")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise RetrievalEvaluationError("Benchmark artifact changed during verification")
    return path, data


def retrieval_identity(config: AppConfig) -> dict[str, object]:
    if (config.retrieval.mode != "dense" or config.retrieval.top_k != 5
            or config.retrieval.max_top_k != 10):
        raise RetrievalEvaluationError("E7 dense baseline requires serving Top5 and evaluation/max depth 10")
    expected = {"model_id": "sentence-transformers/all-MiniLM-L6-v2", "chunk_size_tokens": 220,
                "chunk_overlap_tokens": 30, "normalize": True, "dtype": "float32"}
    if any(getattr(config.embedding, key) != value for key, value in expected.items()):
        raise RetrievalEvaluationError("Locked embedding/chunk configuration mismatch")
    return {"mode": "dense", "serving_top_k": 5, "evaluation_depth": 10, "max_top_k": 10}


def load_evaluation_data(config: AppConfig, manifest: Mapping[str, object], *, split: str = "dev") -> EvaluationData:
    """Dev/freeze never read test qrels. Test access is behind the complete gate."""
    if split not in ("dev", "test"):
        raise RetrievalEvaluationError("Only dev and gated test splits are supported")
    if split == "test":
        validate_test_gate(manifest, config=config)
    retrieval_identity(config)
    counts = {"corpus_count": 5183, "query_count": 1109, "train_query_count": 809, "test_query_count": 300}
    dataset = manifest.get("dataset", {})
    if not isinstance(dataset, Mapping) or any(dataset.get(k) != v for k, v in counts.items()) or config.seed != 42:
        raise RetrievalEvaluationError("Invalid canonical E1 dataset counts/seed")
    if manifest.get("data_integrity", {}).get("status") != "PASS":
        raise RetrievalEvaluationError("Canonical E1 integrity audit is missing or invalid")
    hashes = {}
    for name, expected in EXPECTED_FILE_SHA256.items():
        path = repo_path(getattr(config.paths, name))
        entry = dataset.get("hashes", {}).get(name, {})
        if (entry.get("sha256") != expected or repo_path(entry.get("path", "")) != path):
            raise RetrievalEvaluationError("Canonical dataset identity mismatch")
        # The sealed E1 expected test hash is metadata until after the test gate.
        if name != "qrels_test" or split == "test":
            if file_sha256(path) != expected:
                raise RetrievalEvaluationError(f"Canonical dataset hash mismatch: {name}")
        hashes[name] = {"path": path.relative_to(REPO_ROOT.resolve()).as_posix(), "sha256": expected}
    corpus = load_corpus(config.paths.corpus)
    queries = load_queries(config.paths.queries)
    train = load_qrels(config.paths.qrels_train)
    if len(corpus) != 5183 or len(queries) != 1109 or len(qrel_query_ids(train)) != 809:
        raise RetrievalEvaluationError("Canonical corpus/query/train counts mismatch")
    validate_qrel_references(train, queries=queries, corpus=corpus, source_name="train")
    actual_split = build_dev_practice_split(train)
    split_state = {"seed": 42, "dev_ids": list(actual_split.dev_ids), "practice_ids": list(actual_split.practice_ids)}
    recorded_split = manifest.get("split", {})
    if (not isinstance(recorded_split, Mapping) or len(actual_split.dev_ids) != 100
            or len(actual_split.practice_ids) != 709
            or any(recorded_split.get(k) != v for k, v in split_state.items())):
        raise RetrievalEvaluationError("Canonical E1 split differs from train-derived split")
    qrels = train
    ids = actual_split.dev_ids
    if split == "test":
        qrels = load_qrels(config.paths.qrels_test)
        test_ids = qrel_query_ids(qrels)
        if len(test_ids) != 300 or test_ids & qrel_query_ids(train):
            raise RetrievalEvaluationError("Invalid official test membership/count")
        validate_qrel_references(qrels, queries=queries, corpus=corpus, source_name="test")
        ids = tuple(sorted(test_ids, key=int))
    gold = {qid: {} for qid in ids}
    for row in qrels:
        if row.query_id in gold:
            gold[row.query_id][row.doc_id] = row.relevance
    split_payload = {"name": split, "seed": 42, "query_ids": list(ids),
                     "canonical_train_split_sha256": identity(split_state)}
    return EvaluationData(tuple(queries[qid] for qid in ids), gold, {"counts": counts, "hashes": hashes},
                          {**split_payload, "sha256": identity(split_payload)}, corpus)


def index_identity(config: AppConfig, manifest: Mapping[str, object],
                   corpus: Mapping[str, object]) -> tuple[IndexBundle, dict[str, object]]:
    path = repo_path(config.paths.index_dir / "scifact")
    records = {name: artifact_record(path / name) for name in ("index_manifest.json", "index.faiss", "chunks.jsonl")}
    recorded = manifest.get("index", {})
    if (not isinstance(recorded, Mapping) or recorded.get("status") != "built"
            or recorded.get("manifest_sha256") != records["index_manifest.json"]["sha256"]
            or repo_path(recorded.get("bundle_dir", "")) != path
            or any(recorded.get("files", {}).get(name, {}).get("sha256") != records[name]["sha256"]
                   for name in ("index.faiss", "chunks.jsonl"))):
        raise RetrievalEvaluationError("SciFact bundle differs from canonical index identity")
    bundle = load_bundle(path, expected_corpus_sha256=EXPECTED_FILE_SHA256["corpus"])
    if (bundle.manifest["corpus"].get("documents") != 5183
            or {c.doc_id for c in bundle.chunks} != set(corpus)):
        raise RetrievalEvaluationError("Index must cover the full canonical SciFact corpus")
    for chunk in bundle.chunks:
        doc = corpus[chunk.doc_id]
        if chunk.title != doc.title or doc.text[chunk.char_start:chunk.char_end] != chunk.text:
            raise RetrievalEvaluationError("Index mapping is not canonical source evidence")
    return bundle, {"bundle_path": path.relative_to(REPO_ROOT.resolve()).as_posix(), "files": records}


def canonical_generation_timing(manifest: Mapping[str, object], config: AppConfig) -> dict[str, object]:
    """Reuse canonical E6 fixture timing, labelled with its own source and scope."""
    section = manifest.get("generation_fixture", {})
    if not isinstance(section, Mapping):
        raise RetrievalEvaluationError("Canonical E6 generation timing is missing")
    _, data = verified_artifact(section.get("summary"))
    summary = strict_json(data)
    from app.manifest import runtime_profile_from_config, runtime_profile_sha256
    from app.evaluator import suite_identity
    if (not isinstance(summary, Mapping) or summary.get("status") != "PASS" or summary.get("cases") != 6
            or summary.get("passed") != 6 or summary.get("failed") != 0
            or summary.get("source_git_commit") != manifest.get("git_commit")
            or summary.get("source_git_dirty") is not False or summary.get("prompt") != prompt_identity()
            or summary.get("effective_config") != effective_config_identity(config)
            or summary.get("fixture_suite") != suite_identity()
            or summary.get("runtime_profile_sha256") != runtime_profile_sha256(runtime_profile_from_config(config))):
        raise RetrievalEvaluationError("Canonical E6 timing provenance differs from current config/prompt")
    return {"scope": "canonical_Atlas_fixtures_not_SciFact_generation", "summary": deepcopy(section["summary"]),
            "source_git_commit": summary["source_git_commit"], "fixture_suite": summary["fixture_suite"],
            "latency_ms": summary["latency_ms"], "latency_samples": summary["latency_samples"]}


def verify_dev_artifact(manifest: Mapping[str, object], *, config: AppConfig,
                        verification: Mapping[str, object]) -> dict[str, object]:
    """Reconstruct all 100 queries and scores; a claimed metrics PASS is insufficient."""
    from app.manifest import current_git_state
    commit, dirty = current_git_state()
    if not commit or dirty:
        raise RetrievalEvaluationError("Canonical E7 requires a clean committed source tree")
    if manifest.get("git_commit") != commit or manifest.get("git_dirty") is not False:
        raise RetrievalEvaluationError("Manifest source is stale; refresh canonical E5/E6 first")
    path, data = verified_artifact(verification.get("metrics"))
    summary = strict_json(data)
    if not isinstance(summary, dict):
        raise RetrievalEvaluationError("Malformed metrics summary")
    inputs = load_evaluation_data(config, manifest)
    bundle, index = index_identity(config, manifest, inputs.corpus)
    if (type(summary.get("schema_version")) is not int
            or any(type(summary.get(key)) is not int or summary[key] < 0 for key in
                   ("sample_size", "denominator", "failure_count", "failure_candidate_count"))
            or not isinstance(summary.get("metrics"), Mapping) or set(summary["metrics"]) != set(METRIC_NAMES)
            or any(not finite_number(value, nonnegative=True) or value > 1 for value in summary["metrics"].values())):
        raise RetrievalEvaluationError("Invalid metrics schema/count/number types")
    runtime = summary.get("runtime", {})
    embedding = runtime.get("embedding", {}) if isinstance(runtime, Mapping) else {}
    if (not isinstance(embedding, Mapping) or not isinstance(embedding.get("device"), str) or not embedding["device"]
            or not isinstance(runtime.get("python"), str) or not isinstance(runtime.get("platform"), str)
            or type(runtime.get("faiss_threads")) is not int or runtime["faiss_threads"] < 1
            or not isinstance(runtime.get("versions"), Mapping)
            or any(not isinstance(runtime["versions"].get(name), str) for name in
                   ("numpy", "faiss-cpu", "sentence-transformers", "torch"))
            or any(type(embedding.get(key)) is not type(bundle.manifest["embedding"].get(key))
                   or embedding.get(key) != bundle.manifest["embedding"].get(key) for key in
                   ("model_id", "revision", "output_dimension", "dtype", "normalized", "effective_model_input_limit"))):
        raise RetrievalEvaluationError("Missing/incompatible retrieval runtime provenance")
    _, raw_config = verified_artifact(summary.get("config"))
    config_path = repo_path(summary["config"]["artifact"])
    if effective_config_identity(load_config(config_path)) != effective_config_identity(config):
        raise RetrievalEvaluationError("Raw config resolves to a different effective config")
    if not raw_config:
        raise RetrievalEvaluationError("Empty config artifact")
    expected = {"schema_version": 1, "status": "COMPLETED", "split": "dev", "source_git_commit": commit,
                "source_git_dirty": False, "effective_config": effective_config_identity(config),
                "retrieval": retrieval_identity(config), "split_identity": inputs.split_identity,
                "dataset": inputs.dataset_identity, "index": index, "metric_policy": metric_policy_identity(),
                "generation_timing": canonical_generation_timing(manifest, config)}
    if any(summary.get(k) != v for k, v in expected.items()) or summary.get("source_git_dirty") is not False:
        raise RetrievalEvaluationError("Metrics source/config/split/index/policy provenance mismatch")
    run_path, run_data = verified_artifact(summary.get("retrieval_run"))
    failure_path, failure_data = verified_artifact(summary.get("failure_analysis"))
    if (run_path.parent != path.parent or failure_path.parent != path.parent
            or run_path.name != "retrieval_run.jsonl" or failure_path.name != "failure_analysis.jsonl"
            or path.name != "metrics.json" or not isinstance(summary.get("run_id"), str)
            or not summary["run_id"] or path.parent.name != summary["run_id"]):
        raise RetrievalEvaluationError("Immutable run artifact path/identity mismatch")
    rows = [strict_json(line) for line in run_data.splitlines()]
    outcomes = parse_retrieval_rows(rows, [q.query_id for q in inputs.queries], summary["run_id"])
    chunks = {c.chunk_id: c.doc_id for c in bundle.chunks}
    if any(chunks.get(r.chunk_id) != r.doc_id for o in outcomes for r in o.results):
        raise RetrievalEvaluationError("Retrieval document/chunk IDs differ from canonical mapping")
    computed = aggregate(outcomes, inputs.gold)
    if any(summary.get(k) != v for k, v in computed.items()) or computed["denominator"] != 100:
        raise RetrievalEvaluationError("Metrics/counts/timing differ from raw-run recomputation")
    candidates = failure_candidates(outcomes, inputs.queries, inputs.gold)
    if [strict_json(line) for line in failure_data.splitlines()] != candidates:
        raise RetrievalEvaluationError("Failure analysis differs from raw-run reconstruction")
    if summary.get("failure_candidate_count") != len(candidates):
        raise RetrievalEvaluationError("Failure candidate count mismatch")
    if current_git_state() != (commit, False):
        raise RetrievalEvaluationError("Source changed during E7 artifact verification")
    return {"schema_version": 1, "split": "dev", "source_git_commit": commit, "source_git_dirty": False,
            "metrics_artifact": artifact_record(path), "retrieval_run": deepcopy(summary["retrieval_run"]),
            "failure_analysis": deepcopy(summary["failure_analysis"]), "metric_policy": metric_policy_identity(),
            **computed}


def verify_canonical_generation(manifest: Mapping[str, object], config: AppConfig) -> None:
    from app.manifest import (merge_e5_prompt_manifest, merge_e6_generation_manifest,
                             validate_runtime_profile_lock)
    if not config.llm.runtime_profile_locked or manifest.get("freeze", {}).get("runtime_profile_locked") is not True:
        raise RetrievalEvaluationError("Pre-test baseline requires the E4 runtime lock")
    validate_runtime_profile_lock(manifest, config)
    merge_e5_prompt_manifest(manifest, config=config, verification=manifest.get("rag_verification", {}))
    section = manifest.get("generation_fixture", {})
    reconstructed = merge_e6_generation_manifest(manifest, config=config, verification=section)
    if reconstructed["generation_fixture"] != section:
        raise RetrievalEvaluationError("Canonical E6 identity mismatch")


def sealed_identity(manifest: Mapping[str, object], config: AppConfig) -> dict[str, object]:
    """Validate E1/E4/E5/E6/E7, then build the pre-test identity using train only."""
    from app.manifest import current_git_state, runtime_profile_from_config, runtime_profile_sha256
    commit, dirty = current_git_state()
    if not commit or dirty or manifest.get("git_commit") != commit or manifest.get("git_dirty") is not False:
        raise RetrievalEvaluationError("Freeze requires manifest source == clean current HEAD")
    verify_canonical_generation(manifest, config)
    section = manifest.get("retrieval_benchmark", {})
    verified = verify_dev_artifact(manifest, config=config, verification={"metrics": section.get("metrics_artifact")})
    if verified != section:
        raise RetrievalEvaluationError("Canonical E7 dev section differs from verified artifacts")
    inputs = load_evaluation_data(config, manifest)
    _, index = index_identity(config, manifest, inputs.corpus)
    profile = runtime_profile_from_config(config)
    return {"schema_version": 1, "git_commit": commit, "git_dirty": False,
            "effective_config": effective_config_identity(config), "prompt": dict(prompt_identity()),
            "runtime_profile_sha256": runtime_profile_sha256(profile), "model_runtime": profile,
            "retrieval": retrieval_identity(config), "thresholds": {}, "index": index,
            "dataset": inputs.dataset_identity, "e1_dataset": deepcopy(manifest["dataset"]),
            "e1_data_integrity": deepcopy(manifest["data_integrity"]), "split": deepcopy(manifest["split"]),
            "e5_verification": deepcopy(manifest["rag_verification"]),
            "e6_generation_fixture": deepcopy(manifest["generation_fixture"]),
            "e7_dev_benchmark": deepcopy(section), "metric_policy": metric_policy_identity()}


def freeze_manifest(manifest: Mapping[str, object], *, config: AppConfig) -> dict[str, object]:
    if manifest.get("freeze", {}).get("final_test_frozen") is not False:
        raise RetrievalEvaluationError("Freeze is a one-way explicit transition; baseline already frozen or invalid")
    sealed = sealed_identity(manifest, config)
    from app.manifest import current_git_state
    if current_git_state() != (sealed["git_commit"], False):
        raise RetrievalEvaluationError("Source changed before final freeze")
    merged = deepcopy(dict(manifest))
    merged["freeze"].update(final_test_frozen=True, runtime_profile_locked=True,
        frozen_git_commit=sealed["git_commit"], git_dirty=False,
        frozen_at=datetime.now(timezone.utc).isoformat(),
        identity=sealed, identity_sha256=identity(sealed))
    return merged


def validate_test_gate(manifest: Mapping[str, object], *, config: AppConfig) -> None:
    freeze = manifest.get("freeze", {})
    if not isinstance(freeze, Mapping) or freeze.get("final_test_frozen") is not True:
        raise RetrievalEvaluationError("Official test is gated: explicitly freeze the clean canonical baseline first")
    from app.manifest import current_git_state
    commit, dirty = current_git_state()
    if (not commit or dirty or freeze.get("frozen_git_commit") != commit or freeze.get("git_dirty") is not False
            or not isinstance(freeze.get("frozen_at"), str) or not freeze["frozen_at"]):
        raise RetrievalEvaluationError("Test requires the exact frozen HEAD and clean tree")
    current = sealed_identity(manifest, config)
    if freeze.get("identity") != current or freeze.get("identity_sha256") != identity(current):
        raise RetrievalEvaluationError("Frozen baseline config/prompt/runtime/retrieval/index/dataset identity drift")
    if current_git_state() != (commit, False):
        raise RetrievalEvaluationError("Source changed during pre-test gate")
