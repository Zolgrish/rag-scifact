"""Document metrics and auditable retrieval runs; qrels stay evaluator-side."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time
from typing import Callable, Mapping, Protocol, Sequence

from app.loader import BenchmarkQuery
from app.models import RetrievedDocument

EVALUATION_DEPTH = 10
METRIC_NAMES = ("recall@5", "recall@10", "mrr@10", "ndcg@10")
METRIC_POLICY_VERSION = "scifact-document-metrics-v1"


class RetrievalEvaluationError(RuntimeError):
    """Setup, result or artifact contract violation (never a semantic status)."""


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]: ...


@dataclass(frozen=True)
class QueryOutcome:
    query_id: str
    results: tuple[RetrievedDocument, ...]
    retrieval_ms: float
    error: Mapping[str, str] | None = None


def finite_number(value: object, *, nonnegative: bool = False) -> bool:
    return (type(value) in (int, float) and math.isfinite(value)
            and (not nonnegative or value >= 0))


def validate_results(results: Sequence[RetrievedDocument]) -> None:
    """Validate production dedup/order; never silently dedup or rerank here."""
    if len(results) > EVALUATION_DEPTH:
        raise RetrievalEvaluationError("Results exceed evaluation depth 10")
    docs: set[str] = set()
    previous = math.inf
    for rank, result in enumerate(results, 1):
        if (not isinstance(result, RetrievedDocument) or type(result.rank) is not int
                or result.rank != rank or not isinstance(result.doc_id, str) or not result.doc_id
                or not isinstance(result.chunk_id, str) or not result.chunk_id
                or result.doc_id in docs or not finite_number(result.score)
                or result.score > previous):
            raise RetrievalEvaluationError("Results require unique documents, continuous ranks and descending finite scores")
        docs.add(result.doc_id)
        previous = result.score


def score_query(results: Sequence[RetrievedDocument], gold: Mapping[str, float]) -> dict[str, float]:
    validate_results(results)
    if any(not isinstance(doc, str) or not doc or not finite_number(rel, nonnegative=True)
           for doc, rel in gold.items()):
        raise RetrievalEvaluationError("Qrel grades must be finite non-negative numbers")
    positives = {doc for doc, rel in gold.items() if rel > 0}
    if not positives:
        return dict.fromkeys(METRIC_NAMES, 0.0)
    recalls = {f"recall@{k}": sum(r.doc_id in positives for r in results[:k]) / len(positives)
               for k in (5, 10)}
    first = next((r.rank for r in results if r.doc_id in positives), None)
    # Scaling cancels in DCG/IDCG and avoids overflow for arbitrary finite grades.
    maximum = max(gold.values())
    def gain(rel: float) -> float:
        if maximum < 500:
            return math.expm1(rel * math.log(2))
        return math.exp2(rel - maximum) - math.exp2(-maximum)
    dcg = math.fsum(gain(gold.get(r.doc_id, 0.0)) / math.log2(r.rank + 1) for r in results)
    ideal = sorted(gold.values(), reverse=True)[:10]
    idcg = math.fsum(gain(rel) / math.log2(rank + 1) for rank, rel in enumerate(ideal, 1))
    return {**recalls, "mrr@10": 1.0 / first if first else 0.0,
            "ndcg@10": min(1.0, dcg / idcg)}


def evaluate_queries(retriever: Retriever, queries: Sequence[BenchmarkQuery], *,
                     clock: Callable[[], float] = time.perf_counter) -> tuple[QueryOutcome, ...]:
    """Exactly one production Top10 retrieval per query, including failed calls."""
    if not queries or len({q.query_id for q in queries}) != len(queries):
        raise RetrievalEvaluationError("Evaluation requires non-empty unique query IDs")
    outcomes = []
    for query in queries:
        started = clock()
        error = None
        results: tuple[RetrievedDocument, ...] = ()
        try:
            results = tuple(retriever.retrieve(query.text, EVALUATION_DEPTH))
        except RetrievalEvaluationError:
            raise
        except Exception as exc:
            results = ()
            error = {"type": type(exc).__name__, "message": str(exc)}
        if error is None:
            # A malformed production ranking is an evaluator/retriever contract defect,
            # not a legitimate zero-scoring query failure. Fail the whole run closed.
            validate_results(results)
        elapsed = (clock() - started) * 1000
        if not finite_number(elapsed, nonnegative=True):
            raise RetrievalEvaluationError("Invalid wall-clock retrieval duration")
        outcomes.append(QueryOutcome(query.query_id, results, elapsed, error))
    return tuple(outcomes)


def aggregate(outcomes: Sequence[QueryOutcome], gold: Mapping[str, Mapping[str, float]]) -> dict[str, object]:
    if not outcomes:
        raise RetrievalEvaluationError("Cannot average an empty evaluation denominator")
    scores = [dict.fromkeys(METRIC_NAMES, 0.0) if o.error is not None
              else score_query(o.results, gold[o.query_id]) for o in outcomes]
    timings = [o.retrieval_ms for o in outcomes]
    if any(not finite_number(t, nonnegative=True) for t in timings):
        raise RetrievalEvaluationError("Invalid retrieval timing")
    return {"sample_size": len(outcomes), "denominator": len(outcomes),
            "failure_count": sum(o.error is not None for o in outcomes),
            "metrics": {key: math.fsum(s[key] for s in scores) / len(outcomes) for key in METRIC_NAMES},
            "timing_ms": {"retrieval_p50": statistics.median(timings), "samples": len(timings),
                          "policy": "all_attempted_queries_including_errors; excludes_setup"}}


def failure_assessment(category: str, outcome: QueryOutcome, scores: Mapping[str, float],
                       gold: Mapping[str, float]) -> dict[str, str]:
    """Describe only the failure mode proven by the run and a dev-only next action."""
    positives = [doc for doc, relevance in gold.items() if relevance > 0]
    retrieved_positive = [result for result in outcome.results if result.doc_id in positives]
    scope = (
        "Assessment is limited to the observed retrieval outcome; it does not prove a deeper "
        "embedding, chunking, lexical, or model cause."
    )
    if category == "retrieval_error":
        error_type = outcome.error["type"] if outcome.error else "unknown"
        error_message = outcome.error["message"] if outcome.error else ""
        return {
            "observed_output": f"The retrieval call failed with {error_type}: {error_message}",
            "root_cause_assessment": (
                f"The recorded operational failure ({error_type}) prevented this query from producing a ranking. "
                + scope
            ),
            "concrete_possible_improvement": (
                "Reproduce this query on the dev split, diagnose the recorded exception at the retriever/embedding "
                "boundary, add a regression for the confirmed defect, and rerun the full dev benchmark before freeze."
            ),
        }
    if category == "retrieval_miss":
        return {
            "observed_output": (
                f"None of the {len(positives)} positively judged document(s) appeared in the production Top-10; "
                "Recall@10=0."
            ),
            "root_cause_assessment": (
                "The immediate measured cause is a Top-10 dense-retrieval miss: no judged relevant document reached "
                "the evaluation cutoff. " + scope
            ),
            "concrete_possible_improvement": (
                "On dev only, inspect the judged document's best chunk rank and score beyond rank 10 and compare its "
                "query/chunk representation; use that evidence to decide whether a ranking/representation or chunking "
                "change is warranted before changing the frozen baseline."
            ),
        }
    if category == "partial_recall":
        return {
            "observed_output": (
                f"Top-10 retrieved {len(retrieved_positive)} of {len(positives)} positively judged document(s); "
                f"Recall@10={scores['recall@10']:.12g}."
            ),
            "root_cause_assessment": (
                "The immediate measured cause is incomplete Top-10 coverage: at least one judged relevant document "
                "was ranked outside the evaluation cutoff. " + scope
            ),
            "concrete_possible_improvement": (
                "On dev only, inspect the omitted judged documents' best chunk ranks/scores and score margins against "
                "the retrieved positives; evaluate a ranking or representation improvement only if that inspection "
                "supports it."
            ),
        }
    if category == "ranking_failure":
        first_rank = min((result.rank for result in retrieved_positive), default=0)
        return {
            "observed_output": (
                f"All positively judged documents were present by rank 10, but ranking quality was non-ideal "
                f"(first positive rank={first_rank}, MRR@10={scores['mrr@10']:.12g}, "
                f"nDCG@10={scores['ndcg@10']:.12g})."
            ),
            "root_cause_assessment": (
                "The immediate measured cause is ranking order: relevant evidence was retrieved but placed below a "
                "better rank and/or graded-relevance order. " + scope
            ),
            "concrete_possible_improvement": (
                "On dev only, inspect score margins and the higher-ranked non-relevant chunks for this query; use the "
                "observed ranking evidence to evaluate a ranking/representation improvement before freeze."
            ),
        }
    if category == "zero_positive_qrels":
        return {
            "observed_output": "The evaluator has no positive judgment for this query, so all four metrics are zero by policy.",
            "root_cause_assessment": (
                "This is a judgment-coverage condition rather than a demonstrated retrieval miss. " + scope
            ),
            "concrete_possible_improvement": (
                "Audit the evaluator judgments/data contract before interpreting this query as a retrieval failure; do "
                "not tune retrieval from an unjudged outcome."
            ),
        }
    raise RetrievalEvaluationError(f"Unsupported failure-analysis category: {category}")


def failure_candidates(outcomes: Sequence[QueryOutcome], queries: Sequence[BenchmarkQuery],
                       gold: Mapping[str, Mapping[str, float]]) -> list[dict[str, object]]:
    questions = {q.query_id: q.text for q in queries}
    cases = []
    for outcome in outcomes:
        scores = (dict.fromkeys(METRIC_NAMES, 0.0) if outcome.error is not None
                  else score_query(outcome.results, gold[outcome.query_id]))
        if outcome.error is not None:
            category = "retrieval_error"
        elif scores["recall@10"] == 0:
            category = "retrieval_miss" if any(v > 0 for v in gold[outcome.query_id].values()) else "zero_positive_qrels"
        elif scores["recall@10"] < 1:
            category = "partial_recall"
        elif scores["mrr@10"] < 1 or scores["ndcg@10"] < 1:
            category = "ranking_failure"
        else:
            continue
        cases.append({"schema_version": 1, "query_id": outcome.query_id,
                      "question": questions[outcome.query_id], "category": category,
                      "expected_positive_docs": [{"doc_id": doc, "relevance": rel}
                        for doc, rel in sorted(gold[outcome.query_id].items()) if rel > 0],
                      "retrieved": [{"doc_id": r.doc_id, "chunk_id": r.chunk_id,
                                     "rank": r.rank, "score": r.score} for r in outcome.results],
                      "metrics": scores, "error": outcome.error,
                      **failure_assessment(category, outcome, scores, gold[outcome.query_id])})
    cases.sort(key=lambda c: (c["metrics"]["recall@10"], c["metrics"]["mrr@10"], int(c["query_id"])))
    return cases


def metric_policy_identity() -> dict[str, object]:
    policy = {"version": METRIC_POLICY_VERSION, "evaluation_depth": EVALUATION_DEPTH,
              "metrics": METRIC_NAMES, "zero_positive_qrels": "all_four_zero_keep_denominator",
              "per_query_error": "all_four_zero_keep_denominator",
              "source": "\n".join(inspect.getsource(f) for f in
                  (finite_number, validate_results, score_query, aggregate,
                   failure_assessment, failure_candidates))}
    return {"version": METRIC_POLICY_VERSION, "sha256": hashlib.sha256(
        json.dumps(policy, sort_keys=True, allow_nan=False).encode()).hexdigest()}


def retrieval_rows(outcomes: Sequence[QueryOutcome], run_id: str) -> list[dict[str, object]]:
    rows = []
    for outcome in outcomes:
        common = {"schema_version": 1, "run_id": run_id, "query_id": outcome.query_id}
        rows.append({**common, "record_type": "query", "status": "ERROR" if outcome.error else "OK",
                     "result_count": len(outcome.results), "retrieval_ms": outcome.retrieval_ms,
                     "error": outcome.error})
        rows.extend({**common, "record_type": "result", "doc_id": r.doc_id,
                     "chunk_id": r.chunk_id, "rank": r.rank, "score": r.score} for r in outcome.results)
    return rows


def strict_json(data: str | bytes) -> object:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("Duplicate JSON key")
            value[key] = item
        return value
    def constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Non-finite JSON number")
        return result
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant, parse_float=number)


def parse_retrieval_rows(rows: Sequence[object], query_ids: Sequence[str], run_id: str) -> tuple[QueryOutcome, ...]:
    """Fail closed on missing/extra queries, reordered rows or fabricated failures."""
    outcomes = []
    cursor = 0
    common_keys = {"schema_version", "run_id", "query_id", "record_type"}
    for qid in query_ids:
        if cursor >= len(rows):
            raise RetrievalEvaluationError("Missing query status row")
        row = rows[cursor]
        cursor += 1
        expected = common_keys | {"status", "result_count", "retrieval_ms", "error"}
        if (not isinstance(row, dict) or set(row) != expected or row["record_type"] != "query"
                or type(row["schema_version"]) is not int or row["schema_version"] != 1
                or row["query_id"] != qid or row["run_id"] != run_id
                or type(row["result_count"]) is not int or not 0 <= row["result_count"] <= 10
                or not finite_number(row["retrieval_ms"], nonnegative=True)):
            raise RetrievalEvaluationError("Invalid query status schema/order/identity")
        error = row["error"]
        if row["status"] == "ERROR":
            if (row["result_count"] != 0 or not isinstance(error, dict) or set(error) != {"type", "message"}
                    or not isinstance(error["type"], str) or not error["type"] or not isinstance(error["message"], str)):
                raise RetrievalEvaluationError("Invalid explicit query failure")
        elif row["status"] != "OK" or error is not None:
            raise RetrievalEvaluationError("Invalid query status/error combination")
        results = []
        for _ in range(row["result_count"]):
            if cursor >= len(rows):
                raise RetrievalEvaluationError("Missing result row")
            result = rows[cursor]
            cursor += 1
            if (not isinstance(result, dict) or set(result) != common_keys | {"doc_id", "chunk_id", "rank", "score"}
                    or type(result["schema_version"]) is not int or result["schema_version"] != 1
                    or result["record_type"] != "result" or result["query_id"] != qid or result["run_id"] != run_id):
                raise RetrievalEvaluationError("Invalid result schema/order/identity")
            results.append(RetrievedDocument(result["doc_id"], result["chunk_id"], result["rank"], result["score"]))
        validate_results(results)
        outcomes.append(QueryOutcome(qid, tuple(results), row["retrieval_ms"], error))
    if cursor != len(rows):
        raise RetrievalEvaluationError("Unexpected extra retrieval rows")
    return tuple(outcomes)


def publish_run(directory: Path, rows: Sequence[Mapping[str, object]], candidates: Sequence[Mapping[str, object]],
                summary: Mapping[str, object]) -> dict[str, object]:
    """Reserve a new immutable run; atomically publish each file, metrics last."""
    encoded = {"retrieval_run.jsonl": "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows),
               "failure_analysis.jsonl": "".join(json.dumps(c, ensure_ascii=False, allow_nan=False) + "\n" for c in candidates)}
    # Validate summary before reserving/publishing anything.
    json.dumps(summary, allow_nan=False)
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    from app.data_audit import file_sha256
    from app.config import REPO_ROOT
    completed = dict(summary)
    def record(name: str) -> dict[str, str]:
        path = directory / name
        return {"artifact": path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(), "sha256": file_sha256(path)}
    for name in ("retrieval_run.jsonl", "failure_analysis.jsonl", "metrics.json"):
        if name == "metrics.json":
            completed.update(retrieval_run=record("retrieval_run.jsonl"), failure_analysis=record("failure_analysis.jsonl"))
            text = json.dumps(completed, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        else:
            text = encoded[name]
        fd, temporary = tempfile.mkstemp(dir=directory, prefix=".e7-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, directory / name)
        finally:
            if Path(temporary).exists():
                Path(temporary).unlink()
    return completed
