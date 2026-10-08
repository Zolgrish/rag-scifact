"""Synthetic 100-query E7 artifacts with the existing canonical E5/E6 provenance."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from app.benchmark_manifest import (EvaluationData, artifact_record, canonical_generation_timing,
    identity, retrieval_identity)
from app.config import REPO_ROOT, effective_config_identity, load_config
from app.loader import BenchmarkQuery
from app.manifest import load_manifest
from app.retrieval_evaluator import (QueryOutcome, aggregate, failure_candidates,
    metric_policy_identity, publish_run, retrieval_rows)
from tests.test_e7_metrics import results


class ArtifactFixture:
    def __init__(self, test):
        temporary = tempfile.TemporaryDirectory(dir=REPO_ROOT / "artifacts")
        test.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # Build offline E5/E6 evidence rather than depending on a generated,
        # gitignored canonical manifest or a live Ollama verification.
        from tests.test_e6_manifest import ManifestTests as E6ArtifactFactory
        self.generation = E6ArtifactFactory()
        self.generation.setUp()
        test.addCleanup(self.generation.doCleanups)
        self.config = self.generation.config
        self.manifest = self.generation.merge()
        self.manifest["data_integrity"] = {"status": "PASS", "synthetic": "test-only"}
        self.commit = self.manifest["git_commit"]
        self.queries = tuple(BenchmarkQuery(str(i), "Synthetic question " + str(i)) for i in range(100))
        self.gold = {q.query_id: {"a": 2.5, "b": .5} for q in self.queries}
        self.inputs = EvaluationData(self.queries, self.gold, {"synthetic": "test-only"},
                                    {"name": "dev", "sha256": identity([q.query_id for q in self.queries])}, {})
        self.index = {"synthetic": "test-only"}
        self.bundle = SimpleNamespace(chunks=(SimpleNamespace(doc_id="a", chunk_id="a:0-1"),
                                              SimpleNamespace(doc_id="b", chunk_id="b:0-1")),
                                      manifest={"embedding": {"model_id": self.config.embedding.model_id,
                                          "revision": "offline-revision", "output_dimension": 384,
                                          "dtype": "float32", "normalized": True, "effective_model_input_limit": 256}})
        self.outcomes = tuple(QueryOutcome(q.query_id, tuple(results(["a", "b"])), 1.) for q in self.queries)
        self.summary = {"schema_version": 1, "status": "COMPLETED", "split": "dev", "run_id": "run",
            "source_git_commit": self.commit, "source_git_dirty": False,
            "config": artifact_record("config.yaml"), "effective_config": effective_config_identity(self.config),
            "retrieval": retrieval_identity(self.config), "split_identity": self.inputs.split_identity,
            "dataset": self.inputs.dataset_identity, "index": self.index, "metric_policy": metric_policy_identity(),
            "generation_timing": canonical_generation_timing(self.manifest, self.config),
            "runtime": {"embedding": {**self.bundle.manifest["embedding"], "device": "offline-test"},
                        "python": "3.11.9", "platform": "offline-test", "faiss_threads": 1,
                        "versions": {name: "offline-test" for name in ("numpy", "faiss-cpu", "sentence-transformers", "torch")}},
            "failure_candidate_count": 0,
            **aggregate(self.outcomes, self.gold)}
        self.publish()

    def publish(self):
        self.directory = self.root / self.summary["run_id"]
        self.completed = publish_run(self.directory, retrieval_rows(self.outcomes, self.summary["run_id"]),
            failure_candidates(self.outcomes, self.queries, self.gold), self.summary)
        self.verification = {"metrics": artifact_record(self.directory / "metrics.json")}

    def patches(self, *, states=None):
        stack = ExitStack()
        stack.enter_context(patch("app.manifest._git_state", return_value=(self.commit, False)))
        stack.enter_context(patch("app.manifest.current_git_state", return_value=(self.commit, False), side_effect=states))
        stack.enter_context(patch("app.benchmark_manifest.load_evaluation_data", return_value=self.inputs))
        stack.enter_context(patch("app.benchmark_manifest.index_identity", return_value=(self.bundle, self.index)))
        stack.enter_context(patch("app.indexer.load_bundle", return_value=self.generation.bundle_object))
        stack.enter_context(patch("app.fixtures.validate_atlas_bundle"))
        return stack
