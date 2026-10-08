"""Dev setup failures abort, test gate precedes data, normal runs preserve manifest."""
from contextlib import nullcontext, redirect_stdout, redirect_stderr
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from app.benchmark_manifest import EvaluationData, load_evaluation_data
from app.config import REPO_ROOT, load_config
from app.loader import BenchmarkQuery, load_qrels
from app.manifest import load_manifest
from app.retrieval_evaluator import RetrievalEvaluationError
from scripts import evaluate as cli
from scripts import freeze_final_test as freeze_cli
from tests.test_e7_metrics import results
from tests.test_e5_manifest import locked_manifest


class CLITests(unittest.TestCase):
    def run_cli(self, *, setup_error=False, query_error=False, argv=(), gate_side_effect=None,
                git_state=None, git_states=None):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "artifacts") as tmp:
            root = Path(tmp)
            config = load_config()
            config = replace(config, paths=replace(config.paths, artifact_dir=root))
            path = root / "manifest.json"
            path.write_text('{"unchanged":true}\n', encoding="utf-8")
            original = path.read_bytes()
            manifest = locked_manifest(config)
            queries = tuple(BenchmarkQuery(str(i), "Safe question " + str(i)) for i in range(100))
            inputs = EvaluationData(queries, {q.query_id: {"a": 1.} for q in queries}, {"hashes": {}}, {}, {})
            retriever = Mock()
            retriever.retrieve.side_effect = ([RuntimeError("query failed")] if query_error else [results(["a"])]) + [results(["a"])] * 99
            out, err = io.StringIO(), io.StringIO()
            gate = (patch.object(cli, "validate_final_test_gate", side_effect=gate_side_effect)
                    if gate_side_effect is not None else nullcontext())
            git = (patch.object(cli, "current_git_state", side_effect=git_states)
                   if git_states is not None
                   else patch.object(cli, "current_git_state",
                                     return_value=git_state or (manifest["git_commit"], True)))
            with patch.object(cli, "bootstrap", return_value=(config, Mock(), "run")), \
                 patch.object(cli, "load_manifest", return_value=deepcopy(manifest)), \
                 git, \
                 patch.object(cli, "load_evaluation_data", return_value=inputs) as data, \
                 patch.object(cli, "index_identity", return_value=(Mock(), {"files": {}}), side_effect=RuntimeError("corrupt index") if setup_error else None), \
                 patch.object(cli, "MiniLMEmbedder") as embedder, \
                 patch.object(cli, "DenseRetriever", return_value=retriever), \
                 patch.object(cli, "runtime_metadata", return_value={"device": "offline-test"}), \
                 patch.object(cli, "canonical_generation_timing", return_value={"scope": "canonical-E6"}), \
                 patch.object(cli, "write_manifest_atomic") as write, \
                 gate, \
                 redirect_stdout(out), redirect_stderr(err):
                code = cli.main(list(argv))
            self.assertEqual(path.read_bytes(), original)
            write.assert_not_called()
            rows = [json.loads(line) for line in (root / "run/retrieval_run.jsonl").read_text().splitlines()] if (root / "run/retrieval_run.jsonl").exists() else []
            return code, out.getvalue(), err.getvalue(), rows, data, retriever, embedder

    def test_dev_100_one_top10_per_query_no_update_no_llm(self):
        code, out, _, rows, _, retriever, embedder = self.run_cli()
        self.assertEqual(code, 0)
        summary = json.loads(out)
        self.assertEqual(summary["denominator"], 100)
        self.assertFalse(summary["manifest_updated"])
        self.assertEqual(retriever.retrieve.call_count, 100)
        self.assertTrue(all(call.args[1] == 10 and len(call.args) == 2 for call in retriever.retrieve.call_args_list))
        self.assertEqual([r["query_id"] for r in rows if r["record_type"] == "query"], [str(i) for i in range(100)])
        self.assertTrue(embedder.call_args.kwargs["local_files_only"])

    def test_setup_failure_aborts_before_scored_queries_or_artifacts(self):
        code, _, err, rows, _, retriever, embedder = self.run_cli(setup_error=True)
        self.assertEqual(code, 1)
        self.assertIn("corrupt index", err)
        self.assertEqual(rows, [])
        retriever.retrieve.assert_not_called()
        embedder.assert_not_called()

    def test_per_query_failure_zero_contribution_not_removed(self):
        code, out, _, rows, _, retriever, _ = self.run_cli(query_error=True)
        self.assertEqual(code, 2)
        summary = json.loads(out)
        self.assertEqual(summary["denominator"], 100)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(set(summary["metrics"].values()), {.99})
        self.assertEqual(rows[0]["status"], "ERROR")
        self.assertEqual(retriever.retrieve.call_count, 100)

    def test_mocked_test_request_checks_gate_before_loader_no_auto_freeze(self):
        code, _, err, rows, data, retriever, embedder = self.run_cli(argv=("--split", "test"))
        self.assertEqual(code, 1)
        self.assertIn("gated", err)
        data.assert_not_called()
        retriever.retrieve.assert_not_called()
        embedder.assert_not_called()
        self.assertEqual(rows, [])

    def test_test_run_rechecks_full_gate_after_retrieval_before_publication(self):
        code, _, err, rows, _, retriever, _ = self.run_cli(
            argv=("--split", "test"),
            gate_side_effect=[None, RetrievalEvaluationError("frozen identity drifted during run")],
            git_state=("test-commit", False),
        )
        self.assertEqual(code, 1)
        self.assertIn("frozen identity drifted during run", err)
        self.assertEqual(retriever.retrieve.call_count, 100)
        self.assertEqual(rows, [])

    def test_test_run_rejects_source_change_after_retrieval_before_publication(self):
        commit = "test-commit"
        code, _, err, rows, _, retriever, _ = self.run_cli(
            argv=("--split", "test"),
            gate_side_effect=[None, None],
            git_states=[(commit, False), (commit, True)],
        )
        self.assertEqual(code, 1)
        self.assertIn("Frozen source changed during final-test benchmark", err)
        self.assertEqual(retriever.retrieve.call_count, 100)
        self.assertEqual(rows, [])

    def test_real_dev_loader_reads_only_train_and_canonical_order(self):
        config = load_config()
        from app.data_audit import build_dev_practice_split, EXPECTED_FILE_SHA256
        split = build_dev_practice_split(load_qrels(config.paths.qrels_train))
        manifest = {"dataset": {"corpus_count": 5183, "query_count": 1109,
            "train_query_count": 809, "test_query_count": 300,
            "hashes": {name: {"path": getattr(config.paths, name).relative_to(REPO_ROOT).as_posix(), "sha256": digest}
                       for name, digest in EXPECTED_FILE_SHA256.items()}},
            "split": {"seed": 42, "dev_ids": list(split.dev_ids), "practice_ids": list(split.practice_ids)},
            "data_integrity": {"status": "PASS"}}
        with patch("app.benchmark_manifest.load_qrels", wraps=load_qrels) as reader:
            inputs = load_evaluation_data(config, manifest)
        self.assertEqual([q.query_id for q in inputs.queries], manifest["split"]["dev_ids"])
        self.assertEqual(len(inputs.queries), 100)
        reader.assert_called_once_with(config.paths.qrels_train)
        self.assertTrue(all(set(vars(q)) == {"query_id", "text"} for q in inputs.queries))

    def test_freeze_cli_dirty_rejection_never_writes_or_loads_test(self):
        temporary = tempfile.TemporaryDirectory(dir=REPO_ROOT / "artifacts")
        self.addCleanup(temporary.cleanup)
        config = load_config()
        config = replace(config, paths=replace(config.paths, artifact_dir=Path(temporary.name)))
        manifest = locked_manifest(config)
        (config.paths.artifact_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with patch.object(freeze_cli, "bootstrap", return_value=(config, Mock(), "freeze-run")), \
             patch("app.manifest.current_git_state", return_value=(manifest["git_commit"], True)), \
             patch.object(freeze_cli, "write_manifest_atomic") as write, \
             patch("app.benchmark_manifest.load_qrels") as reader, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(freeze_cli.main([]), 1)
        write.assert_not_called()
        reader.assert_not_called()


if __name__ == "__main__": unittest.main()
