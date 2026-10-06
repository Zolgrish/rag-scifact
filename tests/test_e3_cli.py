"""Offline CLI orchestration, source selection and error-path tests."""

from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import io
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from app.config import load_config
from app.data_audit import EXPECTED_FILE_SHA256
from app.embedder import EmbeddingError
from app.indexer import IndexLoadError
from app.loader import CorpusDocument
from scripts import build_index as build_cli
from scripts import retrieve as retrieve_cli
from tests.test_e3_indexer import make_bundle
from tests.test_e3_retriever import QueryEmbedder


class RetrieveCLITests(unittest.TestCase):
    def run_cli(self, argv, config=None):
        config = config or load_config()
        bundle = make_bundle()
        embedder = QueryEmbedder()
        out, err = io.StringIO(), io.StringIO()
        with patch.object(retrieve_cli, "bootstrap", return_value=(config, Mock(), "cli-test")), \
             patch.object(retrieve_cli, "load_bundle", return_value=bundle) as load, \
             patch.object(retrieve_cli, "MiniLMEmbedder", return_value=embedder), \
             patch("app.loader.load_qrels", side_effect=AssertionError("qrels read")), \
             patch("app.loader.load_corpus", side_effect=AssertionError("corpus read")), \
             redirect_stdout(out), redirect_stderr(err):
            result = retrieve_cli.main(argv)
        return result, json.loads(out.getvalue()) if out.getvalue() else None, embedder, load

    def test_arbitrary_query_and_query_id_use_exact_source(self):
        code, output, embedder, load = self.run_cli(["--query", "  original scientific claim  ", "--top-k", "2"])
        self.assertEqual(code, 0)
        self.assertEqual(output["query"], "original scientific claim")
        self.assertEqual(embedder.calls, ["original scientific claim"])
        self.assertIsNone(output["query_id"])
        self.assertEqual(load.call_args.kwargs["expected_corpus_sha256"], EXPECTED_FILE_SHA256["corpus"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queries.jsonl"
            path.write_text(json.dumps({"_id": "arbitrary-id", "text": "Exact English source text!",
                                       "metadata": {"gold-doc": ["DO NOT PASS THIS"]}}) + "\n", encoding="utf-8")
            config = load_config()
            config = replace(config, paths=replace(config.paths, queries=path))
            code, output, embedder, _ = self.run_cli(["--query-id", "arbitrary-id"], config)
        self.assertEqual(code, 0)
        self.assertEqual(output["query_id"], "arbitrary-id")
        self.assertEqual(embedder.calls, ["Exact English source text!"])
        self.assertNotIn("DO NOT PASS", repr(output))

    def test_missing_id_and_invalid_input_have_validation_exit(self):
        with patch.object(retrieve_cli, "load_queries", return_value={}):
            code, _, embedder, load = self.run_cli(["--query-id", "missing"])
        self.assertEqual(code, 2)
        self.assertEqual(embedder.calls, [])
        load.assert_not_called()
        for args in (["--query", " "], ["--query", "text", "--top-k", "11"]):
            code, _, _, load = self.run_cli(args)
            self.assertEqual(code, 2)
            load.assert_not_called()


    def test_config_drives_default_top_k_and_cli_override(self):
        config = load_config()
        config = replace(
            config,
            retrieval=replace(
                config.retrieval,
                mode="dense",
                top_k=1,
                max_top_k=3,
            ),
        )

        code, output, embedder, _ = self.run_cli(["--query", "claim"], config)
        self.assertEqual(code, 0)
        self.assertEqual(output["top_k"], 1)
        self.assertEqual(embedder.calls, ["claim"])

        code, output, embedder, _ = self.run_cli(
            ["--query", "claim", "--top-k", "2"],
            config,
        )
        self.assertEqual(code, 0)
        self.assertEqual(output["top_k"], 2)
        self.assertEqual(embedder.calls, ["claim"])

    def test_configured_max_and_mode_are_enforced_before_index_loading(self):
        config = load_config()
        limited = replace(
            config,
            retrieval=replace(
                config.retrieval,
                mode="dense",
                top_k=2,
                max_top_k=3,
            ),
        )
        code, _, embedder, load = self.run_cli(
            ["--query", "claim", "--top-k", "4"],
            limited,
        )
        self.assertEqual(code, 2)
        self.assertEqual(embedder.calls, [])
        load.assert_not_called()

        unsupported = replace(
            config,
            retrieval=replace(config.retrieval, mode="bm25"),
        )
        code, _, embedder, load = self.run_cli(["--query", "claim"], unsupported)
        self.assertEqual(code, 1)
        self.assertEqual(embedder.calls, [])
        load.assert_not_called()

    def test_index_and_runtime_failures_are_nonzero_and_never_empty_results(self):
        for target, error in (("load_bundle", IndexLoadError("corrupt bundle")),
                              ("MiniLMEmbedder", EmbeddingError("runtime down"))):
            out, err = io.StringIO(), io.StringIO()
            with patch.object(retrieve_cli, "bootstrap", return_value=(load_config(), Mock(), "failed")), \
                 patch.object(retrieve_cli, "load_bundle", return_value=make_bundle()), \
                 patch.object(retrieve_cli, target, side_effect=error), \
                 redirect_stdout(out), redirect_stderr(err):
                code = retrieve_cli.main(["--query", "text"])
            self.assertEqual(code, 1)
            self.assertEqual(out.getvalue(), "")
            self.assertEqual(json.loads(err.getvalue())["error"], type(error).__name__)

    def test_query_source_options_are_mutually_exclusive(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            retrieve_cli.build_parser().parse_args(["--query", "text", "--query-id", "0"])


class BuildCLITests(unittest.TestCase):
    def test_corpus_hash_checked_before_model_loading(self):
        with patch.object(build_cli, "bootstrap", return_value=(load_config(), Mock(), "bad")), \
             patch.object(build_cli, "file_sha256", return_value="b" * 64), \
             patch.object(build_cli, "MiniLMEmbedder") as runtime, \
             redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            self.assertEqual(build_cli.main([]), 1)
        runtime.assert_not_called()


    def test_build_rejects_unsupported_retrieval_mode_before_model_loading(self):
        config = load_config()
        config = replace(
            config,
            retrieval=replace(config.retrieval, mode="bm25"),
        )
        with patch.object(build_cli, "bootstrap", return_value=(config, Mock(), "bad-mode")), \
             patch.object(build_cli, "MiniLMEmbedder") as runtime, \
             redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            self.assertEqual(build_cli.main([]), 1)
        runtime.assert_not_called()

    def test_successful_build_report_and_manifest_preservation(self):
        from app.embedder import MiniLMEmbedder
        from tests.test_e2_embedder import FakeBackend

        with tempfile.TemporaryDirectory() as tmp:
            config = load_config()
            config = replace(config, paths=replace(config.paths, index_dir=Path(tmp) / "indexes", artifact_dir=Path(tmp) / "artifacts"))
            config.paths.artifact_dir.mkdir()
            preserved = config.paths.artifact_dir / "manifest.json"
            preserved.write_text('{"dataset":"preserve","freeze":{"final_test_frozen":true}}', encoding="utf-8")
            before = preserved.read_bytes()
            logger = logging.getLogger("test-e3-build")
            out = io.StringIO()
            with patch.object(build_cli, "bootstrap", return_value=(config, logger, "build-test")), \
                 patch.object(build_cli, "file_sha256", return_value=EXPECTED_FILE_SHA256["corpus"]), \
                 patch.object(build_cli, "EXPECTED_CORPUS_COUNT", 1), \
                 patch.object(build_cli, "load_corpus", return_value={"doc": CorpusDocument("doc", "Title", "Body text")}), \
                 patch.object(build_cli, "MiniLMEmbedder", return_value=MiniLMEmbedder(backend=FakeBackend())), \
                 patch("app.loader.load_qrels", side_effect=AssertionError("qrels read")), \
                 patch("app.loader.load_queries", side_effect=AssertionError("queries read")), \
                 redirect_stdout(out):
                code = build_cli.main(["--batch-size", "2", "--local-files-only"])
            self.assertEqual(code, 0)
            report = json.loads(out.getvalue())
            self.assertEqual(report["chunks"], report["ntotal"])
            self.assertTrue(report["reload_vectors_and_mapping_equal"])
            self.assertTrue((config.paths.artifact_dir / "build-test" / "index_build.json").is_file())
            self.assertEqual(preserved.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
