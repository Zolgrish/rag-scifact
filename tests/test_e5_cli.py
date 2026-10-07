"""Thin ask/check CLIs, public response, errors and explicit manifest updates."""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from app.citations import RAGOutputValidationError
from app.config import load_config
from app.generator import GeneratorConnectionError, GeneratorResult, RuntimeMetadata
from app.manifest import E5_CANONICAL_SMOKE_QUERY
from app.models import RAGResponse, RetrievedDocument, SemanticStatus
from app.rag import RAGExecution, RAGTrace
from app.retriever import QueryValidationError
from scripts import ask, check_rag


def execution():
    return RAGExecution(RAGResponse("req", "Insufficient evidence.", SemanticStatus.INSUFFICIENT_EVIDENCE,
                        retrieved=[RetrievedDocument("a", "a:0-1", 1, .5, "private", "source")],
                        timing_ms={"retrieval": 1, "generation": 2, "total": 4}, model_id="real"),
                        RAGTrace("req", ("a:0-1",), ("a:0-1",), (), False, 100, "v1", "sha", "real"))


class CLITests(unittest.TestCase):
    def run_ask(self, argv, config=None, failure=None):
        config = config or load_config()
        pipeline = Mock()
        pipeline.ask.return_value = execution()
        pipeline.ask.side_effect = failure
        out, err = io.StringIO(), io.StringIO()
        with patch.object(ask, "bootstrap", return_value=(config, Mock(), "run")), \
             patch.object(ask, "build_rag_pipeline", return_value=pipeline) as factory, \
             redirect_stdout(out), redirect_stderr(err):
            code = ask.main(argv)
        return code, out.getvalue(), err.getvalue(), pipeline, factory

    def test_config_default_and_override_public_json(self):
        config = load_config()
        config = replace(config, retrieval=replace(config.retrieval, top_k=3))
        for argv, expected in ((["--query", " Q "], 3), (["--query", "Q", "--top-k", "2"], 2)):
            code, out, err, pipeline, _ = self.run_ask(argv, config)
            self.assertEqual(code, 0)
            self.assertEqual(err, "")
            self.assertEqual(pipeline.ask.call_args.args, ("Q", expected))
            self.assertTrue(pipeline.ask.call_args.kwargs["request_id"])
            public = json.loads(out)
            self.assertEqual(set(public["retrieved"][0]), {"doc_id", "rank", "score"})
            self.assertEqual(public["model_id"], "real")

    def test_invalid_input_before_factory_and_distinct_exit_codes(self):
        code, out, err, pipeline, factory = self.run_ask(["--query", " "])
        self.assertEqual(code, 2)
        factory.assert_not_called()
        pipeline.ask.assert_not_called()
        for failure, expected in ((QueryValidationError("invalid"), 2),
                                  (GeneratorConnectionError("down"), 1),
                                  (RAGOutputValidationError("ungrounded"), 3)):
            code, out, err, _, _ = self.run_ask(["--query", "Q"], failure=failure)
            self.assertEqual(code, expected)
            self.assertEqual(out, "")
            self.assertEqual(json.loads(err)["error"], type(failure).__name__)

    def test_ordinary_ask_never_mutates_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_bytes(b'{"preserve":"every byte"}\n')
            before = manifest.read_bytes()
            config = load_config()
            config = replace(config, paths=replace(config.paths, artifact_dir=root))
            self.assertEqual(self.run_ask(["--query", "Q"], config)[0], 0)
            self.assertEqual(before, manifest.read_bytes())

    def test_check_command_updates_only_when_explicit(self):
        config = load_config()
        for update in (False, True):
            pipeline = Mock()
            pipeline.ask.return_value = execution()
            pipeline.generator.count_prompt_tokens.return_value = 100
            pipeline.generator.generate.return_value = GeneratorResult("ok", "real", {"prompt_tokens": 100})
            pipeline.generator.inspect_runtime_metadata.return_value = RuntimeMetadata(
                config.llm.runtime, config.llm.runtime_version, config.llm.model_id,
                config.llm.runtime_model_digest, config.llm.quantization, config.llm.context_length)
            with patch.object(check_rag, "bootstrap", return_value=(config, Mock(), "check-run")), \
                 patch.object(check_rag, "current_git_state", return_value=("source-commit", False)), \
                 patch.object(check_rag, "build_rag_pipeline", return_value=pipeline), \
                 patch.object(check_rag, "write_manifest_atomic") as write, \
                 patch.object(check_rag, "load_manifest", return_value={}), \
                 patch.object(check_rag, "merge_e5_prompt_manifest", return_value={"prompt": "verified"}) as merge, \
                 patch.object(Path, "read_bytes", return_value=b"artifact"), \
                 redirect_stdout(io.StringIO()):
                code = check_rag.main(["--update-manifest"] if update else [])
            self.assertEqual(code, 0)
            self.assertEqual(write.call_count, 2 if update else 1)
            self.assertEqual(merge.call_count, int(update))
            pipeline.ask.assert_called_once_with(E5_CANONICAL_SMOKE_QUERY,
                                                 config.retrieval.top_k,
                                                 request_id="check-run")
            artifact_payload = write.call_args_list[0].args[1]
            self.assertEqual(artifact_payload["source_git_commit"], "source-commit")
            self.assertIs(artifact_payload["source_git_dirty"], False)
            self.assertEqual(artifact_payload["query"], E5_CANONICAL_SMOKE_QUERY)

    def test_check_rejects_unlocked_update_before_composition(self):
        config = load_config()
        config = replace(config, llm=replace(config.llm, runtime_profile_locked=False))
        with patch.object(check_rag, "bootstrap", return_value=(config, Mock(), "run")), \
             patch.object(check_rag, "build_rag_pipeline") as factory, \
             redirect_stderr(io.StringIO()):
            self.assertEqual(check_rag.main(["--update-manifest"]), 1)
        factory.assert_not_called()

    def test_canonical_update_rejects_custom_query_before_composition(self):
        with patch.object(check_rag, "bootstrap",
                          return_value=(load_config(), Mock(), "run")), \
             patch.object(check_rag, "build_rag_pipeline") as factory, \
             redirect_stderr(io.StringIO()):
            self.assertEqual(check_rag.main(["--update-manifest", "--query", "custom"]), 1)
        factory.assert_not_called()

    def test_counter_mismatch_cannot_write_verification_or_manifest(self):
        pipeline = Mock()
        pipeline.ask.return_value = execution()
        pipeline.generator.count_prompt_tokens.return_value = 100
        pipeline.generator.generate.return_value = GeneratorResult("ok", "real", {"prompt_tokens": 99})
        with patch.object(check_rag, "bootstrap", return_value=(load_config(), Mock(), "run")), \
             patch.object(check_rag, "build_rag_pipeline", return_value=pipeline), \
             patch.object(check_rag, "write_manifest_atomic") as write, \
             patch.object(check_rag, "merge_e5_prompt_manifest") as merge, \
             redirect_stderr(io.StringIO()):
            self.assertEqual(check_rag.main(["--update-manifest"]), 1)
        write.assert_not_called()
        merge.assert_not_called()


if __name__ == "__main__":
    unittest.main()
