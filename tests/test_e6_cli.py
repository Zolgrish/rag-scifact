"""Fixture CLI setup, no implicit build, no-update behavior and failures."""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from app.config import REPO_ROOT, load_config
from app.evaluator import GENERATION_CASES
from app.generator import RuntimeMetadata
from app.indexer import IndexLoadError
from app.rag import build_rag_pipeline_from_bundle
from app.manifest import ManifestConflictError
from scripts import evaluate_generation as cli
from tests.test_e5_manifest import locked_manifest
from tests.test_e6_grader import execution


class CLITests(unittest.TestCase):
    def run_cli(self, argv=(), *, failed=False, missing=False):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "artifacts") as tmp:
            root = Path(tmp)
            config = load_config()
            config = replace(config, paths=replace(config.paths, artifact_dir=root))
            manifest = root / "manifest.json"
            manifest.write_bytes(b'{"preserve":"unchanged"}\n')
            before = manifest.read_bytes()
            pipeline = Mock()
            outputs = [execution(c) for c in GENERATION_CASES]
            if failed: outputs[0] = RuntimeError("case unavailable")
            pipeline.ask.side_effect = outputs
            pipeline.generator.inspect_runtime_metadata.return_value = RuntimeMetadata(
                config.llm.runtime, config.llm.runtime_version, config.llm.model_id,
                config.llm.runtime_model_digest, config.llm.quantization, config.llm.context_length)
            out, err = io.StringIO(), io.StringIO()
            with patch.object(cli, "bootstrap", return_value=(config, Mock(), "run")), \
                 patch.object(cli, "current_git_state", return_value=("source", True)), \
                 patch.object(cli, "bundle_hashes", return_value={"index.faiss": "unchanged"}), \
                 patch.object(cli, "_record", return_value={"artifact": "path", "sha256": "a" * 64}), \
                 patch.object(cli, "load_bundle", side_effect=IndexLoadError("missing fixture bundle") if missing else None, return_value=Mock()), \
                 patch.object(cli, "validate_atlas_bundle"), \
                 patch.object(cli, "build_atlas_bundle") as build, \
                 patch.object(cli, "build_rag_pipeline_from_bundle", return_value=pipeline) as factory, \
                 patch.object(cli, "merge_e6_generation_manifest", return_value={}) as merge, \
                 redirect_stdout(out), redirect_stderr(err):
                code = cli.main(list(argv))
                self.assertEqual(manifest.read_bytes(), before)
                artifacts = list((root / "run").glob("*")) if (root / "run").exists() else []
                rows = [json.loads(line) for line in (root / "run/generation_run.jsonl").read_text().splitlines()] if artifacts else []
            return code, out.getvalue(), err.getvalue(), build, factory, merge, rows

    def test_normal_run_no_build_or_manifest_update(self):
        code, out, err, build, factory, merge, rows = self.run_cli()
        self.assertEqual(code, 0)
        self.assertEqual(len(rows), 6)
        build.assert_not_called()
        merge.assert_not_called()
        self.assertFalse(json.loads(out)["manifest_updated"])
        self.assertEqual(json.loads(out)["source_git_commit"], "source")

    def test_missing_bundle_fails_without_implicit_build_or_generator(self):
        code, _, err, build, factory, _, rows = self.run_cli(missing=True)
        self.assertEqual(code, 1)
        self.assertIn("IndexLoadError", err)
        self.assertEqual(rows, [])
        build.assert_not_called()
        factory.assert_not_called()

    def test_build_explicit_and_failed_suite_cannot_update(self):
        code, _, _, build, _, merge, rows = self.run_cli(["--build-index", "--local-files-only", "--update-manifest"], failed=True)
        self.assertEqual(code, 2)
        self.assertEqual(len(rows), 6)
        self.assertTrue(build.call_args.kwargs["local_files_only"])
        merge.assert_not_called()

    def test_bundle_factory_preserves_lock_order_and_requested_identity(self):
        config = load_config()
        manifest = locked_manifest(config)
        changed = replace(config, llm=replace(config.llm, model_id="different"))
        with patch("app.rag.load_manifest", return_value=manifest), patch("app.rag.build_generator") as generator, patch("app.rag.load_bundle") as load:
            with self.assertRaises(ManifestConflictError):
                build_rag_pipeline_from_bundle(changed, bundle_path="atlas", expected_corpus_sha256="b" * 64)
        load.assert_not_called(); generator.assert_not_called()
        with patch("app.rag.load_manifest", return_value=manifest), patch("app.rag.build_generator"), \
             patch("app.rag.load_bundle", return_value=Mock()) as load, patch("app.rag.MiniLMEmbedder"), patch("app.rag.DenseRetriever"):
            build_rag_pipeline_from_bundle(config, bundle_path="atlas", expected_corpus_sha256="b" * 64)
        load.assert_called_once_with("atlas", expected_corpus_sha256="b" * 64)


if __name__ == "__main__": unittest.main()
