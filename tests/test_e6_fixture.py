"""Separate corpus and real FAISS persistence using offline injected MiniLM."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.config import REPO_ROOT, load_config
from app.data_audit import file_sha256
from app.embedder import MiniLMEmbedder
from app.fixtures import (ATLAS_DOCUMENTS, FixtureError, build_atlas_bundle, bundle_hashes,
                          guard_fixture_path, load_atlas, validate_atlas_bundle)
from app.indexer import IndexCompatibilityError, load_bundle, save_bundle
from tests.test_e2_embedder import FakeBackend
from tests.test_e3_indexer import make_bundle


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=REPO_ROOT / "artifacts")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = self.root / "atlas.jsonl"
        self.fixture.write_bytes((REPO_ROOT / "data/fixtures/atlas.jsonl").read_bytes())
        base = load_config()
        self.config = replace(base, paths=replace(base.paths, index_dir=self.root / "indexes", artifact_dir=self.root / "artifacts"))
        save_bundle(make_bundle(), self.config.paths.index_dir / "scifact")

    def build(self):
        runtime = MiniLMEmbedder(backend=FakeBackend())
        with patch("app.fixtures.MiniLMEmbedder", return_value=runtime):
            return build_atlas_bundle(self.config, fixture_path=self.fixture,
                bundle_path=self.root / "atlas", run_id="fixture-test", local_files_only=True)

    def test_exact_corpus_schema_content_and_order(self):
        corpus = load_atlas(self.fixture)
        self.assertEqual(tuple(corpus.values()), ATLAS_DOCUMENTS)
        self.assertEqual(list(corpus), [f"F0{i}" for i in range(1, 7)])
        rows = [json.loads(line) for line in self.fixture.read_text().splitlines()]
        self.assertTrue(all(set(r) == {"_id", "title", "text"} for r in rows))
        for changes in ("extra", "wrong", "order", "missing"):
            bad = [dict(r) for r in rows]
            if changes == "extra": bad[0]["expected_status"] = "ANSWERED"
            if changes == "wrong": bad[0]["text"] = "changed"
            if changes == "order": bad.reverse()
            if changes == "missing": bad.pop()
            self.fixture.write_text("".join(json.dumps(r) + "\n" for r in bad))
            with self.assertRaises(FixtureError): load_atlas(self.fixture)

    def test_production_primitives_hashes_alignment_and_no_gold_indexed(self):
        before = bundle_hashes(self.config.paths.index_dir / "scifact")
        bundle = self.build()
        self.assertEqual(before, bundle_hashes(self.config.paths.index_dir / "scifact"))
        self.assertEqual(bundle.manifest["corpus"], {"name": "atlas_fixture", "documents": 6, "sha256": file_sha256(self.fixture)})
        self.assertEqual(bundle.index.ntotal, 6)
        self.assertEqual(bundle.index.d, 384)
        self.assertEqual(tuple(c.doc_id for c in bundle.chunks), tuple(d.doc_id for d in ATLAS_DOCUMENTS))
        self.assertEqual(tuple(c.text for c in bundle.chunks), tuple(d.text for d in ATLAS_DOCUMENTS))
        from app.evaluator import GENERATION_CASES
        encoded = (self.root / "atlas/chunks.jsonl").read_text()
        self.assertTrue(all(case.question not in encoded for case in GENERATION_CASES))
        self.assertNotIn("expected_status", encoded)
        with self.assertRaises(IndexCompatibilityError): load_bundle(self.root / "atlas", expected_corpus_sha256="0" * 64)

    def test_scifact_path_alias_and_nested_paths_rejected_before_embedding(self):
        canonical = self.config.paths.index_dir / "scifact"
        for target in (canonical, canonical / "nested", canonical.parent):
            with self.assertRaises(FixtureError): guard_fixture_path(target, canonical)
        with patch("app.fixtures.MiniLMEmbedder") as runtime, self.assertRaises(FixtureError):
            build_atlas_bundle(self.config, fixture_path=self.fixture, bundle_path=canonical, run_id="bad")
        runtime.assert_not_called()

    def test_wrong_fixture_name_rejected(self):
        bundle = self.build()
        bundle.manifest["corpus"]["name"] = "scifact_beir"
        with self.assertRaises(FixtureError): validate_atlas_bundle(bundle, load_atlas(self.fixture))


if __name__ == "__main__": unittest.main()
