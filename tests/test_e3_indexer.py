"""Offline E3 FAISS input, mapping, bundle and corruption tests."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import faiss
import numpy as np

from app.chunker import MiniLMChunker
from app.embedder import MiniLMEmbedder
from app.indexer import (
    IndexBuildError, IndexCompatibilityError, IndexLoadError,
    build_index, load_bundle, save_bundle,
)
from app.models import Chunk
from tests.test_e2_embedder import FakeBackend


def chunk(doc_id="doc", start=0, text="Exact caf\u00e9 evidence."):
    return Chunk(doc_id, f"{doc_id}:{start}-{start+1}", "Title", text,
                 start, start+1, 0, len(text), 1, "Title", False)


def vectors(count):
    result = np.zeros((count, 384), dtype=np.float32)
    for i in range(count):
        result[i, i % 384] = 1
    return result


def make_bundle(chunks=None, values=None):
    chunks = chunks if chunks is not None else [chunk("a"), chunk("b")]
    runtime = MiniLMEmbedder(backend=FakeBackend())
    return build_index(
        vectors(len(chunks)) if values is None else values, chunks, run_id="synthetic-build",
        corpus={"name": "synthetic", "sha256": "a" * 64, "documents": len({c.doc_id for c in chunks})},
        embedding=runtime.runtime_metadata(), chunking=MiniLMChunker.from_embedder(runtime).runtime_metadata(),
    )


class IndexBuildTests(unittest.TestCase):
    def test_e3_imports_do_not_load_heavy_libraries(self):
        code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'faiss', 'numpy', 'torch', 'transformers', 'sentence_transformers'}:
        raise AssertionError('heavy import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import app.indexer
import app.retriever
import scripts.build_index
import scripts.retrieve
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_builds_exact_flat_ip_and_preserves_alignment(self):
        bundle = make_bundle()
        self.assertIs(type(bundle.index), faiss.IndexFlatIP)
        self.assertEqual(bundle.index.d, 384)
        self.assertEqual(bundle.index.ntotal, len(bundle.chunks))
        scores, positions = bundle.index.search(vectors(1), 2)
        self.assertEqual(bundle.chunks[int(positions[0, 0])].doc_id, "a")
        self.assertEqual(scores[0, 0], 1.0)

    def test_rejects_invalid_vector_contract(self):
        invalid = [vectors(2).tolist(), vectors(2).astype(np.float64), np.ones((2, 383), np.float32),
                   vectors(1), vectors(2)[0], vectors(2) * 2, vectors(2) * 0]
        for value in (np.nan, np.inf, -np.inf):
            candidate = vectors(2)
            candidate[0, 0] = value
            invalid.append(candidate)
        for value in invalid:
            with self.subTest(shape=getattr(value, "shape", None)), self.assertRaises(IndexBuildError):
                make_bundle(values=value)

    def test_rejects_duplicate_or_invalid_chunks(self):
        with self.assertRaisesRegex(IndexBuildError, "Duplicate chunk_id"):
            make_bundle([chunk(), chunk()])
        for c in (replace(chunk(), char_end=2), replace(chunk(), body_token_count=221),
                  replace(chunk(), chunk_id="random"), replace(chunk(), title_truncated=1)):
            with self.assertRaises(IndexBuildError):
                make_bundle([c])
        with self.assertRaisesRegex(IndexBuildError, "empty"):
            make_bundle([])


class IndexPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "bundle"
        self.bundle = make_bundle()
        save_bundle(self.bundle, self.path)

    def edit_manifest(self, edit):
        path = self.path / "index_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        edit(manifest)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def rehash(self, name):
        payload = (self.path / name).read_bytes()
        self.edit_manifest(lambda m: m["files"].update({name: {
            "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload),
        }}))

    def edit_rows(self, edit):
        path = self.path / "chunks.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        edit(rows)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        self.rehash("chunks.jsonl")

    def test_roundtrip_search_alignment_and_positions(self):
        loaded = load_bundle(self.path, expected_corpus_sha256="a" * 64)
        self.assertEqual(loaded.chunks, self.bundle.chunks)
        self.assertIs(type(loaded.index), faiss.IndexFlatIP)
        self.assertEqual(loaded.manifest["mapping_row_count"], 2)
        rows = [json.loads(line) for line in (self.path / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["position"] for row in rows], [0, 1])
        for original, persisted in zip(self.bundle.index.search(vectors(1), 2), loaded.index.search(vectors(1), 2)):
            np.testing.assert_array_equal(original, persisted)

    def test_loading_never_loads_or_embeds_corpus(self):
        with patch("app.loader.load_corpus", side_effect=AssertionError("corpus loaded")), \
             patch("app.chunker.MiniLMChunker.chunk_document", side_effect=AssertionError("corpus chunked")), \
             patch("app.embedder.MiniLMEmbedder.__init__", side_effect=AssertionError("model loaded")), \
             patch("app.embedder.MiniLMEmbedder.encode_chunks", side_effect=AssertionError("corpus embedded")):
            self.assertEqual(load_bundle(self.path).index.ntotal, 2)

    def test_missing_files_and_directory_are_infrastructure_errors(self):
        with self.assertRaises(IndexLoadError):
            load_bundle(self.path / "absent")
        for name in ("index.faiss", "chunks.jsonl", "index_manifest.json"):
            with self.subTest(name=name):
                (self.path / name).unlink()
                with self.assertRaises(IndexLoadError):
                    load_bundle(self.path)
                save_bundle(self.bundle, self.path)

    def test_data_hash_mismatch(self):
        for name in ("index.faiss", "chunks.jsonl"):
            with self.subTest(name=name):
                with (self.path / name).open("ab") as handle:
                    handle.write(b"corruption")
                with self.assertRaisesRegex(IndexLoadError, "SHA256 mismatch"):
                    load_bundle(self.path)
                save_bundle(self.bundle, self.path)

    def test_corrupt_faiss_with_matching_hash_fails_parse(self):
        (self.path / "index.faiss").write_bytes(b"not a FAISS index")
        self.rehash("index.faiss")
        with self.assertRaisesRegex(IndexLoadError, "Cannot load") as captured:
            load_bundle(self.path)
        self.assertIsNotNone(captured.exception.__cause__)

    def test_malformed_mapping_and_nonfinite_json(self):
        for payload in (b"{broken}\n", b"[]\n", b"\n", b'{"position":NaN}\n', b"\xff"):
            with self.subTest(payload=payload):
                (self.path / "chunks.jsonl").write_bytes(payload)
                self.rehash("chunks.jsonl")
                with self.assertRaises(IndexLoadError):
                    load_bundle(self.path)

    def test_mapping_count_positions_and_duplicate_ids(self):
        for edit in (lambda rows: rows.pop(), lambda rows: rows[0].update(position=1),
                     lambda rows: rows[0].update(position=True), lambda rows: rows.append(rows[0]),
                     lambda rows: rows.__setitem__(1, {**rows[0], "position": 1}),
                     lambda rows: rows[0].update(char_end=999)):
            save_bundle(self.bundle, self.path)
            self.edit_rows(edit)
            with self.assertRaises(IndexLoadError):
                load_bundle(self.path)

    def test_manifest_count_schema_and_type_mismatch(self):
        for edit in (lambda m: m.update(ntotal=3, mapping_row_count=3),
                     lambda m: m.update(mapping_row_count=1), lambda m: m.update(ntotal=0),
                     lambda m: m.update(ntotal=True), lambda m: m.update(schema_version=2),
                     lambda m: m.update(dimension=768), lambda m: m.update(metric="l2")):
            save_bundle(self.bundle, self.path)
            self.edit_manifest(edit)
            with self.assertRaises(IndexLoadError):
                load_bundle(self.path)

    def test_incompatible_model_revision_chunk_and_corpus(self):
        for section, field, value in (
            ("embedding", "model_id", "other"), ("embedding", "revision", "other"),
            ("embedding", "dtype", "float64"), ("embedding", "normalized", False),
            ("chunking", "chunk_size_tokens", 221), ("chunking", "overlap_tokens", 31),
            ("chunking", "boundary_policy", "arbitrary"),
            ("chunking", "title_truncation_policy", "other"),
            ("chunking", "embedding_separator", " "),
        ):
            with self.subTest(field=field):
                save_bundle(self.bundle, self.path)
                self.edit_manifest(lambda m: m[section].update({field: value}))
                with self.assertRaises(IndexCompatibilityError):
                    load_bundle(self.path)
        save_bundle(self.bundle, self.path)
        with self.assertRaisesRegex(IndexCompatibilityError, "corpus SHA256"):
            load_bundle(self.path, expected_corpus_sha256="b" * 64)

    def test_actual_index_dimension_type_and_count_are_checked(self):
        for index in (faiss.IndexFlatIP(383), faiss.IndexFlatL2(384), faiss.IndexFlatIP(384)):
            index.add(np.ones((1, index.d), dtype=np.float32))
            faiss.write_index(index, str(self.path / "index.faiss"))
            self.rehash("index.faiss")
            with self.assertRaises(IndexLoadError):
                load_bundle(self.path)

    def test_stored_vector_norms_defensively_validated(self):
        index = faiss.IndexFlatIP(384)
        index.add(vectors(2) * 2)
        faiss.write_index(index, str(self.path / "index.faiss"))
        self.rehash("index.faiss")
        with self.assertRaisesRegex(IndexLoadError, "unit norms"):
            load_bundle(self.path)

    def test_manifest_parse_and_file_size_are_checked(self):
        for payload in ("[]", "{bad}", '{"schema_version":NaN}'):
            (self.path / "index_manifest.json").write_text(payload, encoding="utf-8")
            with self.assertRaises(IndexLoadError):
                load_bundle(self.path)
        save_bundle(self.bundle, self.path)
        self.edit_manifest(lambda m: m["files"]["index.faiss"].update(size_bytes=1))
        with self.assertRaisesRegex(IndexLoadError, "size mismatch"):
            load_bundle(self.path)

    def test_manifest_is_published_last_and_unrelated_files_survive(self):
        unrelated = self.path / "user-note.txt"
        unrelated.write_text("keep", encoding="utf-8")
        published = []
        real_replace = os.replace
        def capture(source, target):
            published.append(Path(target).name)
            real_replace(source, target)
        with patch("app.indexer.os.replace", side_effect=capture):
            save_bundle(self.bundle, self.path)
        self.assertEqual(published, ["index.faiss", "chunks.jsonl", "index_manifest.json"])
        self.assertEqual(unrelated.read_text(encoding="utf-8"), "keep")

    def test_failed_publication_cannot_load_partial_bundle(self):
        real_replace = os.replace
        def fail_mapping(source, target):
            if Path(target).name == "chunks.jsonl":
                raise OSError("simulated write failure")
            real_replace(source, target)
        new = self.path / "new"
        with patch("app.indexer.os.replace", side_effect=fail_mapping), self.assertRaises(IndexBuildError):
            save_bundle(self.bundle, new)
        with self.assertRaises(IndexLoadError):
            load_bundle(new)
        # If replacing an existing bundle, the old manifest rejects new data.
        changed = make_bundle(values=vectors(2)[::-1].copy())
        with patch("app.indexer.os.replace", side_effect=fail_mapping), self.assertRaises(IndexBuildError):
            save_bundle(changed, self.path)
        with self.assertRaisesRegex(IndexLoadError, "SHA256 mismatch"):
            load_bundle(self.path)
        self.assertFalse(list(self.path.parent.glob(".index-stage-*")))


if __name__ == "__main__":
    unittest.main()
