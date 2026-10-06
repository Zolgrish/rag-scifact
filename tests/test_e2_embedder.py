"""Offline E2 embedding contracts using a deterministic injected backend."""

from dataclasses import replace
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from app.chunker import MiniLMChunker, default_e2_config
from app.embedder import EmbeddingError, MiniLMEmbedder, MODEL_REVISION
from app.loader import CorpusDocument
from tests.test_e2_chunker import FakeTokenizer


class FakeBackend:
    tokenizer = FakeTokenizer()
    max_seq_length = 256
    device = "cpu"

    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        if self.result is not None:
            return self.result
        # Deliberately unnormalized float64: wrapper must enforce the contract.
        return np.array([[1 + sum(map(ord, text)) % 23] + [2] * 383 for text in texts], dtype=np.float64)


class EmbedderTests(unittest.TestCase):
    def test_batch_single_empty_and_query_share_semantics(self) -> None:
        backend = FakeBackend()
        embedder = MiniLMEmbedder(backend=backend)
        texts = ["first scientific text", "second text", "Unicode caf\u00e9."]
        batch = embedder.encode_texts(texts, batch_size=2)
        self.assertEqual(batch.shape, (3, 384))
        self.assertEqual(batch.dtype, np.float32)
        self.assertTrue(np.isfinite(batch).all())
        np.testing.assert_allclose(np.linalg.norm(batch, axis=1), 1, atol=1e-6)
        single = embedder.encode_query(texts[0])
        self.assertEqual(single.shape, (384,))
        np.testing.assert_allclose(single, embedder.encode_texts([texts[0]])[0], atol=1e-6)
        np.testing.assert_allclose(single, batch[0], atol=1e-6)
        empty = embedder.encode_texts([])
        self.assertEqual(empty.shape, (0, 384))
        self.assertEqual(empty.dtype, np.float32)
        self.assertEqual(len(backend.calls), 3)
        for _, kwargs in backend.calls:
            self.assertTrue(kwargs["normalize_embeddings"])
            self.assertEqual(kwargs["prompt"], "")
            self.assertEqual(kwargs["precision"], "float32")

    def test_chunks_use_shared_tokenizer_and_exact_embedding_text(self) -> None:
        backend = FakeBackend()
        embedder = MiniLMEmbedder(backend=backend)
        chunker = MiniLMChunker.from_embedder(embedder)
        self.assertIs(chunker.tokenizer, embedder.tokenizer)
        chunks = chunker.chunk_document(CorpusDocument("doc", "title", "evidence body"))
        encoded = embedder.encode_chunks(chunks)
        np.testing.assert_array_equal(encoded, embedder.encode_texts([chunks[0].embedding_text]))
        self.assertEqual(backend.calls[0][0], ["title\n\nevidence body"])
        self.assertEqual(embedder.runtime_metadata()["device"], "cpu")
        self.assertEqual(embedder.runtime_metadata()["revision"], MODEL_REVISION)
        self.assertEqual(chunker.runtime_metadata()["chunk_size_tokens"], 220)

    def test_wrong_dimension_and_batch_shape_rejected(self) -> None:
        for shape in ((1, 383), (1, 385), (384,), (2, 384)):
            with self.subTest(shape=shape), self.assertRaisesRegex(EmbeddingError, "shape"):
                MiniLMEmbedder(backend=FakeBackend(np.ones(shape))).encode_texts(["text"])
        class WrongDimension(FakeBackend):
            def get_sentence_embedding_dimension(self):
                return 768
        with self.assertRaisesRegex(EmbeddingError, "dimension must be 384"):
            MiniLMEmbedder(backend=WrongDimension())

    def test_nonfinite_zero_and_tiny_vectors(self) -> None:
        for value in (np.nan, np.inf, -np.inf, 0.0):
            with self.subTest(value=value), self.assertRaises(EmbeddingError):
                MiniLMEmbedder(backend=FakeBackend(np.full((1, 384), value))).encode_query("text")
        # Stable norm computation even for very small/large finite float32 values.
        for value in (1e-38, 1e38):
            vector = MiniLMEmbedder(backend=FakeBackend(np.full((1, 384), value))).encode_query("text")
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)

    def test_constructor_pins_verified_model_revision(self) -> None:
        calls = {}

        def factory(model_id, **kwargs):
            calls["model_id"] = model_id
            calls["kwargs"] = kwargs
            return FakeBackend()

        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = factory
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            embedder = MiniLMEmbedder()

        self.assertEqual(calls["model_id"], default_e2_config().model_id)
        self.assertEqual(calls["kwargs"]["revision"], MODEL_REVISION)
        self.assertEqual(embedder.runtime_metadata()["revision"], MODEL_REVISION)

    def test_config_is_rejected_before_loading_runtime(self) -> None:
        for key, value in (("model_id", "other"), ("chunk_size_tokens", 221), ("chunk_overlap_tokens", 0), ("normalize", False), ("dtype", "float16")):
            with self.subTest(key=key), self.assertRaisesRegex(EmbeddingError, "locked"):
                MiniLMEmbedder(replace(default_e2_config(), **{key: value}))

    def test_actual_runtime_input_limit_and_no_silent_truncation(self) -> None:
        backend = FakeBackend()
        backend.tokenizer = FakeTokenizer()
        backend.tokenizer.model_max_length = 128
        embedder = MiniLMEmbedder(backend=backend)
        self.assertEqual(embedder.effective_input_limit, 128)
        with self.assertRaisesRegex(EmbeddingError, "limit is 128"):
            embedder.encode_query("word " * 127)
        self.assertEqual(backend.calls, [])
        backend.tokenizer.model_max_length = 10**30
        self.assertEqual(MiniLMEmbedder(backend=backend).effective_input_limit, 256)
        backend.max_seq_length = None
        with self.assertRaisesRegex(EmbeddingError, "max_seq_length"):
            MiniLMEmbedder(backend=backend)

    def test_invalid_inputs_and_runtime_error(self) -> None:
        embedder = MiniLMEmbedder(backend=FakeBackend())
        for values in ([""], [" \n"], [None], "text"):
            with self.assertRaises(EmbeddingError):
                embedder.encode_texts(values)
        for batch_size in (0, -1, True):
            with self.assertRaises(EmbeddingError):
                embedder.encode_texts(["text"], batch_size=batch_size)
        class FailingBackend(FakeBackend):
            def encode(self, *args, **kwargs):
                raise RuntimeError("runtime unavailable")
        with self.assertRaisesRegex(EmbeddingError, "runtime unavailable") as captured:
            MiniLMEmbedder(backend=FailingBackend()).encode_query("text")
        self.assertIsInstance(captured.exception.__cause__, RuntimeError)

    def test_module_imports_do_not_import_third_party_packages(self) -> None:
        code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'torch', 'transformers', 'sentence_transformers'}:
        raise AssertionError('heavy import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import app.chunker
import app.embedder
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
