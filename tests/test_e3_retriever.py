"""Offline dense retrieval tests with exact synthetic FAISS vectors."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import faiss
import numpy as np

from app.embedder import EmbeddingError, MiniLMEmbedder
from app.indexer import IndexCompatibilityError, IndexLoadError, load_bundle, save_bundle
from app.retriever import (
    DenseRetriever,
    QueryValidationError,
    RetrievalConfigurationError,
)
from tests.test_e2_embedder import FakeBackend
from tests.test_e3_indexer import chunk, make_bundle


class QueryEmbedder:
    def __init__(self, vector=None):
        self.vector = vector if vector is not None else np.eye(1, 384, dtype=np.float32)[0]
        self.calls = []

    def runtime_metadata(self):
        return MiniLMEmbedder(backend=FakeBackend()).runtime_metadata()

    def encode_query(self, query):
        self.calls.append(query)
        return self.vector


def score_vectors(scores):
    result = np.zeros((len(scores), 384), dtype=np.float32)
    for i, score in enumerate(scores):
        result[i, 0] = score
        result[i, 1] = np.sqrt(1 - score**2)
    return result


class DenseRetrieverTests(unittest.TestCase):
    def test_dedup_ranking_best_chunk_exact_evidence_and_one_embedding(self):
        chunks = [chunk("b", 0), chunk("a", 0, "First exact source"),
                  chunk("b", 1, "Best\tUnicode caf\u00e9 evidence!"), chunk("c", 0)]
        bundle = make_bundle(chunks, score_vectors([0.5, 0.8, 0.9, 0.2]))
        embedder = QueryEmbedder()
        results = DenseRetriever(bundle, embedder).retrieve("  arbitrary claim \n", 3)
        self.assertEqual(embedder.calls, ["arbitrary claim"])
        self.assertEqual([r.doc_id for r in results], ["b", "a", "c"])
        self.assertEqual([r.rank for r in results], [1, 2, 3])
        self.assertEqual(results[0].chunk_id, chunks[2].chunk_id)
        self.assertEqual(results[0].text, chunks[2].text)
        self.assertAlmostEqual(results[0].score, 0.9, places=6)
        self.assertEqual([r.score for r in results], sorted([r.score for r in results], reverse=True))

    def test_adaptive_expansion_when_many_best_chunks_share_a_document(self):
        chunks = [chunk("many", i) for i in range(60)] + [chunk("second"), chunk("third")]
        bundle = make_bundle(chunks, score_vectors([0.9] * 60 + [0.8, 0.7]))
        calls = []
        actual = faiss.IndexFlatIP.search
        def search(index, query, k, **kwargs):
            calls.append(k)
            return actual(index, query, k, **kwargs)
        embedder = QueryEmbedder()
        with patch.object(faiss.IndexFlatIP, "search", search):
            results = DenseRetriever(bundle, embedder).retrieve("claim", 3)
        self.assertEqual(calls, [32, 62])
        self.assertEqual([r.doc_id for r in results], ["many", "second", "third"])
        self.assertEqual(results[0].chunk_id, chunks[0].chunk_id)
        self.assertEqual(embedder.calls, ["claim"])

    def test_cutoff_ties_expand_and_use_vector_position_order(self):
        chunks = [chunk(str(i)) for i in range(70)]
        bundle = make_bundle(chunks, score_vectors([0.5] * 70))
        results = DenseRetriever(bundle, QueryEmbedder()).retrieve("tie", 5)
        self.assertEqual([r.doc_id for r in results], ["0", "1", "2", "3", "4"])

    def test_small_index_returns_all_available_docs(self):
        bundle = make_bundle([chunk("only", 0), chunk("only", 1)])
        results = DenseRetriever(bundle, QueryEmbedder()).retrieve("text", 10)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].rank, 1)

    def test_validation_rejects_query_and_topk_before_embedding(self):
        embedder = QueryEmbedder()
        retriever = DenseRetriever(make_bundle(), embedder)
        for query in ("", " \n\t", "x" * 2001, None, 123):
            with self.assertRaises(QueryValidationError):
                retriever.retrieve(query, 5)
        for k in (0, 11, True, False, 5.0, "5"):
            with self.assertRaises(QueryValidationError):
                retriever.retrieve("valid", k)
        self.assertEqual(embedder.calls, [])
        retriever.retrieve(" " + "x" * 2000 + " ", 1)
        self.assertEqual(len(embedder.calls[0]), 2000)


    def test_configured_max_top_k_is_enforced(self):
        embedder = QueryEmbedder()
        retriever = DenseRetriever(make_bundle(), embedder, max_top_k=3)
        with self.assertRaises(QueryValidationError):
            retriever.retrieve("claim", 4)
        self.assertEqual(embedder.calls, [])
        self.assertEqual(len(retriever.retrieve("claim", 3)), 2)

        for invalid in (0, 11, True, 3.0, "3"):
            with self.subTest(max_top_k=invalid), self.assertRaises(
                RetrievalConfigurationError
            ):
                DenseRetriever(make_bundle(), QueryEmbedder(), max_top_k=invalid)

    def test_invalid_query_vectors_are_embedding_errors(self):
        for vector in (np.ones((1, 384), np.float32), np.ones(383, np.float32),
                       np.eye(1, 384)[0], np.zeros(384, np.float32),
                       np.ones(384, np.float32), np.full(384, np.nan, np.float32), [1] * 384):
            with self.assertRaises(EmbeddingError):
                DenseRetriever(make_bundle(), QueryEmbedder(vector)).retrieve("text", 1)

    def test_empty_index_and_runtime_mismatch_remain_infrastructure_errors(self):
        bundle = make_bundle()
        with self.assertRaises(IndexLoadError):
            DenseRetriever(replace(bundle, index=faiss.IndexFlatIP(384), chunks=()), QueryEmbedder())
        class OtherRevision(QueryEmbedder):
            def runtime_metadata(self):
                return {**super().runtime_metadata(), "revision": "other"}
        with self.assertRaises(IndexCompatibilityError):
            DenseRetriever(bundle, OtherRevision())
        with self.assertRaises(IndexLoadError):
            load_bundle(Path("missing-e3-test-bundle"))

    def test_reload_returns_identical_results(self):
        bundle = make_bundle([chunk("a", 0), chunk("a", 1), chunk("b")], score_vectors([0.2, 0.8, 0.6]))
        with tempfile.TemporaryDirectory() as tmp:
            save_bundle(bundle, tmp)
            loaded = load_bundle(tmp)
            self.assertEqual(DenseRetriever(bundle, QueryEmbedder()).retrieve("query", 5),
                             DenseRetriever(loaded, QueryEmbedder()).retrieve("query", 5))


if __name__ == "__main__":
    unittest.main()
