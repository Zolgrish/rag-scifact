"""Independent toy math including fractional grades and production document dedup."""
import math
import unittest
from dataclasses import replace

from app.models import RetrievedDocument
from app.retrieval_evaluator import score_query, validate_results, RetrievalEvaluationError
from app.retriever import DenseRetriever
from tests.test_e3_indexer import chunk, make_bundle
from tests.test_e3_retriever import QueryEmbedder, score_vectors


def results(docs):
    return [RetrievedDocument(doc, doc + ":0-1", i, 1 / i) for i, doc in enumerate(docs, 1)]


class MetricTests(unittest.TestCase):
    def test_multi_relevant_recall_and_first_hit_mrr(self):
        scored = score_query(results(["x", "a", "z", "y", "u", "b", "v", "w", "t", "c"]),
                             {"a": 1., "b": 1., "c": 1., "missing": 1.})
        self.assertEqual(scored["recall@5"], .25)
        self.assertEqual(scored["recall@10"], .75)
        self.assertEqual(scored["mrr@10"], .5)

    def test_fractional_graded_ndcg_not_binary_or_integer(self):
        gold = {"a": 2.5, "b": .5, "c": 1.25}
        scored = score_query(results(["b", "a", "x", "c"]), gold)
        dcg = (2**.5 - 1) + (2**2.5 - 1) / math.log2(3) + (2**1.25 - 1) / math.log2(5)
        idcg = (2**2.5 - 1) + (2**1.25 - 1) / math.log2(3) + (2**.5 - 1) / math.log2(4)
        self.assertAlmostEqual(scored["ndcg@10"], dcg / idcg, places=14)
        self.assertAlmostEqual(score_query(results(["a", "c", "b"]), gold)["ndcg@10"], 1.)

    def test_no_hits_and_zero_positive_qrels_are_zero(self):
        for gold in ({"a": 1.}, {"a": 0.}, {}):
            self.assertEqual(set(score_query(results(["x"]), gold).values()), {0.})

    def test_finite_extreme_grades_do_not_overflow(self):
        for grades in ({"a": 1022., "b": 1021.}, {"a": 1e308, "b": 1.}):
            self.assertEqual(score_query(results(["a", "b"]), grades)["ndcg@10"], 1.)

    def test_duplicates_ranks_order_scores_and_grades_rejected(self):
        base = results(["a", "b"])
        bad = ([base[0], replace(base[1], doc_id="a")], [replace(base[0], rank=2)],
               [replace(base[0], rank=True)], [replace(base[0], score=math.nan)],
               [base[0], replace(base[1], score=2.)])
        for values in bad:
            with self.subTest(values=values), self.assertRaises(RetrievalEvaluationError):
                validate_results(values)
        for grade in (math.nan, math.inf, -1., True):
            with self.subTest(grade=grade), self.assertRaises(RetrievalEvaluationError):
                score_query(base, {"a": grade})

    def test_metrics_use_production_dedup_before_cutoff(self):
        chunks = [chunk("a", i) for i in range(8)] + [chunk("b"), chunk("c")]
        bundle = make_bundle(chunks, score_vectors([.9] * 8 + [.8, .7]))
        ranked = DenseRetriever(bundle, QueryEmbedder()).retrieve("arbitrary question", 10)
        self.assertEqual([r.doc_id for r in ranked], ["a", "b", "c"])
        self.assertEqual(score_query(ranked, {"b": 1.})["mrr@10"], .5)


if __name__ == "__main__": unittest.main()
