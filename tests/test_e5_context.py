"""Offline whole-chunk context budgeting contracts."""
import json
from dataclasses import FrozenInstanceError
import unittest

from app.context import ContextBudgetError, build_context
from app.models import RetrievedDocument
from app.prompt import build_messages


def documents(count=5):
    return [RetrievedDocument(str(i), f"{i}:0-220", i, 0.5,
                              f"Title {i}", f"Exact evidence {i}\n\tΩ!")
            for i in range(1, count + 1)]


class ContextTests(unittest.TestCase):
    def build(self, docs, budget, counter=None):
        self.calls = []
        def count(messages):
            self.calls.append(messages)
            return (counter(messages) if counter else
                    10 + 10 * len(json.loads(messages[1]["content"])["context"]))
        return build_context("Question", docs, count_prompt_tokens=count,
                             context_length=budget + 5, max_new_tokens=5)

    def test_all_fit_and_exact_fit_count_full_chat_once(self):
        for budget in (60, 61):
            context = self.build(documents(), budget)
            self.assertEqual(len(self.calls), 1)
            self.assertEqual(self.calls[0], build_messages("Question", context.included))
            self.assertEqual([c.rank for c in context.included], [1, 2, 3, 4, 5])
            self.assertEqual(context.input_token_count, 60)
            self.assertEqual(context.excluded, ())
            self.assertFalse(context.truncated)
            self.assertEqual(context.included[0].text, documents()[0].text)
            with self.assertRaises(FrozenInstanceError):
                context.included[0].text = "different"

    def test_one_over_budget_keeps_largest_whole_prefix(self):
        context = self.build(documents(), 59)
        self.assertEqual([i.rank for i in context.included], [1, 2, 3, 4])
        self.assertEqual([i.rank for i in context.excluded], [5])
        self.assertTrue(context.truncated)
        self.assertEqual(context.input_token_count, 50)
        self.assertEqual(context.included[-1].text, documents()[3].text)
        self.assertLessEqual(len(self.calls), 6)

    def test_prefix_never_skips_large_item_or_slices(self):
        docs = documents(3)
        docs[1] = RetrievedDocument("2", "2:0-220", 2, 0.2, "title", "HUGE")
        def count(messages):
            items = json.loads(messages[1]["content"])["context"]
            return 10 + sum(100 if i["text"] == "HUGE" else 10 for i in items)
        context = self.build(docs, 35, count)
        self.assertEqual([i.doc_id for i in context.included], ["1"])
        self.assertEqual([i.doc_id for i in context.excluded], ["2", "3"])
        self.assertEqual(context.excluded[0].text, "HUGE")

    def test_largest_prefix_does_not_assume_token_counts_are_monotonic(self):
        counts = {0: 5, 1: 8, 2: 20, 3: 15, 4: 25}
        def count(messages):
            prefix = len(json.loads(messages[1]["content"])["context"])
            return counts[prefix]
        context = self.build(documents(4), 16, count)
        self.assertEqual([i.rank for i in context.included], [1, 2, 3])
        self.assertEqual([i.rank for i in context.excluded], [4])
        self.assertEqual(context.input_token_count, 15)
        self.assertTrue(context.truncated)

    def test_empty_and_zero_included(self):
        context = self.build([], 10)
        self.assertEqual(context.included, ())
        self.assertFalse(context.truncated)
        self.assertEqual(len(self.calls), 1)
        context = self.build(documents(1), 10)
        self.assertEqual(context.included, ())
        self.assertEqual(len(context.excluded), 1)
        self.assertTrue(context.truncated)

    def test_base_overflow_and_bad_counter(self):
        with self.assertRaises(ContextBudgetError):
            self.build(documents(), 9)
        for value in (True, -1, 0, 1.5):
            with self.subTest(value=value), self.assertRaises(ContextBudgetError):
                self.build([], 10, lambda _: value)

    def test_bad_configuration_or_rank(self):
        for window, reserve in ((5, 5), (10, 0), (True, 1), (10, -1)):
            with self.assertRaises(ContextBudgetError):
                build_context("Q", [], count_prompt_tokens=lambda _: 1,
                              context_length=window, max_new_tokens=reserve)
        with self.assertRaises(ContextBudgetError):
            self.build(list(reversed(documents())), 100)

    def test_full_rendered_input_counter_controls_budget_not_body_length(self):
        def count(messages):
            self.assertEqual(messages[0]["role"], "system")
            self.assertEqual(json.loads(messages[1]["content"])["question"], "Question")
            return 500 if json.loads(messages[1]["content"])["context"] else 11
        context = self.build(documents(1), 12, count)
        self.assertFalse(context.included)
        self.assertEqual(context.input_token_count, 11)


if __name__ == "__main__":
    unittest.main()
