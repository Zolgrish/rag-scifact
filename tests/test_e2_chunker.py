"""Offline E2 token-window and source-span regressions."""

from dataclasses import replace
import re
import unittest

from app.chunker import (
    ChunkingError, MiniLMChunker, default_e2_config, source_span, token_windows,
)
from app.loader import CorpusDocument


class FakeEncoding(dict):
    def __init__(self, payload, word_ids):
        super().__init__(payload)
        self._word_ids = word_ids

    def word_ids(self):
        return list(self._word_ids)


class FakeTokenizer:
    """Offset-aware stand-in; real WordPiece behavior is validated separately."""
    is_fast = True
    model_max_length = 512

    def __call__(self, text, *, add_special_tokens=True, return_offsets_mapping=False, truncation=False):
        assert truncation is False
        spans = [(m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]
        token_ids = [
            1000 + sum((index + 1) * ord(ch) for index, ch in enumerate(text[start:end]))
            for start, end in spans
        ]
        if add_special_tokens:
            token_ids = [101] + token_ids + [102]
        result = {"input_ids": token_ids}
        if return_offsets_mapping:
            result["offset_mapping"] = spans
        word_ids = list(range(len(spans)))
        if add_special_tokens:
            word_ids = [None] + word_ids + [None]
        return FakeEncoding(result, word_ids)


class ChunkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunker = MiniLMChunker(FakeTokenizer(), 256)

    def test_determinism_source_spans_windows_and_separator(self) -> None:
        for count in (1, 30, 31, 189, 190, 219, 220, 221, 249, 250, 409, 410, 411):
            with self.subTest(count=count):
                doc = CorpusDocument("arbitrary:id", "Study title", " word" * count)
                chunks = self.chunker.chunk_document(doc)
                self.assertEqual(chunks, self.chunker.chunk_document(doc))
                self.assertEqual(len(chunks), len({c.chunk_id for c in chunks}))
                self.assertEqual([(c.token_start, c.token_end) for c in chunks], list(token_windows(count)))
                for chunk in chunks:
                    self.assertEqual(chunk.text, doc.text[chunk.char_start:chunk.char_end])
                    standalone_ids = self.chunker.tokenizer(
                        chunk.text, add_special_tokens=False, truncation=False,
                    )["input_ids"]
                    self.assertEqual(chunk.body_token_count, len(standalone_ids))
                    self.assertEqual(chunk.body_token_count, chunk.token_end - chunk.token_start)
                    self.assertLessEqual(chunk.body_token_count, 220)
                    self.assertEqual(chunk.chunk_id, f"arbitrary:id:{chunk.token_start}-{chunk.token_end}")
                    self.assertEqual(chunk.embedding_text, doc.title + "\n\n" + chunk.text)
                    self.assertFalse(chunk.title_truncated)
                for left, right in zip(chunks, chunks[1:]):
                    left_ids = self.chunker.tokenizer(
                        left.text, add_special_tokens=False, truncation=False,
                    )["input_ids"]
                    right_ids = self.chunker.tokenizer(
                        right.text, add_special_tokens=False, truncation=False,
                    )["input_ids"]
                    self.assertEqual(left.token_end - right.token_start, 30)
                    self.assertEqual(left_ids[-30:], right_ids[:30])
        other = self.chunker.chunk_document(CorpusDocument("other", "", "word"))[0]
        self.assertEqual(other.embedding_text, "word")
        self.assertEqual(other.chunk_id, "other:0-1")

    def test_empty_and_whitespace_body(self) -> None:
        for body in ("", " \n\t\r\u2003"):
            self.assertEqual(self.chunker.chunk_document(CorpusDocument("id", "Title", body)), [])

    def test_unicode_punctuation_and_interior_whitespace(self) -> None:
        body = "  Caf\u00e9,\tA\u0301!\n\u03b2:  \u4e2d\u6587\u2014test.  "
        chunk = self.chunker.chunk_document(CorpusDocument("unicode", "Original", body))[0]
        self.assertEqual(chunk.text, body[chunk.char_start:chunk.char_end])
        self.assertEqual(chunk.text, body.strip())
        self.assertIn("\t", chunk.text)
        self.assertIn("A\u0301", chunk.text)

    def test_long_title_preserves_body_and_logs_metadata(self) -> None:
        doc = CorpusDocument("long-title", "prefix " * 5000, "evidence " * 220)
        with self.assertLogs("app.chunker", level="INFO") as captured:
            chunk = self.chunker.chunk_document(doc)[0]
        self.assertTrue(chunk.title_truncated)
        self.assertEqual(chunk.title, doc.title)
        self.assertTrue(doc.title.startswith(chunk.embedding_title))
        self.assertEqual(chunk.text, doc.text[chunk.char_start:chunk.char_end])
        self.assertEqual(chunk.body_token_count, 220)
        self.assertLessEqual(self.chunker.count_tokens(chunk.embedding_text, add_special_tokens=True), 256)
        for field in ("doc_id=long-title", "original_title_tokens=5000", "retained_title_tokens=34", "model_input_limit=256", "body_token_count=220"):
            self.assertIn(field, captured.output[0])
        self.assertNotIn(doc.text, captured.output[0])

    def test_separator_is_measured_rather_than_assumed_free(self) -> None:
        class SeparatorTokenizer(FakeTokenizer):
            def __call__(self, text, **kwargs):
                result = super().__call__(text, **kwargs)
                if "\n\n" in text:
                    result["input_ids"].extend([999] * 3)
                return result
        chunker = MiniLMChunker(SeparatorTokenizer(), 256)
        with self.assertLogs("app.chunker", level="INFO"):
            chunk = chunker.chunk_document(CorpusDocument("s", "title " * 40, "body " * 220))[0]
        self.assertEqual(chunker.count_tokens(chunk.embedding_text, add_special_tokens=True), 256)
        self.assertEqual(chunker.count_tokens(chunk.embedding_title), 31)

    def test_body_is_never_silently_truncated(self) -> None:
        chunker = MiniLMChunker(FakeTokenizer(), 10)
        with self.assertRaisesRegex(ChunkingError, "Body evidence cannot fit"):
            chunker.chunk_document(CorpusDocument("b", "", "word " * 20))

    def test_invalid_runtime_and_locked_config(self) -> None:
        slow = FakeTokenizer()
        slow.is_fast = False
        with self.assertRaisesRegex(ChunkingError, "fast tokenizer"):
            MiniLMChunker(slow, 256)
        for limit in (0, -1, True, 256.0):
            with self.assertRaises(ChunkingError):
                MiniLMChunker(FakeTokenizer(), limit)
        for key, value in (("chunk_size_tokens", 219), ("chunk_overlap_tokens", 31), ("model_id", "other"), ("normalize", False), ("dtype", "float64")):
            with self.subTest(key=key), self.assertRaises(ChunkingError):
                MiniLMChunker(FakeTokenizer(), 256, config=replace(default_e2_config(), **{key: value}))

    def test_stable_boundaries_can_shorten_window_but_keep_exact_overlap(self) -> None:
        boundaries = [0, 189, 219, 250]
        self.assertEqual(
            list(token_windows(250, stable_boundaries=boundaries)),
            [(0, 219), (189, 250)],
        )

    def test_invalid_stable_boundaries_fail(self) -> None:
        for boundaries in ([1, 10], [0, 10], [0, 10, 9, 20], [0, 10, 10, 20]):
            with self.subTest(boundaries=boundaries), self.assertRaises(ChunkingError):
                list(token_windows(20, stable_boundaries=boundaries))

    def test_missing_and_invalid_offsets_fail_explicitly(self) -> None:
        class BadTokenizer(FakeTokenizer):
            def __call__(self, text, **kwargs):
                return FakeEncoding(
                    {"input_ids": [1], "offset_mapping": [(0, len(text) + 1)]},
                    [0],
                )
        with self.assertRaisesRegex(ChunkingError, "invalid source offsets"):
            MiniLMChunker(BadTokenizer(), 256).chunk_document(CorpusDocument("x", "", "word"))
        class MissingTokenizer(FakeTokenizer):
            def __call__(self, text, **kwargs):
                return {"input_ids": [1]}
        with self.assertRaisesRegex(ChunkingError, "reliable source offsets"):
            MiniLMChunker(MissingTokenizer(), 256).chunk_document(CorpusDocument("x", "", "word"))
        class NoTokens(FakeTokenizer):
            def __call__(self, text, **kwargs):
                return FakeEncoding({"input_ids": [], "offset_mapping": []}, [])
        with self.assertRaisesRegex(ChunkingError, "Non-empty body has no"):
            MiniLMChunker(NoTokens(), 256).chunk_document(CorpusDocument("x", "", "word"))


class TokenWindowTests(unittest.TestCase):
    def test_boundaries_cover_all_tokens_once_except_overlap(self) -> None:
        for count in (0, 1, 30, 31, 189, 190, 219, 220, 221, 249, 250, 409, 410, 411):
            with self.subTest(count=count):
                windows = list(token_windows(count))
                self.assertEqual(len(windows), len(set(windows)))
                covered = set()
                for start, end in windows:
                    self.assertLessEqual(end - start, 220)
                    self.assertGreater(end, start)
                    covered.update(range(start, end))
                self.assertEqual(covered, set(range(count)))
                for left, right in zip(windows, windows[1:]):
                    self.assertEqual(left[1] - right[0], 30)
                    self.assertEqual(left[1] - left[0], 220)
                if count:
                    self.assertEqual(windows[-1][1], count)
                if 0 < count <= 220:
                    self.assertEqual(windows, [(0, count)])
        self.assertEqual(list(token_windows(410)), [(0, 220), (190, 410)])
        self.assertEqual(list(token_windows(411)), [(0, 220), (190, 410), (380, 411)])

    def test_locked_window_config(self) -> None:
        for size, overlap in ((0, 30), (-1, 30), (221, 30), (220, -1), (220, 220), (220, 0), (220.0, 30)):
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ChunkingError):
                list(token_windows(1, chunk_size=size, overlap=overlap))
        for count in (-1, 1.5, True):
            with self.assertRaises(ChunkingError):
                list(token_windows(count))

    def test_source_span_preserves_unicode_and_formatting(self) -> None:
        text = "  café,\tβ!\n"
        offsets = [(2, 6), (6, 7), (8, 9), (9, 10)]
        start, end = source_span(offsets, 0, 4)
        self.assertEqual(text[start:end], "café,\tβ!")
        with self.assertRaises(ChunkingError):
            source_span(offsets, 1, 1)


if __name__ == "__main__":
    unittest.main()
