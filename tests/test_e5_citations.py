"""Strict schema, context-only verbatim citations, structural semantic statuses."""
import json
import unittest

from app.citations import RAGOutputValidationError, parse_output, validate_output
from app.context import ContextBundle, ContextItem


def output(status="ANSWERED", citations=None, answer="Supported claim."):
    return json.dumps({"status": status, "answer": answer, "citations":
                       [{"doc_id": "a", "chunk_id": "a:0-10", "quote": "Exact fact 12."}]
                       if citations is None else citations})


def citation(doc="a", chunk="a:0-10", quote="Exact fact 12."):
    return {"doc_id": doc, "chunk_id": chunk, "quote": quote}


class CitationTests(unittest.TestCase):
    def setUp(self):
        self.context = ContextBundle((ContextItem("a", "a:0-10", 1, "T", "Exact fact 12.\r\nMore\t  evidence."),
                                      ContextItem("b", "b:0-10", 2, "T", "Opposite claim.")),
                                     (ContextItem("c", "c:0-10", 3, "T", "Excluded evidence."),), True, 50)

    def test_valid_schema_exact_and_whitespace_quote(self):
        for quote in ("Exact fact 12.", "12. More evidence.", "12.\n More\t evidence."):
            validate_output(parse_output(output(citations=[citation(quote=quote)])), self.context)

    def test_parser_rejects_nonstandard_json_and_schema(self):
        base = json.loads(output())
        variants = ["{", "```json\n" + output() + "\n```", "prefix " + output(),
                    output() + " suffix", "[]", "null", output().replace('"ANSWERED"', 'NaN'),
                    output().replace('"ANSWERED"', 'Infinity'),
                    output().replace('"status": "ANSWERED"', '"status":"ANSWERED","status":"ANSWERED"'),
                    output().replace('"quote": "Exact fact 12."', '"quote":"x","quote":"x"')]
        for key in base:
            variants.append(json.dumps({k: v for k, v in base.items() if k != key}))
        for change in ({"unexpected": 1}, {"status": "UNKNOWN"}, {"status": False},
                       {"answer": " "}, {"answer": 1}, {"citations": {}},
                       {"citations": [1]}, {"citations": [citation(quote=" ")]},
                       {"citations": [{**citation(), "extra": 1}]},
                       {"citations": [citation(doc=42)]}, {"citations": [{"doc_id": "a"}]}):
            variants.append(json.dumps({**base, **change}))
        for text in variants:
            with self.subTest(text=text), self.assertRaises(RAGOutputValidationError):
                parse_output(text)

    def test_invalid_citation_rejects_whole_answer(self):
        variants = [citation(doc="fake"), citation(chunk="fake"), citation(doc="b"),
                    citation(quote="Opposite claim."), citation(quote="The fact is twelve."),
                    citation(quote="exact fact 12."), citation(quote="Exact fact 12!"),
                    citation(quote="Exact fact 13."),
                    citation("c", "c:0-10", "Excluded evidence.")]
        for bad in variants:
            with self.subTest(bad=bad), self.assertRaises(RAGOutputValidationError):
                validate_output(parse_output(output(citations=[citation(), bad])), self.context)

    def test_identical_duplicate_citation_rejected(self):
        with self.assertRaisesRegex(RAGOutputValidationError, "Duplicate citation"):
            validate_output(parse_output(output(citations=[citation(), citation()])), self.context)

    def test_status_structure(self):
        validate_output(parse_output(output()), self.context)
        validate_output(parse_output(output("INSUFFICIENT_EVIDENCE", [])), self.context)
        validate_output(parse_output(output("INSUFFICIENT_EVIDENCE")), self.context)
        validate_output(parse_output(output("CONFLICTING_EVIDENCE", [citation(),
                        citation("b", "b:0-10", "Opposite claim.")])), self.context)
        for status, cites in (("ANSWERED", []), ("CONFLICTING_EVIDENCE", [citation()]),
                              ("CONFLICTING_EVIDENCE", [citation(), citation(quote="More evidence.")]),
                              ("INSUFFICIENT_EVIDENCE", [citation(doc="fake")])):
            with self.subTest(status=status), self.assertRaises(RAGOutputValidationError):
                validate_output(parse_output(output(status, cites)), self.context)


if __name__ == "__main__":
    unittest.main()
