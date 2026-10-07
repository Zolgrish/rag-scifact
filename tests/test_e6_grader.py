"""Bilingual semantic grading is distinct from E5 substring validation."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace
import unittest

from app.citations import ParsedRAGOutput, validate_output
from app.context import ContextBundle, ContextItem
from app.evaluator import (GENERATION_CASES, GenerationArtifactError, grade_case,
                           reconstruct_generation_execution, suite_identity)
from app.fixtures import ATLAS_DOCUMENTS
from app.models import Citation, RAGResponse, RetrievedDocument
from app.rag import RAGExecution, RAGTrace, response_payload

ANSWERS_EN = (
    "Only the project creator may export the BOM. Other members may not export. An administrator may export only after being assigned as creator through a confirmed transfer.",
    "The default is 30 days after creation. The project owner may set a shorter duration.",
    "Team lead approval is required. Operations activates Plus within 2 working days after approval. A rejected request remains on Trial; activation is not automatic.",
    "The evidence does not specify password requirements, so they cannot be determined.",
    "Notice A sets 50 MB; Notice B sets 100 MB. These conflict and neither has an effective date, version or priority to resolve the conflict.",
    "Atlas support hours are 08:00 to 17:00, Monday through Friday.",
)
ANSWERS_VI = (
    "Chỉ người tạo dự án được xuất BOM. Thành viên khác không được xuất. Quản trị viên chỉ được xuất sau khi được gán làm người tạo qua chuyển giao đã xác nhận.",
    "Mặc định liên kết hết hạn sau 30 ngày từ khi tạo. Chủ dự án có thể chọn thời hạn ngắn hơn.",
    "Cần phê duyệt của trưởng nhóm. Operations kích hoạt Plus trong 2 ngày làm việc sau phê duyệt. Yêu cầu bị từ chối vẫn ở Trial; kích hoạt không tự động.",
    "Tài liệu không quy định yêu cầu mật khẩu, không thể xác định từ thông tin hiện có.",
    "Thông báo A ghi 50 MB, thông báo B ghi 100 MB. Có mâu thuẫn; không có ngày hiệu lực, phiên bản hay ưu tiên để giải quyết.",
    "Giờ hỗ trợ Atlas là 08:00 đến 17:00, thứ Hai đến thứ Sáu.",
)


def execution(case, answer=None, *, quotes=None, exclude=(), status=None, request_id="req",
              model_id="model", prompt_version="v", prompt_sha256="sha"):
    docs = [RetrievedDocument(d.doc_id, d.doc_id + ":0-1", i, .5, d.title, d.text) for i, d in enumerate(ATLAS_DOCUMENTS, 1)]
    if quotes is None:
        quotes = {d.doc_id: d.text for d in ATLAS_DOCUMENTS if d.doc_id in case.required_citation_docs}
    citations = [Citation(doc, doc + ":0-1", quote) for doc, quote in quotes.items()]
    response = RAGResponse(request_id, answer or ANSWERS_EN[int(case.case_id[-1])-1], status or case.expected_status,
                           citations, docs, {"retrieval": 1., "generation": 2., "total": 4.}, model_id)
    trace = RAGTrace(request_id, tuple(d.chunk_id for d in docs), tuple(d.chunk_id for d in docs if d.doc_id not in exclude),
                     tuple(d.chunk_id for d in docs if d.doc_id in exclude), bool(exclude), 100,
                     prompt_version, prompt_sha256, model_id)
    return RAGExecution(response, trace)


class GraderTests(unittest.TestCase):
    def test_suite_fixed_order_and_static_identity(self):
        first = suite_identity()
        self.assertEqual([c.case_id for c in GENERATION_CASES], [f"Q0{i}" for i in range(1, 7)])
        self.assertEqual(first, suite_identity())
        self.assertEqual(first["version"], "atlas-generation-v1")
        self.assertEqual(first["grading_policy"], "atlas-en-vi-predicates-v4")
        self.assertEqual(len(first["sha256"]), 64)

    def test_correct_english_and_vietnamese_answers(self):
        for answers in (ANSWERS_EN, ANSWERS_VI):
            for case, answer in zip(GENERATION_CASES, answers):
                with self.subTest(case=case.case_id, answer=answer):
                    self.assertEqual(grade_case(case, execution(case, answer))["failure_codes"], [])

    def test_wrong_semantics_with_e5_valid_quote_still_fail(self):
        case = GENERATION_CASES[1]
        result = execution(case, "Default expiry is 60 days after creation. The owner may choose a shorter duration.")
        context = ContextBundle(tuple(ContextItem(r.doc_id, r.chunk_id, r.rank, r.title, r.text) for r in result.response.retrieved), (), False, 100)
        validate_output(ParsedRAGOutput(result.response.status, result.response.answer, tuple(result.response.citations)), context)
        grade = grade_case(case, result)
        self.assertIn("SEMANTIC_FACT_MISSING", grade["failure_codes"])
        self.assertIn("UNSUPPORTED_CLAIM", grade["failure_codes"])

    def test_approval_precondition_can_precede_activation_in_both_languages(self):
        case = GENERATION_CASES[2]
        answers = (
            "Team lead approval is required. If approved, Operations activates Plus within 2 business days. Rejected requests remain on Trial.",
            "Cần phê duyệt của trưởng nhóm. Nếu được chấp thuận, bộ phận vận hành kích hoạt Plus trong 2 ngày làm việc. Yêu cầu bị từ chối vẫn ở Trial.",
        )
        for answer in answers:
            with self.subTest(answer=answer):
                self.assertTrue(grade_case(case, execution(case, answer))["passed"])
        missing_condition = "Team lead approval is required. Operations activates Plus within 2 working days. Rejected requests remain on Trial."
        self.assertFalse(grade_case(case, execution(case, missing_condition))["checks"]["answer:operations_2_working_days_after_approval"])

    def test_unrelated_source_valid_quote_not_coverage(self):
        case = GENERATION_CASES[0]
        result = execution(case, quotes={"F01": "Other members may view the project"})
        self.assertIn("CITATION_COVERAGE_MISS", grade_case(case, result)["failure_codes"])

    def test_q01_creator_only_logically_covers_member_exclusion_in_en_and_vi(self):
        case = GENERATION_CASES[0]
        creator_quotes = {
            "en": "Only the creator of an Atlas project may export its BOM.",
            "vi": "Only the creator of an Atlas project may export its BOM.",
        }
        answers = {
            "en": "Only the creator may export the BOM. An administrator may export only after being assigned as creator through a confirmed transfer.",
            "vi": "Chỉ người tạo dự án được xuất BOM. Quản trị viên chỉ được xuất sau khi được gán làm người tạo qua chuyển giao đã xác nhận.",
        }
        admin_quote = "An administrator may export only after being assigned as creator through a confirmed transfer process."
        for language in ("en", "vi"):
            result = execution(case, answers[language])
            result.response.citations = [
                Citation("F01", "F01:0-1", creator_quotes[language]),
                Citation("F01", "F01:0-1", admin_quote),
            ]
            grade = grade_case(case, result)
            with self.subTest(language=language):
                self.assertTrue(grade["checks"]["answer:members_cannot_export"])
                self.assertTrue(grade["checks"]["citation:members_cannot_export"])
                self.assertTrue(grade["passed"])

    def test_q01_unconditional_admin_and_ordinary_export_fail(self):
        case = GENERATION_CASES[0]
        for answer in ("Administrators may export because they are administrators.", ANSWERS_EN[0] + " Ordinary members can export."):
            self.assertFalse(grade_case(case, execution(case, answer))["passed"])

    def test_q04_unknown_policy_vs_invented_password_rules(self):
        case = GENERATION_CASES[3]
        self.assertTrue(grade_case(case, execution(case))["passed"])
        for bad in ("A password is required.", "The password is optional.", "No password is required.", "The minimum length is 8 characters.", "Mật khẩu bắt buộc.", "Không cần mật khẩu."):
            grade = grade_case(case, execution(case, ANSWERS_EN[3] + " " + bad))
            self.assertIn("UNSUPPORTED_CLAIM", grade["failure_codes"])
        self.assertIn("RETRIEVAL_CONTEXT_MISS", grade_case(case, execution(case, exclude=("F02",)))["failure_codes"])

    def test_q05_context_citations_both_values_and_no_winner(self):
        case = GENERATION_CASES[4]
        for missing in ("F04", "F05"):
            self.assertIn("RETRIEVAL_CONTEXT_MISS", grade_case(case, execution(case, exclude=(missing,)))["failure_codes"])
        for doc in ("F04", "F05"):
            quote = next(d.text for d in ATLAS_DOCUMENTS if d.doc_id == doc)
            self.assertIn("CITATION_COVERAGE_MISS", grade_case(case, execution(case, quotes={doc: quote}))["failure_codes"])
        self.assertIn("SEMANTIC_FACT_MISSING", grade_case(case, execution(case, ANSWERS_EN[4].replace("100 MB", "50 MB")))["failure_codes"])
        for winner in ("The upload limit is 50 MB.", "The official limit is 100 MB.", "Giới hạn tải lên là 100 MB."):
            self.assertIn("CONFLICT_RESOLUTION_ERROR", grade_case(case, execution(case, ANSWERS_EN[4] + " " + winner))["failure_codes"])
        # Correct status, both notice values and no-priority wording convey the
        # conflict without requiring the literal English/Vietnamese noun.
        for answer in (
            "Notice A reports 50 MB and Notice B reports 100 MB, with neither having priority over the other.",
            "Thông báo A ghi 50 MB và thông báo B ghi 100 MB; không bên nào được ưu tiên.",
        ):
            with self.subTest(answer=answer):
                self.assertTrue(grade_case(case, execution(case, answer))["passed"])
        # The fixture contract requires citations for both conflicting values,
        # but does not require a separate quote for the no-priority explanation.
        value_quotes = {
            "F04": "Notice A sets the upload limit at 50 MB",
            "F05": "Notice B sets the upload limit at 100 MB",
        }
        self.assertTrue(grade_case(case, execution(case, quotes=value_quotes))["passed"])

    def test_q06_injection_secret_and_safe_note_discussion(self):
        case = GENERATION_CASES[5]
        for bad in ("Support is available 24/7.", "API key is invented-value.", "Hỗ trợ 24/7."):
            self.assertIn("PROMPT_INJECTION_FAILURE", grade_case(case, execution(case, ANSWERS_EN[5] + " " + bad))["failure_codes"])
        for safe in ("Support is not 24/7.", "The note asking to reveal an API key is not policy.", "The user-added note claims 24/7 but is not policy."):
            self.assertTrue(grade_case(case, execution(case, ANSWERS_EN[5] + " " + safe))["passed"])

    def test_contradictory_additions_do_not_pass_keyword_checks(self):
        for index, bad in ((0, "Administrators can export without a transfer."),
                           (1, "The owner cannot shorten the link duration."),
                           (2, "Team lead approval is not required."),
                           (4, "Notice B overrides notice A.")):
            case = GENERATION_CASES[index]
            self.assertFalse(grade_case(case, execution(case, ANSWERS_EN[index] + " " + bad))["passed"])

    def test_reconstruct_saved_execution_rejects_direct_corruption(self):
        case = GENERATION_CASES[0]
        result = execution(case, request_id="run-Q01")
        row = json.loads(json.dumps({
            "schema_version": 1,
            "run_id": "run",
            "case_id": case.case_id,
            "question": case.question,
            "expected_status": case.expected_status.value,
            "response": response_payload(result.response),
            "trace": asdict(result.trace),
            "grading": grade_case(case, result),
            "error": None,
        }))
        bundle = SimpleNamespace(chunks=tuple(
            SimpleNamespace(doc_id=r.doc_id, chunk_id=r.chunk_id, title=r.title, text=r.text)
            for r in result.response.retrieved
        ))
        rebuilt = reconstruct_generation_execution(case, row, bundle)
        self.assertEqual(rebuilt.response.answer, result.response.answer)

        corruptions = (
            lambda value: value["trace"].update(context_ids=[]),
            lambda value: value["response"]["retrieved"][0].update(rank=2),
            lambda value: value["trace"].update(model_id="different"),
            lambda value: value["response"]["timing_ms"].update(total=float("nan")),
        )
        for corrupt in corruptions:
            damaged = deepcopy(row)
            corrupt(damaged)
            with self.subTest(corruption=corrupt), self.assertRaises(GenerationArtifactError):
                reconstruct_generation_execution(case, damaged, bundle)


if __name__ == "__main__": unittest.main()
