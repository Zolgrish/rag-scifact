"""Native exact counter uses E4 transport and never approximates token counts."""
import unittest
from unittest.mock import Mock
import requests

from app.generator import (GeneratorConnectionError, GeneratorConnectTimeoutError,
                           GeneratorReadTimeoutError, GeneratorModelMismatchError,
                           GeneratorMalformedResponseError, GeneratorOutOfMemoryError)
from tests.test_e4_generator import FakeResponse, make_generator, MODEL


def counted():
    return {"model": MODEL, "prompt_eval_count": 123, "eval_count": 1,
            "done": True, "done_reason": "length",
            "message": {"role": "assistant", "content": "{"}}


class CounterTests(unittest.TestCase):
    def test_exact_messages_options_endpoint_and_count(self):
        session = Mock()
        session.request.return_value = FakeResponse(counted())
        generator = make_generator(session=session)
        messages = [{"role": "system", "content": "Rules"}, {"role": "user", "content": "Q"}]
        self.assertEqual(generator.count_prompt_tokens(messages), 123)
        call = session.request.call_args
        self.assertEqual(call.args, ("POST", "http://127.0.0.1:11434/api/chat"))
        self.assertEqual(call.kwargs["json"], {"model": MODEL, "messages": messages,
                         "stream": False, "options": {"num_predict": 1, "temperature": 0, "seed": 42}})
        self.assertEqual(call.kwargs["timeout"], (10, 120))

    def test_explicit_json_mode_leaves_e4_default_request_unchanged(self):
        session = Mock()
        session.request.return_value = FakeResponse({"model": MODEL,
            "choices": [{"message": {"content": '{"answer":"ok"}'}}]})
        generator = make_generator(session=session)
        messages = [{"role": "user", "content": "Q"}]
        generator.generate(messages, response_format={"type": "json_object"})
        self.assertEqual(session.request.call_args.kwargs["json"]["response_format"],
                         {"type": "json_object"})
        generator.generate(messages)
        self.assertNotIn("response_format", session.request.call_args.kwargs["json"])

    def test_malformed_counter_and_model_mismatch_fail_closed(self):
        for changes in ({"prompt_eval_count": True}, {"prompt_eval_count": 0},
                        {"prompt_eval_count": "123"}, {"eval_count": 2}, {"eval_count": True},
                        {"done": False}, {"done_reason": "error"}, {"done_reason": {}}, {"message": {}},
                        {"message": {"role": "user", "content": "x"}}):
            session = Mock()
            session.request.return_value = FakeResponse({**counted(), **changes})
            with self.subTest(changes=changes), self.assertRaises(GeneratorMalformedResponseError):
                make_generator(session=session).count_prompt_tokens([{"role": "user", "content": "Q"}])
        session.request.return_value = FakeResponse({**counted(), "model": "different"})
        with self.assertRaises(GeneratorModelMismatchError):
            make_generator(session=session).count_prompt_tokens([{"role": "user", "content": "Q"}])

    def test_transport_errors_and_oom_use_e4_hierarchy(self):
        for source, expected in ((requests.ConnectionError("down"), GeneratorConnectionError),
                                 (requests.ConnectTimeout("wait"), GeneratorConnectTimeoutError),
                                 (requests.ReadTimeout("wait"), GeneratorReadTimeoutError)):
            session = Mock()
            session.request.side_effect = source
            with self.assertRaises(expected):
                make_generator(session=session).count_prompt_tokens([{"role": "user", "content": "Q"}])
        session = Mock()
        session.request.return_value = FakeResponse({}, status_code=500, text="CUDA out of memory")
        with self.assertRaises(GeneratorOutOfMemoryError):
            make_generator(session=session).count_prompt_tokens([{"role": "user", "content": "Q"}])


if __name__ == "__main__":
    unittest.main()
