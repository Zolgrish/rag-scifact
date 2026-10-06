"""E4 local generator contract tests without live network access."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import requests

from app.generator import (
    GeneratorConfigurationError,
    GeneratorConnectionError,
    GeneratorConnectTimeoutError,
    GeneratorHTTPError,
    GeneratorInputError,
    GeneratorMalformedResponseError,
    GeneratorModelMismatchError,
    GeneratorModelUnavailableError,
    GeneratorOutOfMemoryError,
    GeneratorReadTimeoutError,
    OpenAICompatibleGenerator,
    RuntimeMetadata,
    build_generator,
    validate_runtime_metadata,
)


MODEL = "ministral-3:3b-instruct-2512-q8_0"
BASE_URL = "http://127.0.0.1:11434/v1"


class FakeResponse:
    def __init__(self, payload=None, *, status_code: int = 200, text: str = "") -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text
        self.json_error: Exception | None = None

    def json(self):
        if self.json_error is not None:
            raise self.json_error
        return self._payload


def make_generator(*, session=None, base_url: str = BASE_URL) -> OpenAICompatibleGenerator:
    return OpenAICompatibleGenerator(
        model_id=MODEL,
        base_url=base_url,
        runtime="ollama",
        temperature=0,
        max_new_tokens=512,
        seed=42,
        connect_timeout_s=10,
        read_timeout_s=120,
        session=session,
    )


class E4GeneratorTests(unittest.TestCase):
    def test_successful_generation_uses_locked_parameters_and_returns_usage(self) -> None:
        session = Mock()
        session.request.return_value = FakeResponse(
            {
                "model": MODEL,
                "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 1,
                    "total_tokens": 8,
                },
            }
        )
        generator = make_generator(session=session)

        result = generator.generate([{"role": "user", "content": "Say OK"}])

        self.assertEqual(result.text, "OK")
        self.assertEqual(result.model_id, MODEL)
        self.assertEqual(result.usage, {
            "prompt_tokens": 7,
            "completion_tokens": 1,
            "total_tokens": 8,
        })
        self.assertIsNotNone(result.generation_ms)
        call = session.request.call_args
        self.assertEqual(call.args, ("POST", f"{BASE_URL}/chat/completions"))
        self.assertEqual(call.kwargs["timeout"], (10.0, 120.0))
        self.assertEqual(
            call.kwargs["json"],
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Say OK"}],
                "temperature": 0.0,
                "max_tokens": 512,
                "seed": 42,
                "stream": False,
            },
        )

    def test_usage_is_optional(self) -> None:
        session = Mock()
        session.request.return_value = FakeResponse(
            {
                "model": MODEL,
                "choices": [{"message": {"content": "answer"}}],
            }
        )
        result = make_generator(session=session).generate(
            [{"role": "user", "content": "question"}]
        )
        self.assertIsNone(result.usage)

    def test_readiness_requires_configured_model_in_models_response(self) -> None:
        session = Mock()
        session.request.side_effect = [
            FakeResponse({"data": [{"id": MODEL}, {"id": "other-model"}]}),
            FakeResponse(
                {
                    "model": MODEL,
                    "choices": [{"message": {"content": "ready"}}],
                }
            ),
        ]
        readiness = make_generator(session=session).check_readiness()
        self.assertTrue(readiness.ready)
        self.assertEqual(readiness.model_id, MODEL)
        self.assertEqual(readiness.runtime, "ollama")
        self.assertEqual(session.request.call_count, 2)
        first = session.request.call_args_list[0]
        second = session.request.call_args_list[1]
        self.assertEqual(first.args, ("GET", f"{BASE_URL}/models"))
        self.assertEqual(first.kwargs["timeout"], (10.0, 120.0))
        self.assertEqual(second.args, ("POST", f"{BASE_URL}/chat/completions"))
        self.assertEqual(second.kwargs["json"]["model"], MODEL)
        self.assertEqual(second.kwargs["json"]["temperature"], 0.0)
        self.assertEqual(second.kwargs["json"]["max_tokens"], 8)
        self.assertEqual(second.kwargs["json"]["seed"], 42)

    def test_readiness_fails_if_listed_model_cannot_generate(self) -> None:
        session = Mock()
        session.request.side_effect = [
            FakeResponse({"data": [{"id": MODEL}]}),
            FakeResponse(
                {},
                status_code=500,
                text="CUDA out of memory while loading model",
            ),
        ]
        with self.assertRaises(GeneratorOutOfMemoryError):
            make_generator(session=session).check_readiness()

    def test_readiness_fails_on_mismatched_probe_model(self) -> None:
        session = Mock()
        session.request.side_effect = [
            FakeResponse({"data": [{"id": MODEL}]}),
            FakeResponse(
                {
                    "model": "different-model",
                    "choices": [{"message": {"content": "ready"}}],
                }
            ),
        ]
        with self.assertRaises(GeneratorModelMismatchError):
            make_generator(session=session).check_readiness()

    def test_readiness_rejects_missing_model(self) -> None:
        session = Mock()
        session.request.return_value = FakeResponse({"data": [{"id": "other"}]})
        with self.assertRaises(GeneratorModelUnavailableError):
            make_generator(session=session).check_readiness()

    def test_readiness_rejects_malformed_models_payload(self) -> None:
        session = Mock()
        session.request.return_value = FakeResponse({"data": "not-an-array"})
        with self.assertRaises(GeneratorMalformedResponseError):
            make_generator(session=session).check_readiness()

    def test_loopback_only_base_url_contract(self) -> None:
        self.assertEqual(
            make_generator(base_url="http://localhost:11434/v1/").base_url,
            "http://localhost:11434/v1",
        )
        self.assertEqual(
            make_generator(base_url="http://[::1]:11434/v1").base_url,
            "http://[::1]:11434/v1",
        )
        for value in (
            "",
            "https://api.openai.com/v1",
            "http://192.168.1.10:11434/v1",
            "ftp://127.0.0.1:11434/v1",
            "http://user:pass@127.0.0.1:11434/v1",
            "http://127.0.0.1:11434/other",
        ):
            with self.subTest(value=value):
                with self.assertRaises(GeneratorConfigurationError):
                    make_generator(base_url=value)

    def test_ollama_runtime_metadata_is_observed_from_native_local_apis(self) -> None:
        session = Mock()
        full_digest = "c269e5748d11" + "a" * 52
        session.request.side_effect = [
            FakeResponse({"version": "0.24.0"}),
            FakeResponse(
                {
                    "models": [
                        {
                            "name": MODEL,
                            "model": MODEL,
                            "digest": full_digest,
                            "details": {"quantization_level": "Q8_0"},
                        }
                    ]
                }
            ),
            FakeResponse(
                {
                    "models": [
                        {
                            "name": MODEL,
                            "model": MODEL,
                            "digest": full_digest,
                            "context_length": 32768,
                            "size_vram": 1234,
                        }
                    ]
                }
            ),
        ]
        metadata = make_generator(session=session).inspect_runtime_metadata()
        self.assertEqual(
            metadata,
            RuntimeMetadata(
                runtime="ollama",
                runtime_version="0.24.0",
                model_id=MODEL,
                runtime_model_digest=full_digest,
                quantization="Q8_0",
                context_length=32768,
                size_vram_bytes=1234,
            ),
        )
        urls = [call.args[1] for call in session.request.call_args_list]
        self.assertEqual(
            urls,
            [
                "http://127.0.0.1:11434/api/version",
                "http://127.0.0.1:11434/api/tags",
                "http://127.0.0.1:11434/api/ps",
            ],
        )

    def test_runtime_metadata_validation_accepts_digest_prefix_and_rejects_drift(self) -> None:
        config = SimpleNamespace(
            runtime="ollama",
            runtime_version="0.24.0",
            model_id=MODEL,
            runtime_model_digest="c269e5748d11",
            quantization="Q8_0",
            context_length=32768,
        )
        observed = RuntimeMetadata(
            runtime="ollama",
            runtime_version="0.24.0",
            model_id=MODEL,
            runtime_model_digest="c269e5748d11" + "a" * 52,
            quantization="Q8_0",
            context_length=32768,
        )
        validate_runtime_metadata(config, observed)
        with self.assertRaisesRegex(GeneratorConfigurationError, "quantization mismatch"):
            validate_runtime_metadata(
                config,
                RuntimeMetadata(
                    runtime="ollama",
                    runtime_version="0.24.0",
                    model_id=MODEL,
                    runtime_model_digest=observed.runtime_model_digest,
                    quantization="Q4_K_M",
                    context_length=32768,
                ),
            )

    def test_direct_constructor_rejects_nonfinite_runtime_numbers(self) -> None:
        cases = (
            {"temperature": float("nan")},
            {"temperature": float("inf")},
            {"connect_timeout_s": float("inf")},
            {"read_timeout_s": float("nan")},
        )
        for override in cases:
            with self.subTest(override=override):
                kwargs = {
                    "model_id": MODEL,
                    "base_url": BASE_URL,
                    "runtime": "ollama",
                    "temperature": 0.0,
                    "max_new_tokens": 512,
                    "seed": 42,
                    "connect_timeout_s": 10.0,
                    "read_timeout_s": 120.0,
                }
                kwargs.update(override)
                with self.assertRaises(GeneratorConfigurationError):
                    OpenAICompatibleGenerator(**kwargs)

    def test_input_validation_happens_before_transport(self) -> None:
        session = Mock()
        generator = make_generator(session=session)
        bad_inputs = (
            [],
            "hello",
            [{}],
            [{"role": "", "content": "x"}],
            [{"role": "user", "content": 3}],
        )
        for value in bad_inputs:
            with self.subTest(value=value):
                with self.assertRaises(GeneratorInputError):
                    generator.generate(value)  # type: ignore[arg-type]
        session.request.assert_not_called()

    def test_connection_refused_is_explicit(self) -> None:
        session = Mock()
        session.request.side_effect = requests.exceptions.ConnectionError("refused")
        with self.assertRaises(GeneratorConnectionError):
            make_generator(session=session).generate(
                [{"role": "user", "content": "x"}]
            )

    def test_connect_timeout_is_distinct(self) -> None:
        session = Mock()
        session.request.side_effect = requests.exceptions.ConnectTimeout("slow connect")
        with self.assertRaises(GeneratorConnectTimeoutError):
            make_generator(session=session).check_readiness()

    def test_read_timeout_is_distinct(self) -> None:
        session = Mock()
        session.request.side_effect = requests.exceptions.ReadTimeout("slow generation")
        with self.assertRaises(GeneratorReadTimeoutError):
            make_generator(session=session).generate(
                [{"role": "user", "content": "x"}]
            )

    def test_model_unavailable_http_error_is_explicit(self) -> None:
        session = Mock()
        session.request.return_value = FakeResponse(
            {"error": "model not found"},
            status_code=404,
            text='{"error":"model not found; pull it first"}',
        )
        with self.assertRaises(GeneratorModelUnavailableError) as ctx:
            make_generator(session=session).generate(
                [{"role": "user", "content": "x"}]
            )
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.to_dict()["code"], "generator_model_unavailable")

    def test_ambiguous_404_stays_generic_http_error(self) -> None:
        session = Mock()
        session.request.return_value = FakeResponse(
            {}, status_code=404, text="route not found"
        )
        with self.assertRaises(GeneratorHTTPError):
            make_generator(session=session).check_readiness()

    def test_oom_http_error_is_explicit(self) -> None:
        session = Mock()
        session.request.return_value = FakeResponse(
            {}, status_code=500, text="CUDA out of memory while allocating tensor"
        )
        with self.assertRaises(GeneratorOutOfMemoryError) as ctx:
            make_generator(session=session).generate(
                [{"role": "user", "content": "x"}]
            )
        self.assertEqual(ctx.exception.status_code, 500)

    def test_generic_server_error_is_not_reclassified(self) -> None:
        session = Mock()
        session.request.return_value = FakeResponse(
            {}, status_code=503, text="service temporarily unavailable"
        )
        with self.assertRaises(GeneratorHTTPError) as ctx:
            make_generator(session=session).check_readiness()
        self.assertEqual(ctx.exception.status_code, 503)

    def test_non_json_success_is_malformed_response(self) -> None:
        session = Mock()
        response = FakeResponse(None)
        response.json_error = ValueError("bad json")
        session.request.return_value = response
        with self.assertRaises(GeneratorMalformedResponseError):
            make_generator(session=session).check_readiness()

    def test_malformed_completion_shapes_are_rejected(self) -> None:
        payloads = (
            {},
            {"model": MODEL, "choices": []},
            {"model": MODEL, "choices": [{}]},
            {"model": MODEL, "choices": [{"message": {}}]},
            {"model": MODEL, "choices": [{"message": {"content": "   "}}]},
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                session = Mock()
                session.request.return_value = FakeResponse(payload)
                with self.assertRaises(GeneratorMalformedResponseError):
                    make_generator(session=session).generate(
                        [{"role": "user", "content": "x"}]
                    )

    def test_model_mismatch_prevents_hidden_runtime_fallback(self) -> None:
        session = Mock()
        session.request.return_value = FakeResponse(
            {
                "model": "different-model",
                "choices": [{"message": {"content": "OK"}}],
            }
        )
        with self.assertRaises(GeneratorModelMismatchError):
            make_generator(session=session).generate(
                [{"role": "user", "content": "x"}]
            )

    def test_malformed_usage_is_rejected(self) -> None:
        session = Mock()
        session.request.return_value = FakeResponse(
            {
                "model": MODEL,
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"total_tokens": True},
            }
        )
        with self.assertRaises(GeneratorMalformedResponseError):
            make_generator(session=session).generate(
                [{"role": "user", "content": "x"}]
            )

    def test_build_generator_rejects_unsupported_provider_or_runtime(self) -> None:
        base = dict(
            provider="local_openai_compatible",
            runtime="ollama",
            model_id=MODEL,
            base_url=BASE_URL,
            temperature=0.0,
            max_new_tokens=512,
            seed=42,
            connect_timeout_s=10.0,
            read_timeout_s=120.0,
        )
        with self.assertRaises(GeneratorConfigurationError):
            build_generator(SimpleNamespace(**{**base, "provider": "openai"}))
        with self.assertRaises(GeneratorConfigurationError):
            build_generator(SimpleNamespace(**{**base, "runtime": "vllm"}))
        generator = build_generator(SimpleNamespace(**base), session=Mock())
        self.assertEqual(generator.model_id, MODEL)
        self.assertEqual(generator.base_url, BASE_URL)


if __name__ == "__main__":
    unittest.main()
