"""E4 central LLM configuration validation tests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.config import ConfigurationError, load_config


class E4ConfigTests(unittest.TestCase):
    def test_current_verified_runtime_values_are_loaded(self) -> None:
        config = load_config()
        self.assertEqual(config.llm.provider, "local_openai_compatible")
        self.assertEqual(config.llm.runtime, "ollama")
        self.assertEqual(config.llm.runtime_version, "0.24.0")
        self.assertEqual(config.llm.model_id, "ministral-3:3b-instruct-2512-q8_0")
        self.assertEqual(config.llm.runtime_model_digest, "c269e5748d11")
        self.assertEqual(config.llm.base_url, "http://127.0.0.1:11434/v1")
        self.assertEqual(config.llm.context_length, 32768)
        self.assertEqual(config.llm.temperature, 0.0)
        self.assertEqual(config.llm.max_new_tokens, 512)
        self.assertEqual(config.llm.seed, 42)
        self.assertEqual(config.llm.quantization, "Q8_0")
        self.assertEqual(config.llm.connect_timeout_s, 10.0)
        self.assertEqual(config.llm.read_timeout_s, 120.0)
        self.assertTrue(config.llm.runtime_profile_locked)

    def test_invalid_llm_numeric_settings_fail_fast(self) -> None:
        cases = (
            ("LLM_CONTEXT_LENGTH", "0", "llm.context_length must be > 0"),
            ("LLM_TEMPERATURE", "-0.1", "llm.temperature must be finite and >= 0"),
            ("LLM_TEMPERATURE", "nan", "llm.temperature must be finite and >= 0"),
            ("LLM_MAX_NEW_TOKENS", "0", "llm.max_new_tokens must be > 0"),
            ("LLM_CONNECT_TIMEOUT_S", "0", "llm.connect_timeout_s must be finite and > 0"),
            ("LLM_READ_TIMEOUT_S", "inf", "llm.read_timeout_s must be finite and > 0"),
        )
        for name, value, message in cases:
            with self.subTest(name=name, value=value):
                with patch.dict(os.environ, {name: value}, clear=False):
                    with self.assertRaisesRegex(ConfigurationError, message):
                        load_config()

    def test_profile_can_change_when_explicitly_unlocked(self) -> None:
        overrides = {
            "LLM_RUNTIME_PROFILE_LOCKED": "false",
            "LLM_MODEL": "future-model:latest",
            "LLM_RUNTIME_MODEL_DIGEST": "abcdef123456",
            "LLM_CONTEXT_LENGTH": "16384",
            "LLM_TEMPERATURE": "0.2",
            "LLM_MAX_NEW_TOKENS": "256",
            "LLM_SEED": "7",
            "LLM_QUANTIZATION": "Q4_K_M",
            "LLM_CONNECT_TIMEOUT_S": "5",
        }
        with patch.dict(os.environ, overrides, clear=False):
            config = load_config()
        self.assertFalse(config.llm.runtime_profile_locked)
        self.assertEqual(config.llm.model_id, "future-model:latest")
        self.assertEqual(config.llm.runtime_model_digest, "abcdef123456")
        self.assertEqual(config.llm.context_length, 16384)
        self.assertEqual(config.llm.temperature, 0.2)
        self.assertEqual(config.llm.max_new_tokens, 256)
        self.assertEqual(config.llm.seed, 7)
        self.assertEqual(config.llm.quantization, "Q4_K_M")
        self.assertEqual(config.llm.connect_timeout_s, 5.0)

    def test_invalid_boolean_is_rejected_instead_of_silently_unlocking(self) -> None:
        with patch.dict(
            os.environ,
            {"LLM_RUNTIME_PROFILE_LOCKED": "TRUEE"},
            clear=False,
        ):
            with self.assertRaisesRegex(ConfigurationError, "Invalid boolean value"):
                load_config()


if __name__ == "__main__":
    unittest.main()
