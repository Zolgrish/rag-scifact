"""E0 config smoke tests that work before and after dependency installation."""

from __future__ import annotations

import importlib.util
import unittest

from app.config import ConfigurationError, load_config


class E0ConfigTests(unittest.TestCase):
    def test_config_dependency_state_is_explicit(self) -> None:
        if importlib.util.find_spec("yaml") is None:
            with self.assertRaisesRegex(
                ConfigurationError,
                "PyYAML is not installed",
            ):
                load_config()
            return

        config = load_config()
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.embedding.chunk_size_tokens, 220)
        self.assertEqual(config.embedding.chunk_overlap_tokens, 30)
        self.assertEqual(config.retrieval.top_k, 5)
        self.assertEqual(config.llm.max_new_tokens, 512)
        self.assertEqual(config.llm.runtime, "ollama")
        self.assertEqual(config.llm.runtime_version, "0.24.0")
        self.assertEqual(config.llm.model_id, "ministral-3:3b-instruct-2512-q8_0")
        self.assertEqual(config.llm.runtime_model_digest, "c269e5748d11")
        self.assertEqual(config.llm.base_url, "http://127.0.0.1:11434/v1")
        self.assertEqual(config.llm.context_length, 32768)
        self.assertEqual(config.llm.quantization, "Q8_0")
        self.assertFalse(config.llm.runtime_profile_locked)


if __name__ == "__main__":
    unittest.main()

