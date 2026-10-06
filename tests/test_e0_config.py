"""E0 config smoke tests that work before and after dependency installation."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.config import ConfigurationError, REPO_ROOT, load_config


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
        self.assertEqual(config.retrieval.mode, "dense")
        self.assertEqual(config.retrieval.max_top_k, 10)


    def test_retrieval_top_k_env_override_and_max_contract(self) -> None:
        if importlib.util.find_spec("yaml") is None:
            self.skipTest("PyYAML is not installed")

        with patch.dict(os.environ, {"TOP_K": "7"}, clear=False):
            config = load_config()
        self.assertEqual(config.retrieval.top_k, 7)
        self.assertEqual(config.retrieval.max_top_k, 10)

        source = (REPO_ROOT / "config.yaml").read_text(encoding="utf-8")
        self.assertIn("max_top_k: 10", source)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                source.replace("max_top_k: 10", "max_top_k: 11", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ConfigurationError, "retrieval.max_top_k must be in range 1..10"
            ):
                load_config(path)


if __name__ == "__main__":
    unittest.main()

