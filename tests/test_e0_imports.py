"""E0 smoke tests that require only the Python standard library."""

from __future__ import annotations

import importlib
import unittest


CORE_MODULES = (
    "app",
    "app.config",
    "app.logging_utils",
    "app.models",
    "app.loader",
    "app.chunker",
    "app.embedder",
    "app.indexer",
    "app.retriever",
    "app.context",
    "app.prompt",
    "app.generator",
    "app.citations",
    "app.rag",
    "app.evaluator",
    "scripts.build_index",
    "scripts.retrieve",
    "scripts.ask",
    "scripts.evaluate",
    "scripts.evaluate_generation",
    "scripts.check_environment",
)


class E0ImportTests(unittest.TestCase):
    def test_core_modules_import_without_third_party_dependencies(self) -> None:
        for module_name in CORE_MODULES:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()

