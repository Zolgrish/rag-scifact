"""Offline smoke test for E0 logging."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from app.logging_utils import configure_logging


class E0LoggingTests(unittest.TestCase):
    def test_logging_creates_run_file_with_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger, run_id = configure_logging(
                Path(temp_dir),
                level="INFO",
                filename_prefix="test",
                run_id="e0-smoke",
            )
            logger.info("smoke")

            self.assertEqual(run_id, "e0-smoke")
            log_file = Path(temp_dir) / "test-e0-smoke.log"
            self.assertTrue(log_file.is_file())
            content = log_file.read_text(encoding="utf-8")
            self.assertIn("run_id=e0-smoke", content)
            self.assertIn("smoke", content)

            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)



if __name__ == "__main__":
    unittest.main()
