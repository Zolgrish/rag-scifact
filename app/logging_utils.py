"""Project logging setup."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
import time
import uuid


def new_run_id() -> str:
    """Return a compact unique run identifier."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def configure_logging(
    log_dir: Path,
    *,
    level: str = "INFO",
    filename_prefix: str = "rag-scifact",
    run_id: str | None = None,
) -> tuple[logging.Logger, str]:
    """Configure console + UTF-8 file logging and return logger/run_id."""

    resolved_run_id = run_id or new_run_id()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{filename_prefix}-{resolved_run_id}.log"

    logger = logging.getLogger("rag_scifact")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)sZ %(levelname)s run_id=%(run_id)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime

    class RunIdFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.run_id = resolved_run_id
            return True

    run_filter = RunIdFilter()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    console.addFilter(run_filter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(run_filter)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.info("logging initialized; file=%s", log_file)

    return logger, resolved_run_id
