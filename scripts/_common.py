"""Shared helpers for scaffolded CLI commands."""

from __future__ import annotations

from pathlib import Path

from app.config import ConfigurationError, ensure_output_directories, load_config
from app.logging_utils import configure_logging


def bootstrap(config_path: str | Path):
    """Load config, ensure outputs, and initialize run logging."""

    config = load_config(config_path)
    ensure_output_directories(config)
    logger, run_id = configure_logging(
        config.paths.log_dir,
        level=config.logging.level,
        filename_prefix=config.logging.filename_prefix,
    )
    return config, logger, run_id


def not_implemented(stage: str) -> int:
    print(
        f"{stage} is scaffolded but not implemented yet. "
        "Complete the owning epic in specs/ before running it."
    )
    return 2


__all__ = [
    "ConfigurationError",
    "bootstrap",
    "not_implemented",
]

