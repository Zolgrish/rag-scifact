"""Evaluate SciFact retrieval metrics (implemented in E7)."""

from __future__ import annotations

import argparse

from scripts._common import not_implemented


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--config", default="config.yaml")
    return parser


def main() -> int:
    build_parser().parse_args()
    return not_implemented("evaluate")


if __name__ == "__main__":
    raise SystemExit(main())

