"""Evaluate F01-F06 generation behavior (implemented in E6)."""

from __future__ import annotations

import argparse

from scripts._common import not_implemented


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        default="data/fixtures/atlas.jsonl",
    )
    parser.add_argument("--config", default="config.yaml")
    return parser


def main() -> int:
    build_parser().parse_args()
    return not_implemented("evaluate-generation")


if __name__ == "__main__":
    raise SystemExit(main())

