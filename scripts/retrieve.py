"""Retrieve Top-K SciFact documents (implemented in E3)."""

from __future__ import annotations

import argparse

from scripts._common import not_implemented


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query")
    source.add_argument("--query-id")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--config", default="config.yaml")
    return parser


def main() -> int:
    build_parser().parse_args()
    return not_implemented("retrieve")


if __name__ == "__main__":
    raise SystemExit(main())

