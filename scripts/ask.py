"""Run end-to-end RAG answering (implemented in E5)."""

from __future__ import annotations

import argparse

from scripts._common import not_implemented


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--config", default="config.yaml")
    return parser


def main() -> int:
    build_parser().parse_args()
    return not_implemented("ask")


if __name__ == "__main__":
    raise SystemExit(main())

