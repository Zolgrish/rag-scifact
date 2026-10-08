"""Explicitly seal a clean canonical baseline BEFORE any final-test evaluation."""
from __future__ import annotations

import argparse
import json
import sys

from app.benchmark_manifest import artifact_record
from app.manifest import (current_git_state, freeze_final_test_manifest, load_manifest, write_manifest_atomic)
from app.retrieval_evaluator import RetrievalEvaluationError
from scripts._common import bootstrap


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    logger = None
    try:
        config, logger, run_id = bootstrap(args.config)
        path = config.paths.artifact_dir / "manifest.json"
        before = artifact_record(path)
        sealed = freeze_final_test_manifest(load_manifest(path), config=config)
        if (artifact_record(path) != before
                or current_git_state() != (sealed["freeze"]["frozen_git_commit"], False)):
            raise RetrievalEvaluationError("Manifest/source changed before freeze publication")
        json.dumps(sealed, allow_nan=False)
        write_manifest_atomic(path, sealed)
        logger.info("Final-test baseline sealed run_id=%s source=%s", run_id, sealed["freeze"]["frozen_git_commit"])
        print(json.dumps({"run_id": run_id, "freeze": sealed["freeze"]}, allow_nan=False))
        return 0
    except Exception as exc:
        if logger:
            logger.exception("Final-test freeze rejected")
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
