"""Audit SciFact and write deterministic E1 artifacts/manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from app.config import REPO_ROOT
from app.data_audit import audit_scifact_dataset
from app.loader import DatasetError
from app.manifest import (
    ManifestError,
    load_manifest,
    merge_e1_manifest,
    write_manifest_atomic,
)
from scripts._common import ConfigurationError, bootstrap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--manifest",
        default="artifacts/manifest.json",
        help="Canonical manifest path; relative paths resolve from the repository root.",
    )
    parser.add_argument(
        "--output-dir",
        help="Per-run artifact directory; defaults to artifacts/<run_id>.",
    )
    return parser


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = build_parser().parse_args()
    try:
        config, logger, run_id = bootstrap(args.config)
        logger.info("E1 SciFact dataset audit started")

        _dataset, split, audit = audit_scifact_dataset(config)

        manifest_path = _repo_path(args.manifest)
        existing = load_manifest(manifest_path)
        merged = merge_e1_manifest(
            existing,
            config=config,
            split=split,
            audit=audit,
        )
        write_manifest_atomic(manifest_path, merged)

        output_dir = (
            _repo_path(args.output_dir)
            if args.output_dir
            else config.paths.artifact_dir / run_id
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        audit_path = output_dir / "dataset_audit.json"
        manifest_snapshot = output_dir / "manifest.json"
        _write_json(audit_path, audit)
        _write_json(manifest_snapshot, merged)

        config_path = _repo_path(args.config)
        if config_path.is_file():
            shutil.copy2(config_path, output_dir / "config.snapshot.yaml")

        counts = audit["counts"]
        overlap = audit["query_overlap_audit"]
        logger.info(
            "E1 audit PASS corpus=%s queries=%s train_ids=%s test_ids=%s "
            "dev=%s practice=%s",
            counts["corpus_count"],
            counts["query_count"],
            counts["train_query_count"],
            counts["test_query_count"],
            len(split.dev_ids),
            len(split.practice_ids),
        )
        logger.info(
            "query-overlap raw_exact=%s canonical_exact=%s near=%s",
            overlap["totals"]["raw_exact_duplicate_pairs"],
            overlap["totals"]["canonical_exact_duplicate_pairs"],
            overlap["totals"]["near_duplicate_pairs"],
        )
        logger.info("canonical manifest=%s", manifest_path)
        logger.info("run artifacts=%s", output_dir)

        print(
            json.dumps(
                {
                    "status": "PASS",
                    "run_id": run_id,
                    "dataset_audit": str(audit_path),
                    "manifest": str(manifest_path),
                    "manifest_snapshot": str(manifest_snapshot),
                    "counts": counts,
                    "split": audit["split"],
                    "query_overlap_audit": overlap,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    except (ConfigurationError, DatasetError, ManifestError):
        if "logger" in locals():
            logger.exception("E1 dataset audit failed")
        else:
            import traceback

            traceback.print_exc()
        return 2
    except Exception:
        if "logger" in locals():
            logger.exception("Unexpected E1 dataset audit failure")
        else:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
