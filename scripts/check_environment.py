"""Offline environment inspection.

This script never downloads packages or models. Run it to inspect the current or freshly reproduced benchmark environment.
"""

from __future__ import annotations

import importlib.util
import json
import platform
from pathlib import Path
import sys


PACKAGE_MODULES = {
    "torch": "torch",
    "transformers": "transformers",
    "sentence-transformers": "sentence_transformers",
    "faiss-cpu": "faiss",
    "numpy": "numpy",
    "pandas": "pandas",
    "PyYAML": "yaml",
    "tqdm": "tqdm",
    "requests": "requests",
    "openai": "openai",
    "huggingface-hub": "huggingface_hub",
    "python-dotenv": "dotenv",
}


def _module_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _torch_gpu_info() -> dict[str, object]:
    if not _module_available("torch"):
        return {"torch_installed": False}

    import torch

    result: dict[str, object] = {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        result["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        result["gpu_total_vram_gb"] = round(
            props.total_memory / (1024**3),
            2,
        )
    return result


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    payload = {
        "repo_root": str(repo_root),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            package: _module_available(module)
            for package, module in PACKAGE_MODULES.items()
        },
        "gpu": _torch_gpu_info(),
        "dataset_present": {
            "corpus": (repo_root / "data/scifact/corpus.jsonl").is_file(),
            "queries": (repo_root / "data/scifact/queries.jsonl").is_file(),
            "qrels_train": (
                repo_root / "data/scifact/qrels/train.tsv"
            ).is_file(),
            "qrels_test": (
                repo_root / "data/scifact/qrels/test.tsv"
            ).is_file(),
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

