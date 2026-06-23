"""Download model snapshots after dependency and disk checks."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


MODEL_REPOS = {
    "llava15_7b_4bit": "llava-hf/llava-1.5-7b-hf",
    "minicpmv26_2b": "openbmb/MiniCPM-V-2_6",
    "qwen2vl_2b": "Qwen/Qwen2-VL-2B-Instruct",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_REPOS.keys(), required=True)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--min-free-gb", type=float, default=30.0)
    args = parser.parse_args()

    free_gb = shutil.disk_usage(Path.cwd()).free / 1024**3
    if free_gb < args.min_free_gb:
        raise RuntimeError(f"Only {free_gb:.2f} GB free; refusing to download.")
    project_root = Path.cwd()
    os.environ.setdefault("HF_HOME", str(project_root / ".hf-cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(project_root / ".hf-cache" / "transformers"))
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub/transformers dependencies first.") from exc

    repo = MODEL_REPOS[args.model]
    local_dir = Path(args.models_dir) / args.model
    local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(repo_id=repo, local_dir=str(local_dir), local_dir_use_symlinks=False)
    print(path)


if __name__ == "__main__":
    main()
