"""Verify the local CalibPrune experiment environment."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import shutil
import sys
from pathlib import Path


CORE_MODULES = ["torch", "numpy", "pandas", "sklearn", "matplotlib", "PIL", "yaml"]
FULL_MODULES = [
    "transformers",
    "accelerate",
    "bitsandbytes",
    "datasets",
    "evaluate",
    "netcal",
    "torchmetrics",
    "hydra",
]


def version_for(module_name: str) -> str:
    package_name = {
        "PIL": "Pillow",
        "sklearn": "scikit-learn",
        "yaml": "PyYAML",
        "hydra": "hydra-core",
    }.get(module_name, module_name)
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def check_imports(modules: list[str]) -> list[str]:
    missing = []
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"{module}: OK ({version_for(module)})")
        except Exception as exc:
            print(f"{module}: MISSING ({exc.__class__.__name__}: {exc})")
            missing.append(module)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--core-only", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    os.environ.setdefault("HF_HOME", str(project_root / ".hf-cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(project_root / ".hf-cache" / "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(project_root / "data" / "hf_cache"))
    os.environ.setdefault("TORCH_HOME", str(project_root / ".torch-cache"))

    print(f"python: {sys.version.split()[0]} at {sys.executable}")
    total, used, free = shutil.disk_usage(project_root)
    print(f"disk_free_gb: {free / 1024**3:.2f}")
    print(f"disk_total_gb: {total / 1024**3:.2f}")

    missing = check_imports(CORE_MODULES)
    if not args.core_only:
        missing.extend(check_imports(FULL_MODULES))

    try:
        import torch

        print(f"torch_version: {torch.__version__}")
        print(f"torch_cuda_version: {torch.version.cuda}")
        cuda_ok = bool(torch.cuda.is_available())
        print(f"cuda_available: {cuda_ok}")
        if cuda_ok:
            print(f"gpu_name: {torch.cuda.get_device_name(0)}")
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            print(f"free_vram_gb: {free_bytes / 1024**3:.2f}")
            print(f"total_vram_gb: {total_bytes / 1024**3:.2f}")
        else:
            missing.append("cuda")
    except Exception as exc:
        print(f"torch_cuda_check: FAILED ({exc.__class__.__name__}: {exc})")
        missing.append("torch_cuda_check")

    if not args.core_only and "bitsandbytes" not in missing and "cuda" not in missing:
        try:
            import bitsandbytes as bnb
            import torch

            layer = bnb.nn.Linear4bit(64, 2, quant_type="nf4").cuda()
            x = torch.randn(1, 64, device="cuda")
            y = layer(x)
            print(f"bitsandbytes_4bit_cuda: {bool(y.is_cuda)}")
            if not y.is_cuda:
                missing.append("bitsandbytes_4bit_cuda")
        except Exception as exc:
            print(f"bitsandbytes_4bit_cuda: FAILED ({exc.__class__.__name__}: {exc})")
            missing.append("bitsandbytes_4bit_cuda")

    if free / 1024**3 < 10:
        print("disk_check: FAILED (<10 GB free)")
        missing.append("disk_free")
    if missing:
        print("CHECKS FAILED")
        print("missing_or_failed: " + ", ".join(sorted(set(missing))))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
