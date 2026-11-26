from __future__ import annotations

import json
import os
import platform
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

EXP8_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = EXP8_DIR / "chnsenticorp"
DEFAULT_OUTPUT_DIR = EXP8_DIR / "outputs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用：检查镜像/驱动，或改用 --device cpu")
    return torch.device(name)


def configure_torch(device: torch.device) -> None:
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 8)))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("high")


def amp_dtype(name: str) -> torch.dtype | None:
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(name)


def env_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
        "command": " ".join(sys.argv),
        "device": str(device),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:
        info["transformers"] = None
    return info


def make_run_dir(output_dir: Path, model_name: str, seed: int, tag: str = "") -> Path:
    safe = model_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    parts = [safe, f"s{seed}"]
    if tag:
        parts.append(tag)
    parts.append(time.strftime("%Y%m%d-%H%M%S"))
    run_dir = Path(output_dir) / "runs" / "_".join(parts)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self.start