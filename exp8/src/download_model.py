from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

IGNORE_PATTERNS = ["*.h5", "*.msgpack", "*.ot", "*.onnx", "*.ckpt"]


def main() -> None:
    parser = argparse.ArgumentParser(description="通过 hf-mirror 预下载模型/分词器（只取 PyTorch 需要的文件）")
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    print(f"[hf] endpoint={os.environ.get('HF_ENDPOINT', 'default')}", flush=True)
    for repo_id in args.model:
        path = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            ignore_patterns=IGNORE_PATTERNS,
        )
        print(f"[hf] {repo_id} -> {path}", flush=True)


if __name__ == "__main__":
    main()