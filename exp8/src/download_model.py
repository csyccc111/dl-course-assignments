from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="通过 hf-mirror 预下载模型/分词器")
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    print(f"[hf] endpoint={os.environ.get('HF_ENDPOINT', 'default')}", flush=True)
    for repo_id in args.model:
        path = snapshot_download(repo_id=repo_id, cache_dir=str(args.cache_dir) if args.cache_dir else None)
        print(f"[hf] {repo_id} -> {path}", flush=True)


if __name__ == "__main__":
    main()