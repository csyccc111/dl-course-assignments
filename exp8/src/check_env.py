from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .common import DEFAULT_DATA_DIR, env_info, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="检查运行环境、磁盘与数据文件")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    device = resolve_device(args.device)
    info = env_info(device)
    info["data_dir"] = str(args.data_dir)
    info["data_files"] = {}
    for name in ("train.tsv", "dev.tsv", "test.tsv"):
        path = Path(args.data_dir) / name
        info["data_files"][name] = path.exists()
    info["disk_free_gb"] = round(shutil.disk_usage(Path.cwd()).free / 1024**3, 1)
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()