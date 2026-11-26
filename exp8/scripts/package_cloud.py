from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

EXP8_DIR = Path(__file__).resolve().parents[1]
INCLUDE = ["src", "scripts", "chnsenticorp", "requirements-cloud.txt", "README-cloud.md"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def iter_files(root: Path):
    if root.is_file():
        yield root
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="打包云端运行所需文件")
    parser.add_argument("--output", type=Path, default=EXP8_DIR / "exp8_cloud.zip")
    args = parser.parse_args()

    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in INCLUDE:
            source = EXP8_DIR / name
            if not source.exists():
                raise SystemExit(f"缺少打包文件: {source}")
            for path in iter_files(source):
                archive.write(path, path.relative_to(EXP8_DIR).as_posix())

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    size_mb = args.output.stat().st_size / 1024**2
    print(f"[package] {args.output} ({size_mb:.2f} MB)")
    print(f"[package] sha256={digest}")


if __name__ == "__main__":
    main()