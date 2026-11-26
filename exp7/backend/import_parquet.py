"""把 HF 上的 CASIA-WebFace parquet 分片导成 <root>/<身份>/<图片>.jpg。

用法：
    python -m backend.import_parquet --src data/casia/shard0.parquet --dry-run
    python -m backend.import_parquet --src data/casia/shard0.parquet \
        --out data/casia/images --max-ids 500 --per-id 12 --min-per-id 8

设计要点
--------
- 两遍扫描：先只读 label 列统计每身份图片数（很快），再按需分块读图像列，
  避免把整片 466MB parquet 的图片一次性读进内存（本机可用内存约 5GB）。
- 图像列可能是 HF Image 的 struct{bytes,path}，也可能是裸 binary，两种都兼容。
- 原图是 jpg 就直接写字节（不重编码，避免二次压缩损失）。
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

try:
    from .paths import resolve
except ImportError:  # 允许 python backend/import_parquet.py 直接运行
    from paths import resolve


def _import_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "需要 pyarrow 读取 parquet：\n"
            "  pip install pyarrow\n"
            "（本机 Python 3.14 有 cp314 轮子，pyarrow 26.0.0）"
        ) from exc
    return pq


def _pick_columns(schema) -> tuple:
    names = list(schema.names)
    label_col = None
    for cand in ("label", "person", "identity", "id", "class"):
        if cand in names:
            label_col = cand
            break
    if label_col is None:
        raise SystemExit(f"找不到标签列，现有列: {names}")

    img_col = None
    for field in schema:
        if field.name == label_col:
            continue
        if str(field.type).startswith("struct") or str(field.type) in ("binary", "large_binary"):
            img_col = field.name
            break
    if img_col is None:
        raise SystemExit(f"找不到图像列，现有列: {names}")
    return img_col, label_col


def _image_bytes(value):
    if isinstance(value, dict):
        data = value.get("bytes")
        if data is not None:
            return data
        path = value.get("path")
        return Path(path).read_bytes() if path and Path(path).is_file() else None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="parquet -> 身份目录")
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-ids", type=int, default=0, help="0=全部")
    parser.add_argument("--per-id", type=int, default=12)
    parser.add_argument("--min-per-id", type=int, default=8)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true", help="只打印 schema 与统计")
    args = parser.parse_args()

    pq = _import_pyarrow()
    src = resolve(args.src)
    if not src.is_file():
        raise SystemExit(f"文件不存在: {src}")

    pf = pq.ParquetFile(src)
    img_col, label_col = _pick_columns(pf.schema_arrow)
    print(f"文件: {src.name}  行数: {pf.metadata.num_rows}  行组: {pf.num_row_groups}")
    print(f"列: image={img_col!r}  label={label_col!r}")
    schema = pf.schema_arrow
    print("字段:", [(f.name, str(f.type)) for f in schema])

    labels = pf.read(columns=[label_col]).column(0).to_pylist()
    counts = collections.Counter(labels)
    print(f"身份数: {len(counts)}  图片数: {len(labels)}  "
          f"每人中位数: {sorted(counts.values())[len(counts)//2]}  "
          f"最多: {counts.most_common(3)}")
    if args.dry_run:
        return

    eligible = [pid for pid, cnt in counts.items() if cnt >= args.min_per_id]
    eligible.sort(key=lambda pid: (-counts[pid], str(pid)))
    if args.max_ids > 0:
        eligible = eligible[: args.max_ids]
    if not eligible:
        raise SystemExit(f"没有身份满足每人 >= {args.min_per_id} 张")
    keep = set(eligible)

    out = resolve(args.out) if args.out else src.parent / "images"
    out.mkdir(parents=True, exist_ok=True)
    written: dict = collections.Counter()
    offsets: dict = {}
    skipped = 0

    for batch in pf.iter_batches(batch_size=args.batch, columns=[img_col, label_col]):
        images = batch.column(0).to_pylist()
        batch_labels = batch.column(1).to_pylist()
        for value, pid in zip(images, batch_labels):
            if pid not in keep or written[pid] >= args.per_id:
                continue
            data = _image_bytes(value)
            if not data:
                skipped += 1
                continue
            ident_dir = out / str(pid)
            ident_dir.mkdir(parents=True, exist_ok=True)
            if pid not in offsets:
                # 多个分片导出到同一目录时接着编号，避免覆盖
                offsets[pid] = sum(
                    1 for p in ident_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                )
            ext = ".jpg" if data[:2] == b"\xff\xd8" else ".png"
            index = offsets[pid] + written[pid]
            (ident_dir / f"{pid}_{index:04d}{ext}").write_bytes(data)
            written[pid] += 1
        if len(written) % 50 == 0 and written:
            print(f"  已写 {sum(written.values())} 张 / {len(written)} 个身份")

    total = sum(written.values())
    short = [pid for pid in keep if written[pid] < args.min_per_id]
    print(f"\n导出完成: {total} 张 / {len(written)} 个身份 -> {out}")
    print(f"每人写入: 最少 {min(written.values()) if written else 0}  "
          f"最多 {max(written.values()) if written else 0}")
    if short:
        print(f"警告: {len(short)} 个身份写入不足 {args.min_per_id} 张（原图缺失/损坏），"
              "协议会自动过滤掉它们")
    if skipped:
        print(f"跳过无法解码的图像: {skipped}")


if __name__ == "__main__":
    main()
