"""把 AI Studio 版 LFW 的 pairs.csv 还原成官方 pairs.txt 格式。

官方 pairs.txt：首行折数 10，随后 6000 行
    name i j            -> 同一人
    name1 i name2 j     -> 不同人
每折 300 同 + 300 异。AI Studio 的 pairs.csv 保留了官方顺序，只是行尾多一个空列，
按 600 行一组切分即可还原官方 10 折；脚本会校验每折的构成。
"""

from __future__ import annotations

import argparse
import csv

try:
    from .paths import resolve
except ImportError:  # 允许 python backend/lfw_convert.py 直接运行
    from paths import resolve


def main() -> None:
    parser = argparse.ArgumentParser(description="pairs.csv -> pairs.txt（官方 10 折格式）")
    parser.add_argument("--csv", default="data/lfw/pairs.csv")
    parser.add_argument("--out", default="data/lfw/pairs.txt")
    parser.add_argument("--folds", type=int, default=10)
    args = parser.parse_args()

    rows = []
    for raw in list(csv.reader(resolve(args.csv).open(encoding="utf-8-sig")))[1:]:
        cells = [c.strip() for c in raw if c.strip()]
        if cells:
            rows.append(cells)
    if not rows:
        raise SystemExit("CSV 为空")

    if len(rows) % args.folds:
        raise SystemExit(f"{len(rows)} 行无法均分到 {args.folds} 折")
    per_fold = len(rows) // args.folds

    for k in range(args.folds):
        chunk = rows[k * per_fold:(k + 1) * per_fold]
        same = sum(1 for r in chunk if len(r) == 3)
        diff = sum(1 for r in chunk if len(r) == 4)
        if same + diff != len(chunk):
            raise SystemExit(f"fold {k+1} 有行既不是 3 列也不是 4 列")
        print(f"fold {k+1}: 同类 {same}  异类 {diff}")
        if same and diff and (same + diff) and abs(same - diff) > per_fold * 0.1:
            print(f"  警告: fold {k+1} 同类/异类数量不均衡，可能不是官方顺序")

    lines = [str(args.folds)] + ["\t".join(r) for r in rows]
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写出: {out}（{len(rows)} 对 / {args.folds} 折）")


if __name__ == "__main__":
    main()
