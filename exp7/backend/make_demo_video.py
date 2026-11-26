"""用 LFW 真人照片拼一段可评估的「演示视频」，并写逐帧真值。

为什么需要它：demo 只吃视频文件，但我们没有真实拍摄素材。用已知身份的静态照片
合成视频，可以给 app.py 提供**逐帧真值**，从而算帧级识别率、误拒、陌生人拒识。

用法
----
python -m backend.make_demo_video --root data/lfw/images --out data/demo \
    --names "George_W_Bush,Colin_Powell" --stranger "Donald_Rumsfeld" \
    --frames-per-seg 60 --rounds 2

产物
----
demo.mp4            640x480 / 25fps，每人若干段，最后一段是陌生人
demo_gt.json        {enrolled, stranger, segments[{start,end,identity}],
                     gallery{身份: [注册照路径]}}
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from .paths import resolve
except ImportError:  # 允许 python backend/make_demo_video.py 直接运行
    from paths import resolve

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _pick_images(root: Path, name: str) -> list:
    folder = root / name
    if not folder.is_dir():
        raise SystemExit(f"找不到身份目录: {folder}")
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if len(files) < 6:
        raise SystemExit(f"{name} 只有 {len(files)} 张图，建议 >= 6 张")
    return files


def _render(canvas_size, image, rng, face_frac):
    """把人脸图缩放到画面高度的 face_frac，并在画布内随机平移（不裁切）。"""
    w, h = canvas_size
    target_h = max(48, min(int(h * face_frac), h))
    ratio = target_h / image.shape[0]
    target_w = max(1, min(int(image.shape[1] * ratio), w))
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), 40, dtype="uint8")
    x = rng.randint(0, max(0, w - resized.shape[1]))
    y = rng.randint(0, max(0, h - resized.shape[0]))
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="合成演示视频 + 逐帧真值")
    parser.add_argument("--root", required=True, help="按身份分目录的图片根")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--names", required=True, help="注册身份，逗号分隔")
    parser.add_argument("--stranger", required=True, help="不注册的陌生人身份")
    parser.add_argument("--frames-per-seg", type=int, default=60)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = resolve(args.root)
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    names = [n.strip() for n in args.names.split(",") if n.strip()]
    if len(names) < 2:
        raise SystemExit("至少需要 2 个注册身份")

    pools, gallery = {}, {}
    for name in names + [args.stranger]:
        files = _pick_images(root, name)
        gallery[name] = [str(p) for p in files[:5]]  # 注册照与视频帧不重叠
        pools[name] = files[5:]

    order = []
    for _ in range(args.rounds):
        order.extend(names)
    order.append(args.stranger)

    tmp = Path(tempfile.gettempdir()) / f"exp7_make_demo_{int(time.time() * 1000)}.mp4"
    writer = cv2.VideoWriter(
        str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )
    if not writer.isOpened():
        raise SystemExit("VideoWriter 打开失败")

    segments, frame_index = [], 0
    size = (args.width, args.height)
    for name in order:
        pool = pools[name]
        for _ in range(args.frames_per_seg):
            image = cv2.imdecode(
                np.fromfile(str(rng.choice(pool)), np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                continue
            frame = _render(size, image, rng, face_frac=rng.uniform(0.55, 0.80))
            writer.write(frame)
            frame_index += 1
        segments.append({
            "start": frame_index - args.frames_per_seg,
            "end": frame_index,
            "identity": name,
        })
        print(f"段: {name}  帧 {segments[-1]['start']}..{segments[-1]['end']}")
    writer.release()

    dest = out / "demo.mp4"
    shutil.move(str(tmp), str(dest))
    meta = {
        "video": str(dest),
        "fps": args.fps,
        "size": [args.width, args.height],
        "enrolled": names,
        "stranger": args.stranger,
        "segments": segments,
        "gallery": {n: gallery[n] for n in names},
        "source": str(root),
        "note": "用 LFW 静态照片合成，逐帧真值仅用于评估，不代表真实拍摄条件",
    }
    (out / "demo_gt.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n视频: {dest}（{frame_index} 帧）")
    print(f"真值: {out / 'demo_gt.json'}")


if __name__ == "__main__":
    main()
