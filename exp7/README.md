# 实验 7 · 人脸识别系统设计（A 前端 + B 后端）

课程：深度学习方法与应用（IB00267）· 指导教师：彭小江 · 地点/时间：C5-361 / 2026-xx-xx

报告模板与项目记忆不随仓库发布。

小组分工：A 做视觉前端（检测 + 对齐），B 做识别后端（特征 + 人脸库 + 动态系统）。

## 目录

```
exp7/
├── frontend/          成员 A：YuNet/CenterFace 检测 + evoLVe 5 点对齐 → 112×112
│   ├── face_frontend/   检测器、对齐、质检、可视化、批量/对比/扫描/视频
│   ├── data/            base 公开图 + test(10) + test_hard(30 张 6 类退化) + demo.mp4
│   ├── models/          yunet 2023mar + centerface
│   ├── outputs/         A 已跑完的 compare / compare_hard / sweep / figures
│   └── README.md        A 的交付说明（含实现细节与已知限制）
├── backend/           成员 B：SFace 特征 + 人脸库 + 分类头 + LFW/开集评测 + bridge
├── models/            B 的 onnx：yunet 2023mar / 2026may + sface 2021dec
├── data/              lfw/ + umdfaces/（待下载）+ selfmade/（可选）；不进 git
├── outputs/           B 的运行产物，按 <时间戳>_<实验名>/ 归档
└── requirements.txt   合并后的依赖
```

A 交付包的原样快照未随仓库发布（原目录名 `deep_learning_7` 已重命名为 `frontend`；
A 的代码里没有硬编码旧目录名，`detect_align.py` 等仍在 `frontend/` 内正常运行）。

## 从哪个目录运行

| 要跑什么 | cwd | 命令 |
|---|---|---|
| A 的检测/对齐/对比/扫描 | `exp7/frontend/` | `python detect_align.py compare --input data/test --out outputs/compare` |
| B 的全部脚本 | `exp7/` | `python -m backend.bridge --dir frontend/data/test_hard` |

B 侧所有相对路径都会被 [`backend/paths.py`](backend/paths.py) 锚定到 `exp7/`，
所以从其他目录用脚本全路径调用也不会写散文件。

## 接口：A → B

只交换 **112×112 BGR 对齐图**，两边的 `Face` 类型不互通（A 是 `xyxy`，B 是 `xywh`，
且 SFace `alignCrop` 需要 YuNet 的 1×15 原始行，A 不提供）。

```python
# A 侧
al = align_face(img, face.landmarks, crop_size=112)   # AlignResult(image, tfm, rms, order)
# B 侧
feat = embedder.embed_aligned(al.image)               # 128 维，L2 归一化
```

质检门：`rms > 8.0` 或带 `small_face/profile/blur/low_score/bad_align` 标签的脸不进
注册与比对。`backend/bridge.py` 已经把这一步包好：

```bash
python -m backend.bridge --dir frontend/data/test --limit 5
# → outputs/<时间戳>_bridge/{aligned/*.jpg, report.json}
```

## 两条必须知道的坑

1. **ONNX 模型路径含中文会加载失败**：`cv2.FaceDetectorYN/FaceRecognizerSF` 报
   `Can't read ONNX file`（图片读写的 `imread_unicode` 对它无效）。
   `paths.model_path()` 会把模型复制到 `%TEMP%/exp7_ascii_models/` 再加载。
2. **图片中文路径**：直接 `cv2.imread` 静默返回 `None`，统一走
   `imread_unicode` / `imwrite_unicode`。

## 数据与合规

2026-11-03 决策：**不做多人自采**，指标与演示全部由公开数据集承担。

- `data/lfw/`：LFW 原图 + `pairs.txt`，做 1:1 验证与 1:N 评测。**不要用
  `lfw-deepfunneled` / `lfw-funneled`**（已被检测框居中缩放，和 ArcFace 112 模板不一致）。
- `data/umdfaces/`：UMDFaces batch1（非名人库），做 1:N 注册/识别与**开集拒识**。
  陌生人探针取自从未注册的身份，比自采 10 人可信得多。
- `data/selfmade/`（可选）：只在现场演示时注册本人 + 自愿同学；**必须征得同意、
  课后删除、不进公开仓库**。
- 若还要动态识别：YouTube Faces（3425 段 / 1595 人，OpenDataLab 有镜像）或
  UMDFacesVideos；`app.py` 用 `--video` 输入，与摄像头走同一条链路。
- 公开数据集遵守各自 license，报告注明来源、license 与对齐方式。

## 报告

- A 的章节已成稿：[`frontend/深度学习课程实验-7-第X组-组名-检测与对齐.docx`](frontend/深度学习课程实验-7-第X组-组名-检测与对齐.docx)
  （由 `report_content.py` + `build_report_docx.py` 渲染进模板，正文数据单一来源）
- B 的章节操作手册：[`backend/README.md`](backend/README.md)；写作要点见
  [`../memory/report.md`](../memory/report.md)
- 统稿：B 章节接进同一套 `report_content.py` 机制，不要手改 Word，否则两边的数字对不上。
