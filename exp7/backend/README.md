# B 侧代码：人脸识别特征、人脸库、分类头、LFW 评测

A 的检测与对齐在 [`../frontend/`](../frontend/)；两边怎么串起来见
[`../README.md`](../README.md)。项目状态与阶段计划见
[../../PROJECT_MEMORY.md](../../PROJECT_MEMORY.md)。

## 文件

```
backend/
├── paths.py             # exp7/ 锚定路径 + 非 ASCII 模型副本 + outputs 归档目录
├── bridge.py            # A 前端（检测/对齐/质检）→ B 特征
├── embedder.py          # YuNet + SFace：检测、临时对齐、128 维特征、L2 归一化
├── build_embeddings.py  # 批量提取 embedding，缓存 npz
├── gallery.py           # 人脸库注册、保存、余弦匹配
├── train_head.py        # 冻结 embedding 上的分类头（不是端到端训练）
└── eval_lfw.py          # LFW 1:1 验证，10 折 accuracy / EER / AUC
```

## 环境

用仓库根目录的 `.venv`：Python 3.14.8 / opencv-python 5.0.0 / numpy 2.5.4 /
scikit-learn 1.9.1。本机无 CUDA，跑 baseline 不需要 torch。

```bash
cd exp7
../.venv/Scripts/python.exe -m backend.bridge --image frontend/data/test/t02_obama.jpg
```

命令默认从 `exp7/` 运行。`paths.py` 会把相对路径锚定到 `exp7/`，所以从别的目录用
脚本全路径调用（`python .../exp7/backend/bridge.py --image ...`）同样正确。

## 模型

放到 `models/`：

- `face_detection_yunet_2023mar.onnx`
- `face_recognition_sface_2021dec.onnx`
- `face_detection_yunet_2026may.onnx`（OpenCV 5.x 加载 2023mar 报错时备用）

来源 OpenCV Zoo。仓库用 Git LFS 存模型，raw 链接只会拿到 131 字节的指针文件，
要用 media 直链：

```bash
curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -L -o models/face_recognition_sface_2021dec.onnx \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

`frontend/models/` 里是 A 自己的模型副本（YuNet 与这里的字节相同，另有一个
CenterFace），互不依赖。

## 数据目录

```
data/
├── lfw/                    # LFW 原图 + pairs.txt（不要用 deepfunneled）
└── selfmade/
    ├── alice/
    │   ├── enroll/001.jpg  # 3–5 张注册照
    │   └── test/101.jpg    # 10–20 张测试照
    └── bob/...
```

另外准备 5–10 个**不注册**的人做陌生人拒识，否则 FAR / 拒识率出不来。

> 2026-11-03 决策：不做多人自采。识别指标改由公开数据集提供（LFW 做 1:1，
> UMDFaces 做 1:N + 开集拒识），`selfmade/` 只留给现场演示时的本人注册（可选）。

## 和 A 的接口（2026-11-02 实测）

A 只对外给两样东西，B 只吃对齐图：

```python
from backend.bridge import FrontendBridge
from backend.embedder import FaceEmbedder

bridge = FrontendBridge(conf=0.6, crop=112, max_rms=8.0)
embedder = FaceEmbedder("frontend/models/face_detection_yunet_2023mar.onnx",
                        "models/face_recognition_sface_2021dec.onnx")

for face, feat in bridge.embed(image, embedder):     # feat: 128 维，L2 归一化
    print(face.score, face.rms, face.order, face.flags)
    match = gallery.match(feat)
```

- `face.bbox` 是 A 的口径 `(x1, y1, x2, y2)`；B 自己 `embedder.detect()` 返回的
  `Face.bbox` 是 `(x, y, w, h)`。**两个 Face 不要混用**，桥接层只传 112×112 图。
- `flags` 来自 A 的质检：`small_face` / `low_score` / `profile` / `bad_align` /
  `blur`；`only_ok=True` 时注册与比对会跳过带标签的脸。
- A 的对齐核心是 evoLVe `align_trans`（ArcFace 112 模板），`order=auto` 会在
  `direct`/`swapped` 里取残差小者。实测 65 张脸全部判为 `direct`。
- 阈值：0.363 是 SFace `alignCrop` 下的官方起点。在 A 的模板上实测同人
  0.740~0.936、异人 0.101~0.147，暂时够用，**正式阈值仍要在验证集重扫**。

## 快速开始

### 1. 端到端联调（A 对齐 → B 特征）

```bash
python -m backend.bridge --dir frontend/data/test_hard --limit 10
```

产物自动落进 `outputs/<时间戳>_bridge/`：`aligned/*.jpg` + `report.json`
（逐脸 score / rms / order / flags / 是否入库）。

### 2. 单图（B 自己的临时对齐，仅作 baseline 对照）

```bash
python -m backend.embedder --image data/selfmade/alice/enroll/001.jpg
```

### 3. 批量提取 embedding

```bash
python -m backend.build_embeddings --root data/selfmade --output outputs/selfmade.npz
```

输出 npz 含 `embeddings/labels/paths/splits`，并生成同名 json 记录人数与失败原因。
检测失败的图会跳过，不污染缓存。

### 4. 构建人脸库

```bash
python -m backend.gallery --embeddings outputs/selfmade.npz \
    --output outputs/gallery.npz --split enroll --threshold 0.363
```

每人多条注册 embedding 先平均再归一化。

### 5. 训练分类头

```bash
python -m backend.train_head --embeddings outputs/selfmade.npz \
    --output outputs/head_logreg.pkl
```

- 默认 `logreg`，可换 `--model mlp`；有 `train/enroll + test` 划分就用现成的，否则随机划分。
- 产出 `head_metrics.json`（accuracy、classification report、混淆矩阵）。
- 这是 **closed-set 分类头，不是端到端 CNN 训练**，报告里必须写清楚；开放集识别用人脸库 + 阈值。

### 6. LFW 评测

```bash
python -m backend.eval_lfw --lfw data/lfw --output outputs/lfw_eval.json
python -m backend.eval_lfw --lfw data/lfw --limit-pairs 100     # 冒烟
```

第一轮逐张检测并缓存到 `outputs/lfw_embeddings.pkl`，之后复用；缓存与模型/阈值绑定，
换模型会自动失效重算。10 折每折用其余 9 折选阈值，另报 EER、AUC 与检测失败数。

### 7. 开集 / 1:N 评测（公开数据集，替代自采）

```bash
# UMDFaces：CSV 标注布局（batch 解压目录）
python -m backend.protocol_open --mode csv \
  --csv "D:/umdfaces/umdfaces_batch1/umdfaces_batch1_ultraface.csv" \
  --images-root "D:/umdfaces/umdfaces_batch1" \
  --n-ids 400 --shots 3 --min-images 8 --imposter-frac 0.25

# 任何按人分目录的数据（LFW、自采、A 的测试集）
python -m backend.protocol_open --mode dirs --root data/lfw --n-ids 500 --shots 3
```

协议：按身份切分 gallery / imposter；每人 `--shots` 张注册、其余作 genuine probe；
imposter 身份**从不注册**，作 stranger probe。探针再分 dev/test，阈值只在 dev 上选。
产出 `metrics.json`（Rank-1/5、EER、DIR@FPIR=1e-2/1e-3）、`thresholds_dev.csv`、
`scores.npz`（画 ROC 用）、`embeddings_cache.npz`（重跑秒过）。

质量门会丢掉模糊/小脸样本，所以 `--min-images` 要留余量；`dev` 陌生人探针 < 1000 时
脚本会在 `warnings` 里提示 FPIR=1e-3 不可靠。

### 8. CASIA-WebFace（k-shot / 大库 1:N）

HF 镜像的分片是 parquet，先导成身份目录：

```bash
python -m backend.import_parquet --src data/casia/shard0.parquet \
  --out data/casia/images --per-id 12 --min-per-id 8
```

> **该镜像的图像是 112×112 预对齐 PNG**，必须加 `--prealigned` 直接送特征。
> 实测在紧裁图上再跑一遍检测+对齐会让同脸特征余弦掉到中位 0.913（最低 0.339）。

```bash
python -m backend.protocol_open --mode dirs --root data/casia/images --prealigned \
  --shots 3 --min-images 8 --imposter-frac 0.3 --max-stranger-probes 12
```

数据质量有限（同人余弦均值 0.49，同一 SFace 在这批图上的 1:1 EER 8.3%，而 LFW 上是 1.4%），
**只用于 k-shot 相对趋势**，绝对精度以 LFW 为准。

### 9. 动态识别 demo（视频文件）

不做摄像头、不做真人采集、不做多人跟踪；逐帧独立检测，**只认画面里最大的一张脸**。

```bash
# 1) 合成可评估素材：LFW 三个身份 + 一个陌生人，带逐帧真值
python -m backend.make_demo_video --root data/lfw/images --out data/demo \
  --names "George_W_Bush,Colin_Powell,Tony_Blair" --stranger "Donald_Rumsfeld" \
  --frames-per-seg 60 --rounds 2

# 2) 跑 demo（自动用真值里的注册照建库、算帧级指标、扫阈值）
python -m backend.app --source data/demo/demo.mp4 --gt data/demo/demo_gt.json

# 3) 任意视频 + 自带注册库
python -m backend.app --source clip.mp4 --gallery outputs/<run>/gallery.npz
# 或从视频开头若干帧自动注册一个人
python -m backend.app --source frontend/data/demo.mp4 --auto-enroll 20 --name target
```

产物：`demo_annotated.mp4`（画框/姓名/相似度）、`frames.csv`（逐帧明细）、
`summary.json`（检出率、误接受/误认、延迟分位、阈值扫描）、`gallery.npz`。

**实测（2026-11-12，420 帧 640×480）**：检出率 90.9%，检出帧准确率 97.91%，
帧级准确率 90.0%，陌生人误接受 **0/56**、误认 **0/326**；延迟 25.9ms 均值 / 31.2ms p95
（≈38.6 FPS）；阈值 0.40–0.55 结果一致，0.60 起下降。

### 10. 从零训练小 CNN（实验目的 ②）

```bash
python -m backend.train_small_cnn --root data/train62 --min-images 20 --epochs 40
```

4 个卷积块（16/32/64/128）+ BN + 全局池化，0.106M 参数，112×112 输入；Adam 1e-3、batch 32。
CPU 8 线程约 **16.1 s/epoch**。只依赖 `torch`（CPU 版），不需要 torchvision。

**实测**：LFW 62 类 / 3023 张（2418 训 / 605 测），最终 train 75.89% / test **56.69%**
（最佳 63.14% @ep32，测试损失随后回升 → 过拟合）；同一划分下 SFace 冻结特征 + LogReg
是 **99.83%**。这组对照就是报告里「训练本质 = 数据规模 + 特征质量」的证据。

### 11. 出图

```bash
python -m backend.make_figures     # -> outputs/figures/
```

依赖 `matplotlib`。缺哪个实验产物就跳过哪张图：相似度直方图、阈值-FAR/FRR 曲线、
k-shot 柱状图、训练曲线（含 SFace 对照横线）。

## 注意事项

- **ONNX 路径不能含中文**：`cv2.FaceDetectorYN/FaceRecognizerSF` 遇到非 ASCII 路径会报
  `Can't read ONNX file`，`imread_unicode` 那套对它无效。`paths.model_path()` 已自动把模型
  复制到 `%TEMP%/exp7_ascii_models/` 再加载。
- **图片读写中文路径**：统一用 `imread_unicode` / `imwrite_unicode`，直接 `cv2.imread`
  会静默返回 `None`。
- **对齐模板要一致**：注册、评测、demo 都走 A 的 evoLVe 112×112；`embedder.align()`
  的 SFace `alignCrop` 只用于 baseline 对照，不要混着训练和测试。
- **LFW 用原图重新对齐**，`lfw-deepfunneled` 的模板和 ArcFace 不同。
- **阈值不要拍脑袋**：验证集扫出来，报告 FAR/FRR 或 accuracy/EER。
- **产物要留档**：每个实验一个 `outputs/<时间戳>_<名>/` 目录，别覆盖上一次的数字。
- **自采数据合规**：征得同意、课后删除、不进公开仓库（`data/` 与 `outputs/` 已在 .gitignore）。
- 公开数据集（LFW、CASIA-WebFace 等）遵守各自 license，报告注明来源。

## 下一步

1. 用 A 的 test/test_hard 出人脸库 Rank-1、FAR/FRR 与阈值扫描曲线（阶段 2）。
2. LFW 真实评测数字（阶段 2）。
3. 补一个能反向传播的小训练，覆盖实验目的「训练人脸识别模型的本质」（阶段 3）。
4. `app.py` 摄像头 demo：注册、识别、拒识、FPS，并自己处理多人场景的身份漂移（阶段 4）。
5. 章节并进 A 的 `report_content.py`，统稿（阶段 5）。
