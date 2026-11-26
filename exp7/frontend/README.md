# 人脸检测与对齐（视觉前端 · 成员 A）

> 本目录原为 `deep_learning_7/`，并入小组工作区后重命名为 `exp7/frontend/`；
> 内部结构与命令不变，仍在 `exp7/frontend/` 下运行。A→B 的串联见
> [`../README.md`](../README.md)（一键联调：`python -m backend.bridge`）。

深度学习课程实验 7「人脸识别系统设计」的**视觉前端**部分：把人脸从原图/视频帧里找出来、用 5 个关键点做相似变换对齐、裁成 112×112，交给后端识别模型。

- **检测**：YuNet（主方案，OpenCV 自带，230 KB）+ CenterFace（对比方案，7.3 MB）
- **对齐**：移植 [`face.evoLVe`](https://github.com/ZhaoJ9014/face.evoLVe) 的 `applications/align`（`align_trans.py` + `matlab_cp2tform.py`），修掉 6 处 Python 2 遗留问题与 1 处逻辑 bug
- **产出**：对齐人脸图、检测可视化、逐脸 CSV 报表、检测器对比表、阈值扫描曲线、失败样本归因

---

## 1. 环境

```bash
pip install -r requirements.txt
```

| 依赖 | 版本（实测） | 说明 |
|---|---|---|
| Python | 3.14.0 | 3.8+ 均可 |
| opencv-python | 5.0.0.93 | 4.5.4+ 即可，只用到 `FaceDetectorYN` 与 `dnn` |
| numpy | 2.3.4 | 2.x 也可以（已修掉 `np.bool`） |
| matplotlib | 3.10.7 | 可选，只用于画阈值曲线 |

纯 CPU 跑，不需要 GPU、不需要 PyTorch。

> 本次实测用的是 Python 3.14 的虚拟环境（Windows 上可能同时存在多个 Python，
> 若 `python` 不是目标解释器，命令里显式写全路径即可）。

## 2. 下载模型

```bash
python download_models.py
```

会拉到 `models/` 下：

```
models/face_detection_yunet_2023mar.onnx   0.23 MB
models/centerface.onnx                     7.53 MB
```

脚本会自动识别 Git LFS 指针文件（130 字节那种伪文件）并切换备用地址；两个仓库都能直连时也可以手动下载放进去。

## 3. 快速开始

```bash
# 单图：看效果，打印每张脸的置信度/正脸度/对齐残差
python detect_align.py image --input demo.jpg --detector both

# 批量：跑整个目录，导出可视化 + 对齐人脸 + CSV 报表
python detect_align.py batch --input data/test --out outputs/run

# 检测器对比：YuNet vs CenterFace，出 Markdown 对比表
python detect_align.py compare --input data/test --out outputs/compare

# 阈值扫描：置信度 0.3~0.9 的检出曲线
python detect_align.py sweep --input data/test --out outputs/sweep

# 视频流：逐帧检测 + 对齐，输出标注视频与 FPS 统计
python detect_align.py video --input demo.mp4 --out outputs/video
```

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--detector` | `yunet` | `yunet` / `centerface` / `both` |
| `--conf` | 按检测器 | YuNet 0.6、CenterFace 0.5；显式指定则覆盖 |
| `--nms` | 0.3 | NMS 阈值 |
| `--crop` | 112 | 对齐输出边长，支持 128/160 |
| `--order` | `auto` | 关键点顺序：`auto` / `direct` / `swapped` |
| `--reflective` | 关闭 | 是否允许镜像相似变换（**不要开**，见第 6 节） |
| `--max-side` | 不缩放 | 大图先降采样到该边长再检测，1080p 以上提速一倍 |
| `--min-size` | 24 | 小于该像素数的人脸标 `small_face` |

## 4. 目录结构

```
detect_align.py            CLI 入口（image / batch / compare / sweep / video）
download_models.py         模型下载
make_testset.py            构建测试集（原图 + 6 类合成退化）
analyze_failures.py        按场景标签聚合失败样本
build_report_docx.py       把报告内容渲染进 Word 模板
report_content.py          报告正文（单一数据源，可导出 md / docx）

face_frontend/
├── align_trans.py         对齐核心：相似变换 + 参考点 + warp（改编自 evoLVe）
├── detectors.py           YuNet / CenterFace 统一封装 + IoU-NMS
├── quality.py             关键点顺序自检、质量指标、失败归因
├── viz.py                 检测框/关键点/对齐拼图/对照图
├── io_utils.py            中文路径安全的图片读写
└── pipeline.py            批量处理、对比实验、阈值扫描、视频流
```

## 5. 输出说明

跑完 `batch` / `compare` 后：

```
outputs/run/
├── vis/          <图名>_<检测器>.jpg      检测框 + 5 点 + 置信度
├── aligned/      <图名>_<检测器>_f0.jpg   112×112 对齐人脸
├── grid/         <图名>_<检测器>_grid.jpg 该图所有对齐人脸的拼图
├── report_yunet.csv                       逐脸明细（Excel 直接打开，utf-8-sig）
└── summary.json                           汇总指标
```

`report_*.csv` 每张脸一行，字段：

| 字段 | 含义 |
|---|---|
| `score` | 检测置信度 |
| `bbox_w` / `bbox_h` / `min_side` | 人脸框尺寸 |
| `frontal` | 眼距 / 框宽，正脸约 0.30~0.45，偏小即侧脸 |
| `align_rms` | **对齐残差**，5 点拟合 RMS，单位：112 裁剪图里的像素 |
| `align_scale` | 原图→裁剪图的缩放系数 |
| `roll_deg` | 人脸滚转角 |
| `sharp` | 人脸区 Laplacian 方差 |
| `landmark_order` | 自检选中的关键点顺序（`direct` / `swapped`） |
| `reasons` | 失败标签，`small_face` / `low_score` / `profile` / `bad_align` / `blur`，多标签用 `\|` 分隔 |

`compare` 额外产出 `compare.md`（对比表 + 一致性统计）与 `agreement.csv`（逐图 IoU 明细）。

## 6. 三个值得看的实现细节

### 6.1 为什么 `reflective=False`

evoLVe 原实现允许**镜像**相似变换。问题是：人脸的「左右眼顺序写反」和「正常顺序 + 镜像」在几何上是同一件事，两种解的残差都是 0，靠残差根本发现不了，而对齐结果是一张左右翻转的脸。

实测（同一张脸，只改顺序和 `reflective`）：

| 关键点输入 | reflective | 拟合残差 |
|---|---|---|
| 顺序正确 | True | 2.540 px |
| 顺序颠倒 | False | **162.521 px** |
| 顺序颠倒 | True | 2.540 px |

所以：默认关掉镜像解，并对 `direct` / `swapped` 两种候选顺序各求一次变换取残差小者。这样「检测器的关键点顺序约定」就变成了可自动判定的问题，不用为每个检测器写死顺序表。

### 6.2 CenterFace 的关键点通道是交叉存放的

`landmark` 输出的 10 个通道不是 `(x0,y0,x1,y1,…)`，而是 **2j 存 y、2j+1 存 x**。写反不会报错，只会让残差偏大、关键点画出来像被拉歪——典型的静默错误。

### 6.3 中文路径

Windows 下 `cv2.imread` / `cv2.imwrite` 走单字节 ANSI 路径，项目路径里有中文时会**静默返回 None**（不抛异常）。所有读写统一走 `io_utils.imread_unicode` / `imwrite_unicode`（`np.fromfile` + `imdecode`）。

## 7. 与后端的接口约定

前端只对外产出这两样东西，后端不应依赖检测器/对齐的任何内部细节：

```python
Face:                                # detectors.py
    bbox       np.ndarray (4,)  float32   # x1, y1, x2, y2，原图坐标
    score      float
    landmarks  np.ndarray (5,2) float32   # 5 点，原图坐标
    detector   str

AlignResult:                         # quality.py
    image      np.ndarray (H,W,3) uint8   # 112×112 BGR 对齐人脸
    tfm        np.ndarray (2,3) float32   # 原图->对齐图的 warpAffine 矩阵
    rms        float                      # 对齐残差，用的阈值筛掉坏样本
    order      str                        # 'direct' / 'swapped'
```

推荐后端这样取数据：

```python
from face_frontend.detectors import YuNetDetector
from face_frontend.quality import align_face

det = YuNetDetector(conf_thresh=0.6)
for f in det.detect(img):
    al = align_face(img, f.landmarks, crop_size=112)
    if al.rms > 8.0:          # 对齐质量太差，不要送去注册/比对
        continue
    embedding = model(al.image)   # 后端：112×112 BGR -> 特征
```

## 8. 复现实验

```bash
python download_models.py
python make_testset.py                       # data/test, data/test_hard
python detect_align.py compare --input data/test      --out outputs/compare
python detect_align.py compare --input data/test_hard --out outputs/compare_hard
python detect_align.py sweep   --input data/test      --out outputs/sweep
python analyze_failures.py --report outputs/compare_hard/run/report_yunet.csv \
                           --images data/test_hard --mode suffix
python build_report_docx.py                  # 生成实验报告 docx + md
```

`data/base` 里是 5 张公开人脸图（含单人、双人、远景小脸），`make_testset.py` 会在此基础上生成多人拼图与 6 类退化图。**换成自己的数据集只需改 `--input`**，代码不依赖具体目录结构。

## 9. 已知限制

- 测试集是「真实图 + 合成退化」，样本量小，且合成旋转只能模拟滚转（roll），模拟不了偏航（yaw），所以文中指标只用于方案间相对比较
- 没有在 WIDER FACE 上做定量评估
- 视频是逐帧独立检测，无人脸跟踪，多人场景下框序号会抖动
- 不包含识别模型（成员 B 负责）
