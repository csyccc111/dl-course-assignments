# 实验 8 运行说明（优云智算 / 实验室 Linux 服务器）

> 实验室服务器同样适用：把 `/root/workspace` 换成你的持久目录（例如 `/media/lenovo/workspace-csy`），
> 先 `conda activate` 你的环境，再按下面步骤走。

## 目录约定

优云智算实例登录后先执行 `pwd`，本文假设工作目录是 `/root/workspace`（持久数据盘）。
如果平台给的是别的目录，把下面命令里的 `/root/workspace` 换成实际路径即可。

```text
/root/workspace/dl-course-assignments/exp8/
  chnsenticorp/        # 三个 tsv
  src/                 # 训练与评测代码
  scripts/             # 一键脚本
  models/hf/           # HF 缓存（首次下载后复用）
  outputs/             # 运行产物
  logs/                # 日志
```

## 1. 克隆仓库

```bash
cd /root/workspace
git clone https://github.com/csyccc111/dl-course-assignments.git
cd dl-course-assignments/exp8
```

如果仓库设为私有，先在云主机上配置 GitHub PAT 或 SSH key；不要把 token 写进脚本或日志。

## 2. 检查显卡

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

必须看到 `True` 和 `RTX 4090`。如果镜像自带 torch 不支持 4090（CUDA < 11.8），换镜像，不要在云上重装 torch。

## 3. 安装依赖与模型

```bash
bash scripts/setup_cloud.sh
```

脚本会：

- 设置 `HF_ENDPOINT=https://hf-mirror.com`、`HF_HOME=$PWD/models/hf`
- 安装 `requirements-cloud.txt`（不装 torch）
- 生成数据统计与 `outputs/figures/fig1_data.png`
- 下载 `bert-base-chinese`（约 400 MB）和 `hfl/rbt3`（约 156 MB）

如果 pip 慢，可以换源：

```bash
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash scripts/setup_cloud.sh
```

## 4. 冒烟

先在当前 shell 里加载缓存配置：

```bash
source scripts/env.sh
python -m src.train_scratch --model lstm --smoke --device cuda
python -m src.train_bert --model hfl/rbt3 --smoke --device cuda
```

两次都应在一分钟内结束，并打印 `[done]`。

## 5. 全量训练

```bash
nohup bash scripts/run_all.sh > logs/run_all.log 2>&1 &
tail -f logs/run_all.log
```

`run_all.sh` 依次执行：

1. 3 个从零模型（RNN/LSTM/Transformer）× 3 个 seed
2. LSTM 序列长度消融 64/128/192
3. `bert-base-chinese` 与 `hfl/rbt3` 各 3 个 seed
4. 汇总 `outputs/summary.csv`
5. 生成全部图
6. 打包 `outputs_*.tar.gz`

4090 上预计 30–60 分钟。

## 6. 回收结果

```bash
ls -lh outputs_*.tar.gz
```

在 JupyterLab 里下载该压缩包，解压到本地 `exp8/` 下即可。报告只依赖 `outputs/`，不需要云端权重。

## 7. 关机

训练完在控制台停止实例，不要只关网页。优云智算按实例运行时长计费。

## 常见问题

- **rbt3 加载报 `torch>=2.6`**：`hfl/rbt3` 只有 `pytorch_model.bin`，transformers 4.57.6 在 torch<2.6 时拒绝加载。
  先按上文把 `.bin` 转成 `model.safetensors`；不想折腾就跳过它：
  `BERT_MODELS=bert-base-chinese bash scripts/run_all.sh`。
- **`import sklearn` 时 segfault**：conda 里旧 scipy 和 pip 新 numpy 的 ABI 不匹配。按 `requirements-cloud.txt`
  固定 `numpy==1.26.4 / scipy==1.13.1 / scikit-learn==1.5.2`，或新建独立 venv（不要继承系统 scipy）。
- **卡在 `Fetching 10 files: 90%`**：`snapshot_download` 默认把 Flax/TF 权重（`flax_model.msgpack`、`tf_model.h5`）也算进去。
  PyTorch 权重已经下完，直接 `Ctrl+C`；重跑 `python -m src.download_model --model bert-base-chinese --model hfl/rbt3`
  （已默认忽略 `*.h5/*.msgpack/*.ot/*.onnx`），或只保留 `model.safetensors`。
- **pip 报 `unknown opcode 112`**：conda/系统 Python 3.12 里的 pip 太旧。执行
  `python -m pip install --upgrade --no-deps --force-reinstall "pip==24.3.1" -i https://pypi.tuna.tsinghua.edu.cn/simple`，
  再重装依赖；不行用 `conda install -n <env> -c conda-forge "pip>=24" setuptools wheel -y`。
- **HF 下载失败**：确认 `HF_ENDPOINT=https://hf-mirror.com`；或先手动 `python -m src.download_model --model bert-base-chinese`。
- **transformers 冲突**：镜像里已有旧版时，换干净的 PyTorch 镜像，或 `python -m venv .venv-cloud && source .venv-cloud/bin/activate` 后重跑安装。
- **路径含中文**：云端统一用 ASCII 路径，不要把项目放到中文目录。
- **OOM**：4090 24G 足够；若同时跑多个任务，串行执行，不要并发。
- **DataLoader 卡住**：把 `--num-workers 0` 传给训练脚本。