#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$PWD/models/hf}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

mkdir -p "$HF_HOME" logs
python -V
nvidia-smi || true

# Python 3.12 + 旧 pip 会在解析依赖 marker 时崩溃（unknown opcode 112）
python -m pip install -q --upgrade --no-deps --force-reinstall "pip==24.3.1" \
  -i "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
python -m pip --version
python -m src.check_env
python -m pip install -q -r requirements-cloud.txt -i "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
python -m src.eda
python -m src.download_model --model bert-base-chinese --model hfl/rbt3