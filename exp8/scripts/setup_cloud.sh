#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$PWD/models/hf}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$HF_HOME" logs
python -V
nvidia-smi || true

python -m src.check_env
python -m pip install -q -r requirements-cloud.txt -i "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
python -m src.eda
python -m src.download_model --model bert-base-chinese --model hfl/rbt3