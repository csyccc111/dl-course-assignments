#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$PWD/models/hf}"
export TOKENIZERS_PARALLELISM=false

DATA_DIR="${DATA_DIR:-$PWD/chnsenticorp}"
OUT="${OUT:-$PWD/outputs}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
AMP="${AMP:-bf16}"
SEEDS="${SEEDS:-42 43 44}"
MODELS="${BERT_MODELS:-bert-base-chinese hfl/rbt3}"

for seed in $SEEDS; do
  for model in $MODELS; do
    python -m src.train_bert \
      --model "$model" --seed "$seed" \
      --data-dir "$DATA_DIR" --output-dir "$OUT" \
      --device "$DEVICE" --amp "$AMP" --num-workers "$NUM_WORKERS"
  done
done

python -m src.summarize