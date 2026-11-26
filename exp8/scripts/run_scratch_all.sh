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

for seed in $SEEDS; do
  for model in rnn lstm transformer; do
    python -m src.train_scratch \
      --model "$model" --seed "$seed" \
      --data-dir "$DATA_DIR" --output-dir "$OUT" \
      --device "$DEVICE" --amp "$AMP" --num-workers "$NUM_WORKERS"
  done
done

for max_len in 64 128 192; do
  python -m src.train_scratch \
    --model lstm --seed 42 --max-len "$max_len" --tag "len${max_len}" \
    --data-dir "$DATA_DIR" --output-dir "$OUT" \
    --device "$DEVICE" --amp "$AMP" --num-workers "$NUM_WORKERS"
done

python -m src.summarize