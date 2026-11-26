#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bash scripts/setup_cloud.sh
python -m src.eda
python -m src.toys
bash scripts/run_scratch_all.sh
bash scripts/run_bert_all.sh
python -m src.summarize
python -m src.make_figures

OUTPUT_ARCHIVE="outputs_$(date +%Y%m%d_%H%M).tar.gz"
tar --exclude='outputs/cache' --exclude='outputs/runs/*/best.pt' -czf "$OUTPUT_ARCHIVE" outputs
echo "[done] $OUTPUT_ARCHIVE"