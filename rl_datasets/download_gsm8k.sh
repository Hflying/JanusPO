#!/usr/bin/env bash
#   GSM8K   verl   parquet  verl examples/data_preprocess/gsm8k.py  
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARPO_ROOT="$(dirname "$SCRIPT_DIR")"
OUT_DIR="${GSM8K_LOCAL_DIR:-${SCRIPT_DIR}/gsm8k}"
mkdir -p "$OUT_DIR"

export PYTHONPATH="${ARPO_ROOT}/verl_arpo_entropy:${PYTHONPATH:-}"
python3 "${ARPO_ROOT}/verl_arpo_entropy/examples/data_preprocess/gsm8k.py" --local_dir "$OUT_DIR"
echo " : ${OUT_DIR}/train.parquet   ${OUT_DIR}/test.parquet"
