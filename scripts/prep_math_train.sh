#!/usr/bin/env bash
# Prepare MATH-lighteval train.parquet for RL training.
# Output: rl_datasets/math/train.parquet (and test.parquet, but we use math500 for eval)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="${PARENT_DIR}/rl_datasets/math"
mkdir -p "$OUT_DIR"

if [ -f "${OUT_DIR}/train.parquet" ]; then
  echo "[prep_math_train] already present: ${OUT_DIR}/train.parquet"
  python3 -c "import pandas as pd; df = pd.read_parquet('${OUT_DIR}/train.parquet'); print(f'  rows: {len(df)}')"
  exit 0
fi

echo "[prep_math_train] downloading + preprocessing MATH-lighteval..."
cd "$PARENT_DIR"
export PYTHONPATH="${PARENT_DIR}/verl_arpo_entropy:${PYTHONPATH:-}"
python3 verl_arpo_entropy/examples/data_preprocess/math_dataset.py --local_dir "$OUT_DIR"

echo "[prep_math_train] done → ${OUT_DIR}/train.parquet"
python3 -c "import pandas as pd; df = pd.read_parquet('${OUT_DIR}/train.parquet'); print(f'  rows: {len(df)}'); print(df.iloc[0])"
