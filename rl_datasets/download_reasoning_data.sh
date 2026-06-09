#!/usr/bin/env bash
#   Hugging Face   ARPO-RL-Reasoning-10K   train_10k.parquet  valid.parquet 
#   train_10k.parquet README   test.parquet  
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ARPO_RL_DATA_OUT="$SCRIPT_DIR"
mkdir -p "$ARPO_RL_DATA_OUT"

if ! command -v python3 &>/dev/null; then
  echo "  python3 : conda activate arpo  "
  exit 1
fi

python3 << 'PY'
import os
import shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

out_dir = Path(os.environ["ARPO_RL_DATA_OUT"])
repo_id = "dongguanting/ARPO-RL-Reasoning-10K"
train_src = hf_hub_download(repo_id, "train_10k.parquet", repo_type="dataset")
train_dst = out_dir / "train_10k.parquet"
shutil.copy2(train_src, train_dst)
print(f" : {train_dst}")

df = pd.read_parquet(train_dst)
n = min(300, len(df))
val_dst = out_dir / "valid.parquet"
df.iloc[-n:].to_parquet(val_dst, index=False)
print(f"  ({n}  ): {val_dst}")
PY
