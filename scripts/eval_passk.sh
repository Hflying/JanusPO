#!/usr/bin/env bash
# ============================================================================
# pass@k evaluation for §6.6 "Exploration without collapse"
# Three 3B checkpoints:
#   SFT init (Qwen2.5-3B-Instruct zero-shot)
#   GRPO clean  step 30   (GSM8K peak: 74.8%)
#   JanusPO     step 200  (GSM8K peak: 84.5%)
# Two benchmarks:
#   GSM8K   (in-domain)
#   MATH-500 (cross-benchmark)
# Sampling:
#   n = 16 (paper reports pass@1, pass@2, pass@4, pass@8, pass@16)
#   T = 0.6
#   max_resp = 1024
# Outputs:
#   checkpoints/_eval/<label>_<bench>_passk/run.log                full trainer log
#   checkpoints/_eval/<label>_<bench>_passk/validation_data/      per-sample JSONL
#   checkpoints/_eval/passk_summary.txt                           aggregated results
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"

SUMMARY="${PARENT_DIR}/checkpoints/_eval/passk_summary.txt"
mkdir -p "$(dirname "$SUMMARY")"
date "+=== pass@k eval started %F %T ===" | tee "$SUMMARY"

MATH_PARQUET="${PARENT_DIR}/rl_datasets/math500/test.parquet"
GSM_PARQUET="${PARENT_DIR}/rl_datasets/gsm8k/test.parquet"
AIME_PARQUET="${PARENT_DIR}/rl_datasets/aime/aime_combined_test.parquet"
for f in "$MATH_PARQUET" "$GSM_PARQUET" "$AIME_PARQUET"; do
  [ -f "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

# ckpt label | HF dir
declare -a CKPTS=(
  "sft|$HOME/models/Qwen2.5-3B-Instruct"
  "grpo30|${PARENT_DIR}/checkpoints/grpo_clean_v2/hf_step30"
  "janus200|${PARENT_DIR}/checkpoints/plan_c_asymkl_v2_ub25_v1/hf_step200"
)

# bench label | parquet
declare -a BENCHES=(
  "gsm8k|${GSM_PARQUET}"
  "math500|${MATH_PARQUET}"
  "aime|${AIME_PARQUET}"
)

run_one() {
  local ckpt_label="$1"; local ckpt_path="$2"
  local bench_label="$3"; local parquet="$4"
  local name="${ckpt_label}_${bench_label}_passk"
  local out="${PARENT_DIR}/checkpoints/_eval/${name}"
  local val_dir="${out}/validation_data"

  echo ""
  echo "=============================================" | tee -a "$SUMMARY"
  echo "[eval] ${name}  n=16 T=0.6"              | tee -a "$SUMMARY"
  echo "  ckpt    : $ckpt_path"                   | tee -a "$SUMMARY"
  echo "  parquet : $parquet"                     | tee -a "$SUMMARY"
  echo "=============================================" | tee -a "$SUMMARY"

  # skip if already finished
  if [ -f "${out}/run.log" ] \
     && grep -q "val-core/.*reward/best@16" "${out}/run.log" 2>/dev/null; then
    echo "  (skip, already completed)" | tee -a "$SUMMARY"
    grep -oE "val-core/[^[:space:]]+reward/(best|mean|maj)@[0-9]+[^[:space:]]*:[0-9.]+" \
      "${out}/run.log" | sort -u | tee -a "$SUMMARY"
    return 0
  fi

  mkdir -p "$val_dir"

  # AIME problems need longer chain-of-thought; GSM8K/MATH-500 fit in 1024.
  local max_resp=1024
  if [ "$bench_label" = "aime" ]; then
    max_resp=4096
  fi

  ARPO_EVAL_CKPT="$ckpt_path" \
  ARPO_EVAL_PARQUET="$parquet" \
  ARPO_EVAL_NAME="$name" \
  ARPO_EVAL_TEMP=0.6 \
  ARPO_EVAL_N=16 \
  ARPO_EVAL_MAX_PROMPT=1024 \
  ARPO_EVAL_MAX_RESP=$max_resp \
  ARPO_EVAL_BATCH=32 \
  ARPO_VLLM_GPU_MEMORY_UTIL=0.70 \
  ARPO_VLLM_MAX_NUM_SEQS=64 \
  ARPO_VLLM_MAX_BATCH_TOKENS=16384 \
  ARPO_VALIDATION_DATA_DIR="$val_dir" \
    bash "${SCRIPT_DIR}/eval_val_only.sh" > /tmp/passk_${name}.stdout 2>&1
  local ec=$?

  if [ $ec -ne 0 ]; then
    echo "  EVAL FAILED (ec=$ec); tail of stdout:" | tee -a "$SUMMARY"
    tail -20 /tmp/passk_${name}.stdout | tee -a "$SUMMARY"
  else
    echo "  all pass@k metrics:" | tee -a "$SUMMARY"
    grep -oE "val-core/[^[:space:]]+reward/(best|mean|maj)@[0-9]+[^[:space:]]*:[0-9.]+" \
      "${out}/run.log" 2>/dev/null | sort -u | tee -a "$SUMMARY"
  fi

  ray stop --force >/dev/null 2>&1 || true
  pkill -f 'verl.trainer.main_ppo' 2>/dev/null || true
  sleep 30
}

for bench_row in "${BENCHES[@]}"; do
  IFS='|' read -r bench_label parquet <<< "$bench_row"
  for ckpt_row in "${CKPTS[@]}"; do
    IFS='|' read -r ckpt_label ckpt_path <<< "$ckpt_row"
    run_one "$ckpt_label" "$ckpt_path" "$bench_label" "$parquet"
  done
done

echo ""
echo "===========================================" | tee -a "$SUMMARY"
echo " all pass@k evals done at $(date +%F\ %T)" | tee -a "$SUMMARY"
echo "===========================================" | tee -a "$SUMMARY"
