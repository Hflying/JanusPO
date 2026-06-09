#!/usr/bin/env bash
# ===========================================================================
# T0-B: 7B PagedAdamW32bit   bundle

#   step 1. Convert verl-FSDP ckpt -> HF format (~8 min)
#             plan_c_asymkl_v2_7b_bf16_v1/global_step_233/actor -> hf_step233
#   step 2. GSM8K test.parquet  n=4 t=0.6  eval        (~25 min, 1319  )
#   step 3. MATH-500 test.parquet  n=4 t=0.6  eval     (~15 min,  500  )
#   step 4. Summarize → eval_results/t0b_step233_n4_summary.txt

# Total wallclock: ~50 min  (1 GPU, sequential)

#   paper   8-bit baseline (GSM8K 92.9% / MATH-500 -3.1pp)   protocol 
# Re-run   convert   eval  
# ===========================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"

CKPT_ROOT="${PARENT_DIR}/checkpoints"
EVAL_ROOT="${CKPT_ROOT}/_eval"
SUMMARY_DIR="${PARENT_DIR}/eval_results"
mkdir -p "$SUMMARY_DIR"

GSM8K_PARQUET="${PARENT_DIR}/rl_datasets/gsm8k/test.parquet"
MATH500_PARQUET="${PARENT_DIR}/rl_datasets/math500/test.parquet"

JANUS7B_BF16_VERL="${CKPT_ROOT}/plan_c_asymkl_v2_7b_bf16_v1/global_step_233/actor"
JANUS7B_BF16_HF="${CKPT_ROOT}/plan_c_asymkl_v2_7b_bf16_v1/hf_step233"

EVAL_TEMP=0.6
EVAL_N=4
SUFFIX="_n4"

# 7B vLLM   T0-A 7B  
VLLM_UTIL=0.55
VLLM_MAX_SEQS=32

# ── prerequisite checks ──────────────────────────────────────────────────
for f in "$GSM8K_PARQUET" "$MATH500_PARQUET"; do
  if [ ! -f "$f" ]; then
    echo "[err] missing parquet: $f"; exit 1
  fi
done
if [ ! -d "$JANUS7B_BF16_VERL" ]; then
  echo "[err] verl ckpt missing: $JANUS7B_BF16_VERL"; exit 1
fi

# ── step 1: convert verl -> HF ───────────────────────────────────────────
convert_if_needed() {
  local verl_dir="$1"
  local hf_dir="$2"
  if [ -f "${hf_dir}/config.json" ]; then
    echo "[convert] already exists: ${hf_dir}"; return 0
  fi
  echo ""
  echo "############################################################"
  echo "# [convert] ${verl_dir}"
  echo "#       ->  ${hf_dir}"
  echo "############################################################"
  PYTHONPATH="${PARENT_DIR}/verl_arpo_entropy:${PYTHONPATH:-}" \
    python \
    merge_ckpt/convert_checkpoint_from_verl_to_hf.py merge \
      --backend fsdp \
      --local_dir "$verl_dir" \
      --target_dir "$hf_dir"
}

t_start=$(date +%s)
convert_if_needed "$JANUS7B_BF16_VERL" "$JANUS7B_BF16_HF" || exit 1
t_convert=$(date +%s)

# ── step 2/3:   eval GSM8K → MATH-500 ──────────────────────────────
run_eval() {
  local ckpt="$1"
  local parquet="$2"
  local name="$3"
  local existing_log="${EVAL_ROOT}/${name}/run.log"
  if [ -f "$existing_log" ] && grep -q "val-core/.*reward/mean@" "$existing_log"; then
    local prev=$(grep -oE "'val-core/[^']*reward/mean@[0-9]+':\s*[0-9.]+" "$existing_log" | tail -1)
    echo "[eval] skipping ${name}: existing → ${prev}"
    return 0
  fi
  if [ ! -f "${ckpt}/config.json" ]; then
    echo "[eval] ERROR: ckpt missing config.json: ${ckpt}"; return 1
  fi
  echo ""
  echo "############################################################"
  echo "# [eval] ${name}"
  echo "# ckpt    = ${ckpt}"
  echo "# dataset = ${parquet}"
  echo "############################################################"
  ARPO_EVAL_CKPT="$ckpt" \
  ARPO_EVAL_PARQUET="$parquet" \
  ARPO_EVAL_NAME="$name" \
  ARPO_EVAL_TEMP="$EVAL_TEMP" \
  ARPO_EVAL_N="$EVAL_N" \
  ARPO_VLLM_GPU_MEMORY_UTIL="$VLLM_UTIL" \
  ARPO_VLLM_MAX_NUM_SEQS="$VLLM_MAX_SEQS" \
  bash "${SCRIPT_DIR}/eval_val_only.sh"
}

run_eval "$JANUS7B_BF16_HF" "$GSM8K_PARQUET"   "janus7b_bf16_gsm8k_step233${SUFFIX}"   || exit 1
t_gsm8k=$(date +%s)

run_eval "$JANUS7B_BF16_HF" "$MATH500_PARQUET" "janus7b_bf16_math500_step233${SUFFIX}" || exit 1
t_math500=$(date +%s)

# ── step 4: summary ──────────────────────────────────────────────────────
SUMMARY="${SUMMARY_DIR}/t0b_step233${SUFFIX}_summary.txt"
{
  echo "# T0-B step_233 (PagedAdamW32bit) eval (n=${EVAL_N}, temp=${EVAL_TEMP})"
  echo "# generated: $(date -u +%F_%T)"
  echo "# ckpt:      ${JANUS7B_BF16_HF}"
  echo ""
  echo "# ---- timings ----"
  printf "convert      : %4d s\n" $((t_convert - t_start))
  printf "GSM8K  n=4   : %4d s\n" $((t_gsm8k - t_convert))
  printf "MATH-500 n=4 : %4d s\n" $((t_math500 - t_gsm8k))
  printf "TOTAL        : %4d s  (~%.1f min)\n" $((t_math500 - t_start)) "$(awk -v a=$t_math500 -v b=$t_start 'BEGIN{printf "%.2f", (a-b)/60.0}')"
  echo ""
  echo "# ---- scores ----"
  for name in "janus7b_bf16_gsm8k_step233${SUFFIX}" "janus7b_bf16_math500_step233${SUFFIX}"; do
    log="${EVAL_ROOT}/${name}/run.log"
    if [ ! -f "$log" ]; then
      printf "%-50s  %s\n" "$name" "MISSING LOG"; continue
    fi
    score=$(grep -oE "'val-core/[^']*reward/mean@[0-9]+':\s*[0-9.]+" "$log" | tail -1)
    [ -z "$score" ] && score=$(grep -oE "'val-core/[^']*acc/mean@[0-9]+':\s*[0-9.]+" "$log" | tail -1)
    [ -z "$score" ] && score="not-found"
    printf "%-50s  %s\n" "$name" "$score"
  done
} | tee "$SUMMARY"

echo ""
echo "[done] summary -> $SUMMARY"
