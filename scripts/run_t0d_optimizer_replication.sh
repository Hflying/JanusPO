#!/usr/bin/env bash
# ===========================================================================
# T0-D: 7B PagedAdamW {8-bit vs 32-bit} controlled replication — seed=2 pair

#   step 1.  TRAIN 8-bit  seed=2  (~6h, 233 steps; expect NaN ≥ step 90)
#   step 2.  TRAIN fp32   seed=2  (~7-8h, 233 steps; expect stable)
#   step 3.  CONVERT both verl ckpt → HF format
#   step 4.  EVAL  GSM8K + MATH-500 with n=4, T=0.6 for both ckpts (~50 min/ea)
#   step 5.  SUMMARIZE → eval_results/t0d_seed2_summary.txt

# Total wallclock: ~14-16 h (1× GH200, sequential)

# Combined with the existing seed=1 default runs (already in paper §5.4),
# this yields a 2×2 paired (8-bit/fp32) × (seed-1/seed-2) table.

# Re-run safe: existing ckpts and eval logs auto-skipped.
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

EXPNAME_8BIT="plan_c_asymkl_v2_7b_8bit_seed2_v1"
EXPNAME_FP32="plan_c_asymkl_v2_7b_paged32_seed2_v1"

#   ckpt  8-bit   ≥90   NaN regime  peak  
# step 40   step 233   weight   NaN-frozen 
#   NaN  fp32   final ckpt 
EVAL_STEPS_8BIT=("50" "233")
EVAL_STEPS_FP32=("233")

EVAL_TEMP=0.6
EVAL_N=4

VLLM_UTIL=0.55
VLLM_MAX_SEQS=32

# ── prerequisite checks ──────────────────────────────────────────────────
for f in "$GSM8K_PARQUET" "$MATH500_PARQUET"; do
  if [ ! -f "$f" ]; then echo "[err] missing parquet: $f"; exit 1; fi
done
if [ -z "${ARPO_ACTOR_MODEL_PATH:-}" ]; then
  _DEF="${HOME}/models/Qwen2.5-7B-Instruct"
  if [ -d "$_DEF" ] && [ -f "$_DEF/config.json" ]; then
    export ARPO_ACTOR_MODEL_PATH="$_DEF"
  else
    echo "[err]   export ARPO_ACTOR_MODEL_PATH   Qwen2.5-7B-Instruct"; exit 1
  fi
fi

t_start=$(date +%s)

# ── step 1: TRAIN 8-bit seed=2 ─────────────────────────────────────────
EXPDIR_8BIT="${CKPT_ROOT}/${EXPNAME_8BIT}"
if [ -d "${EXPDIR_8BIT}/global_step_233" ]; then
  echo "[train-8bit] already finished (global_step_233 exists), skipping training"
else
  echo ""
  echo "############################################################"
  echo "# [train] 8-bit seed=2  → ${EXPNAME_8BIT}"
  echo "#     ~6h step 90+   NaN regime  paper  "
  echo "############################################################"
  mkdir -p "$EXPDIR_8BIT"
  ARPO_USE_8BIT_ADAMW=1 \
  ARPO_USE_PAGED_ADAMW_32BIT=0 \
  ARPO_SEED=2 \
  ARPO_EXPERIMENT_NAME="$EXPNAME_8BIT" \
  bash "${SCRIPT_DIR}/ARPO_7B_GSM8K_plan_c_asymkl_v2_bf16.sh" \
    2>&1 | tee "${EXPDIR_8BIT}/run.log"
  if [ ! -d "${EXPDIR_8BIT}/global_step_233" ] && [ ! -d "${EXPDIR_8BIT}/global_step_50" ]; then
    echo "[train-8bit]   ckpt  ${EXPDIR_8BIT}/run.log"
    exit 1
  fi
fi
t_train_8bit=$(date +%s)

# ── step 2: TRAIN fp32 seed=2 ──────────────────────────────────────────
EXPDIR_FP32="${CKPT_ROOT}/${EXPNAME_FP32}"
if [ -d "${EXPDIR_FP32}/global_step_233" ]; then
  echo "[train-fp32] already finished (global_step_233 exists), skipping training"
else
  echo ""
  echo "############################################################"
  echo "# [train] fp32 seed=2  → ${EXPNAME_FP32}"
  echo "#     ~7-8h  NaN-free"
  echo "############################################################"
  mkdir -p "$EXPDIR_FP32"
  ARPO_USE_8BIT_ADAMW=0 \
  ARPO_USE_PAGED_ADAMW_32BIT=1 \
  ARPO_SEED=2 \
  ARPO_EXPERIMENT_NAME="$EXPNAME_FP32" \
  bash "${SCRIPT_DIR}/ARPO_7B_GSM8K_plan_c_asymkl_v2_bf16.sh" \
    2>&1 | tee "${EXPDIR_FP32}/run.log"
  if [ ! -d "${EXPDIR_FP32}/global_step_233" ]; then
    echo "[train-fp32]   step 233  ${EXPDIR_FP32}/run.log"
    #   exit  eval   ckpt
  fi
fi
t_train_fp32=$(date +%s)

# ── step 3: CONVERT verl → HF ──────────────────────────────────────────
convert_if_needed() {
  local verl_dir="$1"; local hf_dir="$2"
  if [ -f "${hf_dir}/config.json" ]; then
    echo "[convert] already exists: ${hf_dir}"; return 0
  fi
  if [ ! -d "$verl_dir" ]; then
    echo "[convert] skip: missing ${verl_dir}"; return 1
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

declare -a HF_DIRS=()
declare -a HF_LABELS=()

for s in "${EVAL_STEPS_8BIT[@]}"; do
  verl="${EXPDIR_8BIT}/global_step_${s}/actor"
  hf="${EXPDIR_8BIT}/hf_step${s}"
  if convert_if_needed "$verl" "$hf"; then
    HF_DIRS+=("$hf"); HF_LABELS+=("8bit_seed2_step${s}")
  fi
done
for s in "${EVAL_STEPS_FP32[@]}"; do
  verl="${EXPDIR_FP32}/global_step_${s}/actor"
  hf="${EXPDIR_FP32}/hf_step${s}"
  if convert_if_needed "$verl" "$hf"; then
    HF_DIRS+=("$hf"); HF_LABELS+=("fp32_seed2_step${s}")
  fi
done

t_convert=$(date +%s)

# ── step 4: EVAL GSM8K + MATH-500 (n=4, T=0.6) ──────────────────────────
run_eval() {
  local ckpt="$1"; local parquet="$2"; local name="$3"
  local existing="${EVAL_ROOT}/${name}/run.log"
  if [ -f "$existing" ] && grep -q "val-core/.*reward/mean@" "$existing"; then
    local prev=$(grep -oE "'val-core/[^']*reward/mean@[0-9]+':\s*[0-9.]+" "$existing" | tail -1)
    echo "[eval] skip ${name}: ${prev}"; return 0
  fi
  if [ ! -f "${ckpt}/config.json" ]; then
    echo "[eval] missing ${ckpt}/config.json"; return 1
  fi
  echo ""
  echo "############################################################"
  echo "# [eval] ${name}    (${parquet##*/})"
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

for i in "${!HF_DIRS[@]}"; do
  ck="${HF_DIRS[$i]}"; lab="${HF_LABELS[$i]}"
  run_eval "$ck" "$GSM8K_PARQUET"   "janus7b_${lab}_gsm8k_n4"   || true
  run_eval "$ck" "$MATH500_PARQUET" "janus7b_${lab}_math500_n4" || true
done

t_eval=$(date +%s)

# ── step 5: SUMMARY ──────────────────────────────────────────────────
SUMMARY="${SUMMARY_DIR}/t0d_seed2_summary.txt"
{
  echo "# T0-D: PagedAdamW {8-bit vs fp32} × seed=2 controlled replication"
  echo "# generated: $(date -u +%F_%T)"
  echo ""
  echo "# ---- timings (seconds) ----"
  printf "train 8-bit seed=2 : %5d s  (~%.2f h)\n" $((t_train_8bit - t_start))            "$(awk -v a=$t_train_8bit -v b=$t_start 'BEGIN{printf "%.3f",(a-b)/3600}')"
  printf "train fp32  seed=2 : %5d s  (~%.2f h)\n" $((t_train_fp32 - t_train_8bit))       "$(awk -v a=$t_train_fp32 -v b=$t_train_8bit 'BEGIN{printf "%.3f",(a-b)/3600}')"
  printf "convert ckpts      : %5d s\n"             $((t_convert    - t_train_fp32))
  printf "eval (all ckpts)   : %5d s  (~%.2f h)\n" $((t_eval       - t_convert))         "$(awk -v a=$t_eval -v b=$t_convert 'BEGIN{printf "%.3f",(a-b)/3600}')"
  printf "TOTAL              : %5d s  (~%.2f h)\n" $((t_eval       - t_start))           "$(awk -v a=$t_eval -v b=$t_start 'BEGIN{printf "%.3f",(a-b)/3600}')"
  echo ""
  echo "# ---- eval scores (mean@${EVAL_N}, T=${EVAL_TEMP}) ----"
  for i in "${!HF_DIRS[@]}"; do
    lab="${HF_LABELS[$i]}"
    for ds in gsm8k math500; do
      log="${EVAL_ROOT}/janus7b_${lab}_${ds}_n4/run.log"
      if [ ! -f "$log" ]; then
        printf "  %-40s  MISSING\n" "${lab}_${ds}"; continue
      fi
      score=$(grep -oE "'val-core/[^']*reward/mean@[0-9]+':\s*[0-9.]+" "$log" | tail -1)
      [ -z "$score" ] && score=$(grep -oE "'val-core/[^']*acc/mean@[0-9]+':\s*[0-9.]+" "$log" | tail -1)
      [ -z "$score" ] && score="not-found"
      printf "  %-40s  %s\n" "${lab}_${ds}" "$score"
    done
  done
  echo ""
  echo "# ---- training peak val (mean@1) ----"
  for tag in "${EXPNAME_8BIT}" "${EXPNAME_FP32}"; do
    log="${CKPT_ROOT}/${tag}/run.log"
    if [ ! -f "$log" ]; then echo "  ${tag}  MISSING run.log"; continue; fi
    peak=$(python3 -c "
import re
with open('${log}') as f: txt=f.read()
ms = re.findall(r'step:(\d+)\s.*?val-core/openai/gsm8k/reward/mean@1:([0-9.]+)', txt)
if not ms: print('no val matches'); raise SystemExit
ms = [(int(s), float(v)) for s,v in ms]
best = max(ms, key=lambda x: x[1])
last = ms[-1]
print(f'peak={best[1]*100:.2f}% @ step {best[0]} (final step {last[0]} val={last[1]*100:.2f}%)')
" 2>/dev/null || echo "parse-fail")
    printf "  %-40s  %s\n" "$tag" "$peak"
  done
} | tee "$SUMMARY"

echo ""
echo "[done] summary -> $SUMMARY"
