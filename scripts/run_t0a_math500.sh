#!/usr/bin/env bash
# ===========================================================================
# T0-A: MATH-500 zero-shot transfer eval bundle

#   1. Convert verl-FSDP ckpts to HF format (if not yet converted):
#        - 3B champion: plan_c_asymkl_v2_ub25_v1/global_step_200/actor -> hf_step200
#        - 7B champion: plan_c_asymkl_v2_7b_v1/global_step_50/actor    -> hf_step50
#   2. Run val-only eval on MATH-500 for 4 models:
#        (a) Qwen2.5-3B-Instruct  (zero-shot)
#        (b) 3B JanusPO step 200
#        (c) Qwen2.5-7B-Instruct  (zero-shot)
#        (d) 7B JanusPO step 50
#   3. Summarize scores to eval/math500_t0a_summary.txt

# Expected total wallclock: ~45 min for 4× 500-prompt greedy decodes on GH200.

# Safety: exits on first failure; re-run can resume past already-done steps.
# ===========================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"

CKPT_ROOT="${PARENT_DIR}/checkpoints"
MATH500_PARQUET="${PARENT_DIR}/rl_datasets/math500/test.parquet"
SUMMARY_DIR="${PARENT_DIR}/eval_results"
mkdir -p "$SUMMARY_DIR"

# ── Eval mode ────────────────────────────────────────────────────────────
# greedy  : n=1, temp=0.0, suffix=""         
# n4      : n=4, temp=0.6, suffix="_n4"      
EVAL_MODE="${ARPO_EVAL_MODE:-greedy}"
case "$EVAL_MODE" in
  greedy) EVAL_N=1; EVAL_TEMP=0.0; NAME_SUFFIX="" ;;
  n4)     EVAL_N=4; EVAL_TEMP=0.6; NAME_SUFFIX="_n4" ;;
  *) echo "unknown ARPO_EVAL_MODE=$EVAL_MODE (must be greedy|n4)"; exit 2 ;;
esac
echo "[mode] ${EVAL_MODE}  (n=${EVAL_N}, temp=${EVAL_TEMP}, suffix='${NAME_SUFFIX}')"

# Models (edit paths if yours differ)
QWEN3B="${ARPO_QWEN25_3B:-$HOME/models/Qwen2.5-3B-Instruct}"
QWEN7B="${ARPO_QWEN25_7B:-$HOME/models/Qwen2.5-7B-Instruct}"
JANUS3B_VERL="${CKPT_ROOT}/plan_c_asymkl_v2_ub25_v1/global_step_200/actor"
JANUS3B_HF="${CKPT_ROOT}/plan_c_asymkl_v2_ub25_v1/hf_step200"
JANUS7B_VERL="${CKPT_ROOT}/plan_c_asymkl_v2_7b_v1/global_step_50/actor"
JANUS7B_HF="${CKPT_ROOT}/plan_c_asymkl_v2_7b_v1/hf_step50"

if [ ! -f "$MATH500_PARQUET" ]; then
  echo "[prep] MATH-500 parquet missing, generating now..."
  python rl_datasets/download_math500.py
fi

convert_if_needed() {
  local verl_dir="$1"
  local hf_dir="$2"
  if [ -f "${hf_dir}/config.json" ]; then
    echo "[convert] already exists: ${hf_dir}"
    return 0
  fi
  if [ ! -d "$verl_dir" ]; then
    echo "[convert] ERROR: verl ckpt dir missing: ${verl_dir}"
    return 1
  fi
  echo "[convert] ${verl_dir}  ->  ${hf_dir}"
  PYTHONPATH="${PARENT_DIR}/verl_arpo_entropy:${PYTHONPATH:-}" \
    python \
    merge_ckpt/convert_checkpoint_from_verl_to_hf.py merge \
      --backend fsdp \
      --local_dir "$verl_dir" \
      --target_dir "$hf_dir"
}

# ── Step 1: conversions ──────────────────────────────────────────────────
convert_if_needed "$JANUS3B_VERL" "$JANUS3B_HF" || exit 1
convert_if_needed "$JANUS7B_VERL" "$JANUS7B_HF" || exit 1

# ── Step 2: run 4 evals ( ) ───────────────────────────────────────────
run_eval() {
  local ckpt="$1"
  local name="$2"
  local size="${3:-3b}"   # 3b | 7b
  local existing_log="${CKPT_ROOT}/_eval/${name}/run.log"
  if [ ! -f "${ckpt}/config.json" ]; then
    echo "[eval] skipping ${name}: ckpt missing (${ckpt})"
    return 0
  fi
  if [ -f "$existing_log" ] && grep -q "val-core/.*/reward/mean@" "$existing_log"; then
    local prev_score=$(grep -oE "'val-core/[^']*reward/mean@[0-9]+':\s*[0-9.]+" "$existing_log" | tail -1)
    echo "[eval] skipping ${name}: existing result → ${prev_score}"
    return 0
  fi
  echo ""
  echo "############################################################"
  echo "# [eval] ${name}   (size=${size})"
  echo "# ckpt = ${ckpt}"
  echo "############################################################"
  # 7B   vLLM util   batch  KV cache OOM
  local vllm_util=0.80
  local vllm_max_seqs=128
  if [ "$size" = "7b" ]; then
    vllm_util=0.55
    vllm_max_seqs=32
  fi
  ARPO_EVAL_CKPT="$ckpt" \
  ARPO_EVAL_PARQUET="$MATH500_PARQUET" \
  ARPO_EVAL_NAME="$name" \
  ARPO_EVAL_TEMP="$EVAL_TEMP" \
  ARPO_EVAL_N="$EVAL_N" \
  ARPO_VLLM_GPU_MEMORY_UTIL="$vllm_util" \
  ARPO_VLLM_MAX_NUM_SEQS="$vllm_max_seqs" \
  bash "${SCRIPT_DIR}/eval_val_only.sh"
}

run_eval "$QWEN3B"     "qwen3b_zs_math500${NAME_SUFFIX}"          3b
run_eval "$JANUS3B_HF" "janus3b_math500_step200${NAME_SUFFIX}"    3b
run_eval "$QWEN7B"     "qwen7b_zs_math500${NAME_SUFFIX}"          7b
run_eval "$JANUS7B_HF" "janus7b_math500_step50${NAME_SUFFIX}"     7b

# ── Step 3: Summarize ────────────────────────────────────────────────────
SUMMARY="${SUMMARY_DIR}/math500_t0a_summary${NAME_SUFFIX}.txt"
{
  echo "# MATH-500 T0-A results (mode=${EVAL_MODE}, n=${EVAL_N}, temp=${EVAL_TEMP})"
  echo "# generated: $(date -u +%F_%T)"
  for name in "qwen3b_zs_math500${NAME_SUFFIX}" "janus3b_math500_step200${NAME_SUFFIX}" "qwen7b_zs_math500${NAME_SUFFIX}" "janus7b_math500_step50${NAME_SUFFIX}"; do
    log="${CKPT_ROOT}/_eval/${name}/run.log"
    if [ ! -f "$log" ]; then
      printf "%-40s  %s\n" "$name" "MISSING LOG"
      continue
    fi
    # verl   val_reward_fn   dict   'val-core/.../reward/mean@1': 0.xxx
    score=$(grep -oE "'val-core/[^']*reward/mean@[0-9]+':\s*[0-9.]+" "$log" | tail -1 \
            || grep -oE "'val-core/[^']*acc/mean@[0-9]+':\s*[0-9.]+"    "$log" | tail -1 \
            || echo "not-found")
    printf "%-40s  %s\n" "$name" "$score"
  done
} | tee "$SUMMARY"

echo ""
echo "[done] summary -> $SUMMARY"
