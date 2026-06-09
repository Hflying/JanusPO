#!/usr/bin/env bash
# ============================================================================
# MATH-500 cross-benchmark zero-shot evaluation for all GSM8K-trained 3B RL
# checkpoints. Produces the data for paper §6.3 "Cross-benchmark transfer":
#   row 0 : Qwen2.5-3B-Instruct zero-shot              (reference ceiling)
#   row 1 : GRPO clean step 30    (peak at 74.8% GSM8K)
#   row 2 : DAPO   step 10        (peak at 58.6% GSM8K before collapse)
#   row 3 : DAPO   step 233       (final collapsed policy, 0.2% GSM8K)
#   row 4 : JanusPO step 200      (peak at 84.5% GSM8K, ours)
# Each eval: greedy T=0, n=1, max_resp=1024, batch=64  --> 500 MATH problems
# Uses verl's val_only path, so we rely on the same pipeline as the paper.
# ------- GPU requirement -------
# One eval uses ~40–50 GB at vllm_gpu_util=0.80. We SERIALIZE the evals
# because a DAPO collapsed ckpt will emit only 3–5 tokens per problem and
# run in ~2 min, while a healthy ckpt takes 10–15 min. Total ~60 min.
# ------- Usage -------
#   conda activate arpo
#   bash scripts/eval_math500_all_ckpts.sh
# Outputs:
#   checkpoints/_eval/<EXP>_math500/run.log        per-eval trainer log
#   checkpoints/_eval/math500_all_summary.txt      aggregated summary
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"

SUMMARY="${PARENT_DIR}/checkpoints/_eval/math500_all_summary.txt"
mkdir -p "$(dirname "$SUMMARY")"
date "+=== MATH-500 cross-benchmark eval, started %F %T ===" | tee "$SUMMARY"

MATH500_PARQUET="${PARENT_DIR}/rl_datasets/math500/test.parquet"
[ -f "$MATH500_PARQUET" ] || { echo "FATAL: missing ${MATH500_PARQUET}"; exit 1; }

# Column layout: label | hf ckpt path | expected GSM8K peak (for paper table context)
declare -a ROWS=(
  "qwen3b_zs|$HOME/models/Qwen2.5-3B-Instruct|58.6_zeroshot"
  "grpo_step30|${PARENT_DIR}/checkpoints/grpo_clean_v2/hf_step30|74.8"
  "dapo_step50|${PARENT_DIR}/checkpoints/dapo_3b_gsm8k_v1/hf_step50|12.9_collapsing"
  "dapo_step233|${PARENT_DIR}/checkpoints/dapo_3b_gsm8k_v1/hf_step233|0.2_collapsed"
  "rloo_step50|${PARENT_DIR}/checkpoints/rloo_3b_gsm8k_v1/hf_step50|12.7_collapsed"
  "janus_step200|${PARENT_DIR}/checkpoints/plan_c_asymkl_v2_ub25_v1/hf_step200|84.5"
)

# ── helper: convert verl FSDP ckpt to HF if not already ────────────────────
need_convert() {
  local exp="$1"
  local step="$2"
  local hf_dir="${PARENT_DIR}/checkpoints/${exp}/hf_step${step}"
  local fsdp_dir="${PARENT_DIR}/checkpoints/${exp}/global_step_${step}/actor"

  if [ -d "$hf_dir" ] && [ -f "$hf_dir/config.json" ]; then
    echo "[convert] $hf_dir exists, skip"; return 0
  fi
  [ -d "$fsdp_dir" ] || { echo "FATAL: missing fsdp ckpt $fsdp_dir"; return 1; }

  echo "[convert] FSDP→HF: $fsdp_dir → $hf_dir"
  mkdir -p "$hf_dir"
  PYTHONPATH="${PARENT_DIR}/verl_arpo_entropy:${PYTHONPATH:-}" \
  python3 "${PARENT_DIR}/merge_ckpt/convert_checkpoint_from_verl_to_hf.py" merge \
    --backend fsdp \
    --hf_model_path "$HOME/models/Qwen2.5-3B-Instruct" \
    --local_dir "$fsdp_dir" \
    --target_dir "$hf_dir" 2>&1 | tail -5
}

# Pre-flight: convert FSDP ckpts we need
echo "[step 1/2] converting FSDP checkpoints to HF format (idempotent)" | tee -a "$SUMMARY"
need_convert "grpo_clean_v2"        "30"  || exit 1
need_convert "dapo_3b_gsm8k_v1"     "50"  || exit 1
need_convert "dapo_3b_gsm8k_v1"     "233" || exit 1
# janus already converted under hf_step200

# ── helper: run one eval ───────────────────────────────────────────────────
run_eval() {
  local label="$1"
  local ckpt="$2"
  echo ""
  echo "============================================================" | tee -a "$SUMMARY"
  echo "[eval] ${label}   ckpt=${ckpt}" | tee -a "$SUMMARY"
  echo "============================================================" | tee -a "$SUMMARY"
  if [ ! -d "$ckpt" ] || [ ! -f "$ckpt/config.json" ]; then
    echo "  SKIP: ckpt missing" | tee -a "$SUMMARY"
    return
  fi

  ARPO_EVAL_CKPT="$ckpt" \
  ARPO_EVAL_PARQUET="$MATH500_PARQUET" \
  ARPO_EVAL_NAME="${label}_math500" \
  ARPO_EVAL_TEMP=0.0 \
  ARPO_EVAL_N=1 \
  ARPO_EVAL_MAX_RESP=1024 \
  ARPO_EVAL_MAX_PROMPT=1024 \
  ARPO_EVAL_BATCH=64 \
  ARPO_VLLM_GPU_MEMORY_UTIL=0.75 \
    bash "${SCRIPT_DIR}/eval_val_only.sh" > /tmp/math500_${label}.log 2>&1
  local ec=$?

  local log="${PARENT_DIR}/checkpoints/_eval/${label}_math500/run.log"
  local score
  score=$(grep -E "val/test_score|val-core/.*MATH-lighteval/reward/mean@1|val/.*reward.*mean@1" "$log" 2>/dev/null \
           | tail -5 | head -3)

  if [ -z "$score" ]; then
    echo "  (no numeric line found, tailing run.log)" | tee -a "$SUMMARY"
    tail -20 "$log" 2>/dev/null | tee -a "$SUMMARY"
  else
    printf "  score lines:\n%s\n" "$score" | tee -a "$SUMMARY"
  fi
  [ $ec -ne 0 ] && echo "  WARN: eval exited $ec" | tee -a "$SUMMARY"

  # release GPU before next eval
  ray stop --force >/dev/null 2>&1 || true
  pkill -f 'verl.trainer.main_ppo' 2>/dev/null || true
  sleep 20
}

# ── Phase 2: serialize evals ───────────────────────────────────────────────
echo "[step 2/2] running evals serially"  | tee -a "$SUMMARY"
for row in "${ROWS[@]}"; do
  IFS='|' read -r label ckpt ctx <<< "$row"
  run_eval "$label" "$ckpt"
done

echo ""
echo "================================================================" | tee -a "$SUMMARY"
echo " All MATH-500 evals done at $(date +%F\ %T)"  | tee -a "$SUMMARY"
echo "================================================================" | tee -a "$SUMMARY"
echo "summary in: $SUMMARY"
