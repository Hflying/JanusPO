#!/usr/bin/env bash
# ============================================================================
# 7B DAPO collapse reproduction on Qwen2.5-7B-Instruct + GSM8K.
# Based on ARPO_7B_GSM8K_plan_c_asymkl_v2.sh (which has the memory
# engineering needed to fit 7B + FSDP + vLLM on a single 96 GB GH200).
# The only CHANGES vs that baseline are the DAPO-defining knobs:
#   ARPO_USE_ASYMCLIP=1  CLIP_RATIO_POS=0.20  CLIP_RATIO_NEG=0.28  (DAPO §3.2)
#   ARPO_USE_ASYMKL=0   (KL regularizer OFF per DAPO §3.4)
#   KL_LOSS_COEF=0      (redundant with use_kl_loss=False but set both)
#   ENTROPY_COEFF=0     (no Plan-C entropy guard)
#   ENTROPY_UPPER_BOUND=999 (effectively disabled)
#   REWARD_MANAGER="naive"  (no turn-weighting)
#   TOTAL_STEPS=100         (enough to observe peak + collapse onset)
# Expected (pre-registered) outcome: if DAPO collapse is scale-independent,
# we should see peak val_acc ≈ SFT init (~86%) at step 10-20, then response
# length dropping < 20 tokens, then val_acc < 5% by step ~60-80.
# Usage:
#   conda activate arpo
#   bash scripts/ARPO_7B_GSM8K_dapo.sh
# ============================================================================
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"

# Override all DAPO-defining knobs, then delegate to the 7B infra script.
export ARPO_EXPERIMENT_NAME="${ARPO_EXPERIMENT_NAME:-dapo_7b_gsm8k_v1}"
export ARPO_USE_ASYMCLIP=1
export ARPO_CLIP_RATIO_POS=0.20
export ARPO_CLIP_RATIO_NEG=0.28
export ARPO_USE_ASYMKL=0
export ARPO_KL_LOSS_COEF=0.0
export ARPO_USE_KL_LOSS=False
export ARPO_ENTROPY_COEFF=0.0
export ARPO_ENTROPY_UPPER_BOUND=999
export ARPO_ENTROPY_WEIGHT=0.0
# Keep memory settings from the proven JanusPO 7B run:
#   use_8bit_adamw=1, param_offload=1, optimizer_offload=1, vllm_util=0.20,
#   max_num_seqs=32, max_batch_tokens=4096
export ARPO_USE_8BIT_ADAMW="${ARPO_USE_8BIT_ADAMW:-1}"
export ARPO_ACTOR_PARAM_OFFLOAD="${ARPO_ACTOR_PARAM_OFFLOAD:-1}"
export ARPO_OPTIMIZER_OFFLOAD="${ARPO_OPTIMIZER_OFFLOAD:-1}"
export ARPO_SAVE_FREQ="${ARPO_SAVE_FREQ:-20}"
export ARPO_TEST_FREQ="${ARPO_TEST_FREQ:-10}"
export ARPO_TOTAL_EPOCHS=1
# Only train for ~100 steps; enough to observe collapse trajectory
export ARPO_TOTAL_STEPS="${ARPO_TOTAL_STEPS:-100}"

# Flip reward manager from gtpo (turn-weighted) → naive (one-shot reward).
# We do this by editing the trainer command line via env var injection.
# The parent 7B script uses REWARD_MANAGER="gtpo" hardcoded; we patch via
# a wrapper: run the same python module invocation but override reward_manager.
# Simplest: set an env var that the parent script reads, or better, run the
# parent script with one small override. The parent script reads
# REWARD_MANAGER from an internal variable, so we expose an env hook.
export ARPO_REWARD_MANAGER="${ARPO_REWARD_MANAGER:-naive}"

echo "================================================================"
echo " 7B DAPO collapse reproduction (100 steps)"
echo "   AsymClip: $ARPO_CLIP_RATIO_POS / $ARPO_CLIP_RATIO_NEG"
echo "   KL off, entropy guard off, reward_manager=$ARPO_REWARD_MANAGER"
echo "   8-bit AdamW, param_offload, vllm_util=0.20"
echo "================================================================"
bash "${SCRIPT_DIR}/ARPO_7B_GSM8K_plan_c_asymkl_v2.sh"
