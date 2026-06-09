#!/usr/bin/env bash
# ===========================================================================
# T2-B  multi-turn JanusPO RL on DeepSearch (gtpo_3b_deepsearch_v6_janus)
# Background:
#   v5 OOM'd on step-1 backward (3.0B / single GH200-94GB). The fix below
#   trades 2x rollouts (n=8 → 4) and a tighter vLLM KV cache for headroom
#   for backward + activations. All other algorithm components stay
#   identical to ARPO_3B_DeepSearch_GTPO.sh (AsymClip + AsymKL + entropy
#   coeff + entropy_upper_bound + w_t(n) + H(n)).
#   v5 (OOM)         | v6_janus (this run)
#   ----------------+-------------------
#   gpu_mem_util 0.45 | 0.25            ← vLLM frees ~19GB for actor backward
#   max_resp 2000     | 1500            ← match script default; cuts activation
#   ppo_max_token 8000| 4000            ← halve micro-batch size
#   rollout n 8       | 4               ← halve total tokens per step
#   Step 1 expected memory: ~50–55 GB allocated (vs 72 GB on v5)
# Wallclock budget: ~30s/step × 500 steps ≈ 4 h (1× GH200, mock search)
#                   real Bing search is ~2–3× slower
# Output:
#   checkpoints/gtpo_3b_deepsearch_v6_janus/
#     run.log
#     global_step_25, _50, ...   (verl FSDP ckpts; convert to HF for eval)
# Quick sanity:
#   bash scripts/run_t2b_janus_multiturn_fix.sh
#   # then in another shell:
#   tail -F checkpoints/gtpo_3b_deepsearch_v6_janus/run.log | grep -E "step:|val-core"
# ===========================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"

EXPNAME="${ARPO_EXPERIMENT_NAME:-gtpo_3b_deepsearch_v6_janus}"
SAVE_PATH="${PARENT_DIR}/checkpoints/${EXPNAME}"

# Use the SFT warm-up checkpoint as the actor starting point.
SFT_CKPT="${ARPO_ACTOR_MODEL_PATH:-${PARENT_DIR}/checkpoints/deepsearch_sft_v1/global_step_273}"
if [ ! -f "${SFT_CKPT}/config.json" ]; then
  echo "[err] SFT ckpt missing: ${SFT_CKPT}"; exit 1
fi
export ARPO_ACTOR_MODEL_PATH="${SFT_CKPT}"

# vLLM v1's CuMemAllocator (a) bans expandable_segments and (b) refuses
# gpu_memory_utilization < ~0.30 for 3B models on 94GB GPUs ("No available
# memory for the cache blocks"). v0 path is more lenient; the eval script
# already uses VLLM_USE_V1=0 successfully.
export VLLM_USE_V1=0
unset PYTORCH_CUDA_ALLOC_CONF

# ── memory-tight overrides (vs default ARPO_3B_DeepSearch_GTPO.sh) ─────────
export ARPO_MAX_PROMPT_LENGTH=2000
export ARPO_MAX_RESPONSE_LENGTH=1500
export ARPO_TRAIN_BATCH_SIZE=8
export ARPO_PPO_MINI_BATCH_SIZE=8
export ARPO_ROLLOUT_N=4
export ARPO_INITIAL_ROLLOUTS=2
export ARPO_BEAM_SIZE=1
export ARPO_BRANCH_PROBABILITY=0.5

export ARPO_VLLM_GPU_MEMORY_UTIL=0.25
export ARPO_VLLM_MAX_NUM_SEQS=16
export ARPO_VLLM_MAX_BATCH_TOKENS=8000

export ARPO_ACTOR_PARAM_OFFLOAD=1
export ARPO_OPTIMIZER_OFFLOAD=1
export ARPO_PPO_MAX_TOKEN_LEN_PER_GPU=4000

# ── algo knobs (same as champion 3B GSM8K + multi-turn H(n)) ───────────────
export ARPO_KL_LOSS_COEF=0.08
export ARPO_ACTOR_LR=1e-6
export ARPO_CLIP_RATIO_POS=0.20
export ARPO_CLIP_RATIO_NEG=0.30
export ARPO_KL_BETA_POS=0.5
export ARPO_KL_BETA_NEG=1.5
export ARPO_W_T_MODE=exp_decay
export ARPO_W_T_BETA=0.9
export ARPO_W_T_ALPHA=0.5
export ARPO_ENTROPY_COEFF=0.005
export ARPO_ENTROPY_UPPER_BOUND=2.5
export ARPO_USE_HN_ENTROPY=true
export ARPO_HN_ALPHA=1.0

# ── runtime knobs ──────────────────────────────────────────────────────────
export ARPO_TEST_FREQ=10
export ARPO_SAVE_FREQ=25
export ARPO_TOTAL_EPOCHS=1
export ARPO_TOOL_CALL_LIMIT=5
export ARPO_USE_MOCK_SEARCH=1   # offline-safe; flip to 0 for real Bing
export ARPO_LOG_VAL_GENERATIONS=4

# patch ppo_max_token_len_per_gpu via sed-injected env (the underlying script
# computes it as 2*(prompt+response)). We instead wrap with a custom hydra
# override appended after the upstream call by writing a thin local override.
export ARPO_EXPERIMENT_NAME="$EXPNAME"

mkdir -p "$SAVE_PATH"

# Drop a manifest of all ARPO_* env vars next to the run for reproducibility.
{
  echo "# T2-B run manifest  $(date -u +%FT%TZ)"
  env | grep -E '^(ARPO_|PYTORCH_CUDA_ALLOC_CONF=)' | sort
} > "${SAVE_PATH}/manifest.env"
echo "[t2b] manifest -> ${SAVE_PATH}/manifest.env"

echo ""
echo "================================================================"
echo " T2-B JanusPO multi-turn RL  (memory-tight, single GH200)"
echo "  exp:    ${EXPNAME}"
echo "  actor:  ${ARPO_ACTOR_MODEL_PATH}"
echo "  out:    ${SAVE_PATH}/run.log"
echo "  vLLM:   util=${ARPO_VLLM_GPU_MEMORY_UTIL}  max_seqs=${ARPO_VLLM_MAX_NUM_SEQS}"
echo "  rollout n=${ARPO_ROLLOUT_N}  resp_len=${ARPO_MAX_RESPONSE_LENGTH}"
echo "================================================================"

bash "${SCRIPT_DIR}/ARPO_3B_DeepSearch_GTPO.sh"
