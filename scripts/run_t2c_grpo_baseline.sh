#!/usr/bin/env bash
# ===========================================================================
# T2-C  multi-turn _vanilla GRPO_ baseline on DeepSearch
#       (gtpo_3b_deepsearch_v8_grpo_paired)
# Same starting checkpoint, data, memory budget AND low-exploration knobs
# as the winning JanusPO run (v8); only the GTPO algorithm components are
# stripped:
#   removed:    AsymClip      → symmetric clip (eps=0.2)
#               AsymKL        → β_pos = β_neg = 1.0
#               entropy_coeff → 0  (already 0 in v8)
#               entropy_upper_bound → unused (sentinel large value)
#               w_t(n)        → uniform
#               H(n)          → off (already off in v8)
# v8 sweet spot: step 50 (HLE reward -0.926 vs SFT -0.976 = 3× format-correct
# rate, HLE f1 0.343% vs SFT 0.298%). After step 100 v8 also degraded
# (response/turn collapse). T2-C should be evaluated at the same step
# windows (25/50/75/100); we expect collapse to happen even earlier
# without AsymKL/AsymClip anchoring policy back to SFT.
# Paper §7.5 Table 3 paired comparison:
#   row 1: SFT only (T2-A, GAIA 0.000% / HLE 0.298% f1 best@4)
#   row 2: SFT + GRPO  (this run, T2-C, eval @ best step in 25-100)
#   row 3: SFT + JanusPO (T2-B v8 step 50, GAIA 0.000% / HLE 0.343%)
# ===========================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"

EXPNAME="${ARPO_EXPERIMENT_NAME:-gtpo_3b_deepsearch_v8_grpo_paired}"
SAVE_PATH="${PARENT_DIR}/checkpoints/${EXPNAME}"

SFT_CKPT="${ARPO_ACTOR_MODEL_PATH:-${PARENT_DIR}/checkpoints/deepsearch_sft_v1/global_step_273}"
if [ ! -f "${SFT_CKPT}/config.json" ]; then
  echo "[err] SFT ckpt missing: ${SFT_CKPT}"; exit 1
fi
export ARPO_ACTOR_MODEL_PATH="${SFT_CKPT}"

# vLLM v1 incompatibilities (see comments in run_t2b_janus_multiturn_fix.sh)
export VLLM_USE_V1=0
unset PYTORCH_CUDA_ALLOC_CONF

# ── memory budget (identical to T2-B for fair comparison) ─────────────────
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

# ── algo knobs: VANILLA GRPO (strip every JanusPO component) ──────────────
export ARPO_KL_LOSS_COEF=0.08      # match v8
export ARPO_ACTOR_LR=5e-7          # match v8 (was 1e-6 in old version → match winning JanusPO)
# symmetric clip (= classic PPO/GRPO)
export ARPO_CLIP_RATIO_POS=0.20
export ARPO_CLIP_RATIO_NEG=0.20
# symmetric β_KL (THIS is the key contrast vs v8 which uses 0.5 / 1.5)
export ARPO_KL_BETA_POS=1.0
export ARPO_KL_BETA_NEG=1.0
# uniform turn weighting
export ARPO_W_T_MODE=uniform
export ARPO_W_T_BETA=1.0
export ARPO_W_T_ALPHA=0.0
# kill entropy regularizers (already off in v8)
export ARPO_ENTROPY_COEFF=0.0
export ARPO_ENTROPY_UPPER_BOUND=99.0      # sentinel: never triggers
# kill H(n) turn-count entropy modulation (off in v8)
export ARPO_USE_HN_ENTROPY=false
export ARPO_HN_ALPHA=0.0

# ── low-exploration knobs to match v8 (so any difference is due to GTPO comps) ──
export ARPO_ROLLOUT_TEMP=0.7              # match v8 (was default 1.0)

# ── runtime knobs ──────────────────────────────────────────────────────────
export ARPO_TEST_FREQ=10
export ARPO_SAVE_FREQ=25
export ARPO_TOTAL_EPOCHS=1
export ARPO_TOOL_CALL_LIMIT=5
export ARPO_USE_MOCK_SEARCH=1
export ARPO_LOG_VAL_GENERATIONS=4

export ARPO_EXPERIMENT_NAME="$EXPNAME"

mkdir -p "$SAVE_PATH"
{
  echo "# T2-C run manifest  $(date -u +%FT%TZ)"
  env | grep -E '^(ARPO_|VLLM_USE_V1=)' | sort
} > "${SAVE_PATH}/manifest.env"
echo "[t2c] manifest -> ${SAVE_PATH}/manifest.env"

cat <<EOF
================================================================
 T2-C vanilla-GRPO multi-turn baseline (paired with T2-B)
  exp:    ${EXPNAME}
  actor:  ${ARPO_ACTOR_MODEL_PATH}
  out:    ${SAVE_PATH}/run.log
  algo:   GRPO  (no AsymClip / AsymKL / entropy / w_t / H(n))
  expects: 'naive' GRPO trajectory; same KL=${ARPO_KL_LOSS_COEF},
           same lr=${ARPO_ACTOR_LR}, same SFT init.
================================================================
EOF

bash "${SCRIPT_DIR}/ARPO_3B_DeepSearch_GTPO.sh"
