#!/usr/bin/env bash
# ===========================================================================
# T2-B v7  multi-turn JanusPO RL on DeepSearch
#         (gtpo_3b_deepsearch_v7_janus)
# Why v7 (post-mortem of v6_janus step_25 eval):
#   v6 RL was working (best@4 reward -0.981 → -0.935 on GAIA, -0.976 → -0.949
#   on HLE), but f1 stayed at 0% because 50–69% of rollouts hit the 1500-token
#   response cap before they could emit <answer>; the model also collapsed
#   from hn_turns_mean ≈ 4.5 → 1.0 because the H(n) reward weighting
#   (h(t)=(t/n)^α with α=1.0) implicitly punishes early turns and pushes
#   all reasoning into the first single ultra-long turn — which then gets
#   truncated. Two upper-bound issues conspired to lock content learning.
# Fixes (all six are within memory budget after re-balancing rollouts):
#                  v6  →  v7   reason
#   ── responses ─────────────────────────────────────────────────────────
#   max_response_length     1500 → 2000     ↓clip_ratio (was 0.69)
#   tool_call_limit         5    → 8        more room for multi-turn
#   ── algo ──────────────────────────────────────────────────────────────
#   use_hn_entropy          true → false    stop biasing toward turn-1
#   hn_alpha                1.0  → 0.0      (consistency)
#   entropy_upper_bound     2.5  → 4.0      v6 entropy reached 3.9; bound was tripping
#   ── budget rebalance to fit longer responses on a single GH200-94GB ──
#   rollout_n               4    → 3        rollouts × seq tokens stays roughly equal
#   ppo_max_token_len_per_gpu 4000 → 4000   forced to == max_seq (prompt+resp);
#                                            verl asserts max_token_len >= max_seq
#   save_freq               25   → 50       36GB ckpt write costs ~5min/save
# Memory math (single GH200-94GB, vLLM mem_util=0.25 → ~24GB cache):
#   v6 step 19 (resp 679, n=4):     allocated 92.7GB / reserved 97.7GB  ← OOM at step 27
#   v7 expected:                    similar peak (longer seq but n=3 + lower ppo_max_token)
# Wallclock budget:
#   v6 was ~145–280s/step depending on response length.
#   v7 with 2000-token cap + n=3 expected ~200–320s/step;
#   250 steps ≈ 14–22h. Save_freq=50 → 5 ckpts in that window.
# Output:
#   checkpoints/gtpo_3b_deepsearch_v7_janus/
#     run.log
#     manifest.env
#     global_step_50, _100, _150, _200, _250
# Usage:
#   tmux new -s t2b_v7 'bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate arpo && cd ${ARPO_ROOT} && bash scripts/run_t2b_v7_janus_fix.sh"'
# ===========================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"

EXPNAME="${ARPO_EXPERIMENT_NAME:-gtpo_3b_deepsearch_v7_janus}"
SAVE_PATH="${PARENT_DIR}/checkpoints/${EXPNAME}"

# Start from the SFT warm-up checkpoint, not from v6 (we want a clean
# experiment so paper §7.5 row 3 is comparable to row 2).
SFT_CKPT="${ARPO_ACTOR_MODEL_PATH:-${PARENT_DIR}/checkpoints/deepsearch_sft_v1/global_step_273}"
if [ ! -f "${SFT_CKPT}/config.json" ]; then
  echo "[err] SFT ckpt missing: ${SFT_CKPT}"; exit 1
fi
export ARPO_ACTOR_MODEL_PATH="${SFT_CKPT}"

# vLLM v1 incompatibilities (see comments in run_t2b_janus_multiturn_fix.sh)
export VLLM_USE_V1=0
unset PYTORCH_CUDA_ALLOC_CONF

# ── Memory: longer responses, fewer rollouts to compensate ───────────────
export ARPO_MAX_PROMPT_LENGTH=2000
export ARPO_MAX_RESPONSE_LENGTH=2000        # +33% vs v6 (was 1500)
export ARPO_TRAIN_BATCH_SIZE=8
export ARPO_PPO_MINI_BATCH_SIZE=8
export ARPO_ROLLOUT_N=3                      # -25% vs v6 (was 4)
export ARPO_INITIAL_ROLLOUTS=2
export ARPO_BEAM_SIZE=1
export ARPO_BRANCH_PROBABILITY=0.5

export ARPO_VLLM_GPU_MEMORY_UTIL=0.25
export ARPO_VLLM_MAX_NUM_SEQS=12             # was 16; matches lower n
export ARPO_VLLM_MAX_BATCH_TOKENS=8000

export ARPO_ACTOR_PARAM_OFFLOAD=1
export ARPO_OPTIMIZER_OFFLOAD=1
export ARPO_PPO_MAX_TOKEN_LEN_PER_GPU=4000   # must be >= max_prompt+max_response = 4000

# ── Algo: same JanusPO core, but kill the H(n) bias that collapsed turns──
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
export ARPO_ENTROPY_UPPER_BOUND=4.0          # was 2.5; v6 hit 3.9 by step 17
export ARPO_USE_HN_ENTROPY=false             # was true; H(n) collapsed turns→1
export ARPO_HN_ALPHA=0.0

# ── Runtime ───────────────────────────────────────────────────────────────
export ARPO_TEST_FREQ=10
export ARPO_SAVE_FREQ=50                     # was 25; 36GB write costs ~5min
export ARPO_TOTAL_EPOCHS=1
export ARPO_TOOL_CALL_LIMIT=8                # was 5
export ARPO_USE_MOCK_SEARCH=1
export ARPO_LOG_VAL_GENERATIONS=4

export ARPO_EXPERIMENT_NAME="$EXPNAME"

mkdir -p "$SAVE_PATH"
{
  echo "# T2-B v7 run manifest  $(date -u +%FT%TZ)"
  echo "# parent: gtpo_3b_deepsearch_v6_janus (OOM @ step 27, eval @ step_25 showed reward↑ but f1=0% due to truncation + turn collapse)"
  env | grep -E '^(ARPO_|VLLM_USE_V1=)' | sort
} > "${SAVE_PATH}/manifest.env"
echo "[t2b_v7] manifest -> ${SAVE_PATH}/manifest.env"

cat <<EOF
================================================================
 T2-B v7 JanusPO multi-turn RL  (post-mortem fixes for v6)
  exp:    ${EXPNAME}
  actor:  ${ARPO_ACTOR_MODEL_PATH}
  out:    ${SAVE_PATH}/run.log
  fixes vs v6:
    response_length 1500 → 2000  (clip_ratio↓)
    tool_call_limit  5   → 8     (multi-turn room↑)
    H(n) entropy    on   → off   (turn-1 collapse fix)
    entropy_upper   2.5  → 4.0   (was tripping)
    rollout_n        4   → 3     (memory)
    ppo_max_token   4000 → 3500  (memory)
    save_freq       25   → 50    (I/O cost)
================================================================
EOF

bash "${SCRIPT_DIR}/ARPO_3B_DeepSearch_GTPO.sh"
