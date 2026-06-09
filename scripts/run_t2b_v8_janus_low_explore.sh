#!/usr/bin/env bash
# ===========================================================================
# T2-B v8  multi-turn JanusPO RL on DeepSearch
#         (gtpo_3b_deepsearch_v8_janus)
# Why v8 — Lessons from v6 and v7:
#   v6 (H(n) on, T=1.0, resp 1500, ent 0.005):
#     step 25 eval: reward best@4 -0.981 → -0.935 ✓ (RL working)
#                   f1 best@4    0.0% → 0.0% ✗
#     trace: by step 17 entropy 3.9 → upper-bound 2.5 tripped → AsymKL pulled
#            policy back to SFT distribution → turns collapsed 4.5→1, response
#            length grew 800→1170, clip_ratio 0.12→0.69. Model emitted
#            <answer> tag (reward) but ran out of token budget for content.
#   v7 (H(n) off, T=1.0, resp 2000, ent 0.005, ent_ub 4.0):
#     step 1-6 looked great (clip_ratio 0.04, t/step 113s).
#     step 30+: response_length grew until 96-100% rollouts hit the 2000-token
#     cap → all rewards = -1 → all advantages = 0 → pg_loss = 0 → policy
#     stuck in "infinite reasoning" mode → death spiral. Killed at step 42.
#   Diagnosis: BOTH failures are policy *exploration* killing useful gradient.
#   The SFT init has well-formatted short multi-turn outputs; RL keeps trying
#   to "discover" new modes (longer reasoning, single-turn dump) that all
#   degrade reward extraction. We don't need PPO to find new policies — we
#   need it to *fine-tune* SFT toward higher reward. So: shrink every
#   exploration knob to its minimum.
# v8 strategy: minimize exploration, hug SFT distribution.
#                       v6   →   v7   →  v8
#   max_response_length 1500    2000    1500   ← rollback to v6 (what SFT trained on)
#   tool_call_limit     5       8       5      ← rollback to v6
#   rollout temperature 1.0     1.0     0.7    ← LESS exploration in rollouts
#   entropy_coeff       0.005   0.005   0      ← no entropy bonus, don't reward exploration
#   entropy_upper_bound 2.5     4.0     2.0    ← cap entropy hard
#   use_hn_entropy      true    false   false  ← keep off (didn't help in v6)
#   hn_alpha            1.0     0.0     0.0    ← (consistency)
#   AsymKL β_neg        1.5     1.5     1.5    ← keep — pulls policy back to SFT
#                                                 (this is now a FEATURE, not bug)
#   actor_lr            1e-6    1e-6    5e-7   ← halve LR — slow drift from SFT
#   rollout_n           4       3       4      ← rollback to v6 (more samples for variance)
#   ppo_max_token_per_gpu 4000  4000    4000   ← required >= max_seq 3500
#   save_freq           25      50      25     ← back to early ckpts for fast eval
# Memory budget (single GH200-94GB):
#   v6 with n=4 resp=1500 → reserved 96.97 GB peak (tight but stable)
#   v8 same config + lower temp + ent=0  → expected similar peak (~96 GB)
# Wallclock:
#   v6 was 145-280s/step. v8 with T=0.7 should be slightly FASTER (lower
#   temperature → fewer sampling tokens before EOS). Expect 130-200s/step.
#   250 steps ≈ 9-14h. Save_freq=25 → 10 ckpts in window.
# Output:
#   checkpoints/gtpo_3b_deepsearch_v8_janus/
#     run.log
#     manifest.env
#     global_step_25, _50, _75, _100, ... _250
# Usage:
#   tmux new -s t2b_v8 'bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate arpo && cd ${ARPO_ROOT} && bash scripts/run_t2b_v8_janus_low_explore.sh"'
# ===========================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"

EXPNAME="${ARPO_EXPERIMENT_NAME:-gtpo_3b_deepsearch_v8_janus}"
SAVE_PATH="${PARENT_DIR}/checkpoints/${EXPNAME}"

SFT_CKPT="${ARPO_ACTOR_MODEL_PATH:-${PARENT_DIR}/checkpoints/deepsearch_sft_v1/global_step_273}"
if [ ! -f "${SFT_CKPT}/config.json" ]; then
  echo "[err] SFT ckpt missing: ${SFT_CKPT}"; exit 1
fi
export ARPO_ACTOR_MODEL_PATH="${SFT_CKPT}"

export VLLM_USE_V1=0
unset PYTORCH_CUDA_ALLOC_CONF

# ── Sequence + batch (rollback to v6 sizing) ─────────────────────────────
export ARPO_MAX_PROMPT_LENGTH=2000
export ARPO_MAX_RESPONSE_LENGTH=1500          # rollback v7→v6
export ARPO_TRAIN_BATCH_SIZE=8
export ARPO_PPO_MINI_BATCH_SIZE=8
export ARPO_ROLLOUT_N=4                        # rollback v7→v6 (more samples)
export ARPO_INITIAL_ROLLOUTS=2
export ARPO_BEAM_SIZE=1
export ARPO_BRANCH_PROBABILITY=0.5

export ARPO_VLLM_GPU_MEMORY_UTIL=0.25
export ARPO_VLLM_MAX_NUM_SEQS=16
export ARPO_VLLM_MAX_BATCH_TOKENS=8000

export ARPO_ACTOR_PARAM_OFFLOAD=1
export ARPO_OPTIMIZER_OFFLOAD=1
# must be >= max_prompt+max_response = 3500
export ARPO_PPO_MAX_TOKEN_LEN_PER_GPU=4000

# ── Algo: keep AsymClip + AsymKL (JanusPO core), kill all exploration ────
export ARPO_KL_LOSS_COEF=0.08
export ARPO_ACTOR_LR=5e-7                      # half of v6 (1e-6) — slow drift
export ARPO_CLIP_RATIO_POS=0.20
export ARPO_CLIP_RATIO_NEG=0.30
export ARPO_KL_BETA_POS=0.5
export ARPO_KL_BETA_NEG=1.5                    # KEEP strong — pulls policy back to SFT
export ARPO_W_T_MODE=exp_decay
export ARPO_W_T_BETA=0.9
export ARPO_W_T_ALPHA=0.5

# ── KILL exploration knobs ───────────────────────────────────────────────
export ARPO_ROLLOUT_TEMP=0.7                   # was 1.0 — sharper sampling, shorter outputs
export ARPO_ENTROPY_COEFF=0.0                  # was 0.005 — no entropy bonus
export ARPO_ENTROPY_UPPER_BOUND=2.0            # was 2.5 / 4.0 — hard cap
export ARPO_USE_HN_ENTROPY=false               # keep off
export ARPO_HN_ALPHA=0.0

# ── Runtime ───────────────────────────────────────────────────────────────
export ARPO_TEST_FREQ=10
export ARPO_SAVE_FREQ=25                       # back to early ckpts
export ARPO_TOTAL_EPOCHS=1
export ARPO_TOOL_CALL_LIMIT=5                  # rollback v7→v6
export ARPO_USE_MOCK_SEARCH=1
export ARPO_LOG_VAL_GENERATIONS=4

export ARPO_EXPERIMENT_NAME="$EXPNAME"

mkdir -p "$SAVE_PATH"
{
  echo "# T2-B v8 run manifest  $(date -u +%FT%TZ)"
  echo "# parent: v6 (turn-collapse + f1=0) and v7 (response runaway, score=0 by step 30)"
  echo "# strategy: minimize exploration, hug SFT distribution; let JanusPO fine-tune only"
  env | grep -E '^(ARPO_|VLLM_USE_V1=)' | sort
} > "${SAVE_PATH}/manifest.env"
echo "[t2b_v8] manifest -> ${SAVE_PATH}/manifest.env"

cat <<EOF
================================================================
 T2-B v8 JanusPO multi-turn RL  (low-exploration, hug SFT)
  exp:    ${EXPNAME}
  actor:  ${ARPO_ACTOR_MODEL_PATH}
  out:    ${SAVE_PATH}/run.log
  changes vs v7:
    response_length 2000 → 1500 (rollback)
    rollout_temp     1.0 → 0.7  (less sampling spread)
    entropy_coeff   0.005→ 0    (no exploration bonus)
    entropy_upper    4.0 → 2.0  (hard cap)
    actor_lr         1e-6→ 5e-7 (slow drift)
    tool_call_limit  8   → 5    (rollback)
    rollout_n        3   → 4    (rollback)
    save_freq        50  → 25   (early eval)
================================================================
EOF

bash "${SCRIPT_DIR}/ARPO_3B_DeepSearch_GTPO.sh"
