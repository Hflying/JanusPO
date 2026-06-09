#!/usr/bin/env bash
# ============================================================================
# JanusPO on Qwen2.5-3B-Instruct + MATH (lighteval), single-turn math reasoning.
# Cross-domain validation of the JanusPO recipe that won GSM8K (84.5% best
# val acc, plan_c_asymkl_v2_ub25_v1).

# Identical recipe to the GSM8K winner:
#   • AsymClip:        ε_+=0.20  ε_-=0.30
#   • AsymKL:          β_pos=0.5  β_neg=1.5  (kl_loss_coef=0.08, low_var_kl)
#   • Entropy reg:     entropy_coeff=0.005, entropy_upper_bound=2.5
#   • Reward manager:  gtpo with w_t_mode=exp_decay, w_t_beta=0.9, α=0.5
#   • lr 1e-6, rollout_n=8, T=0.9, batch=32, 1 epoch

# Eval set: MATH-500 (rl_datasets/math500/test.parquet, 500 problems)
# Train set: MATH-lighteval train (~7500 problems) — auto-prepared by
# scripts/prep_math_train.sh if missing.

# Usage:
#   conda activate arpo
#   export ARPO_ACTOR_MODEL_PATH="$HOME/models/Qwen2.5-3B-Instruct"
#   bash scripts/ARPO_3B_MATH_janus.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"

if [ -f "${SCRIPT_DIR}/local.env" ]; then
  set -a; source "${SCRIPT_DIR}/local.env"; set +a
fi

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
ARPO_SEED="${ARPO_SEED:-1}"
export PYTHONHASHSEED="${ARPO_SEED}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export RAY_memory_usage_threshold=0.8
export RAY_memory_monitor_refresh_ms=0

if python3 -c "import xformers" 2>/dev/null; then
  export VLLM_ATTENTION_BACKEND=XFORMERS
fi
export PYTHONPATH="${PARENT_DIR}/verl_arpo_entropy:${PYTHONPATH:-}"

if ! python3 -c "import ray" 2>/dev/null; then
  echo "error:   conda activate arpo"; exit 1
fi
if [ "${ARPO_ENABLE_TORCH_COMPILE:-0}" != "1" ]; then
  export TORCHDYNAMO_DISABLE=1
fi

PROJECT_NAME="math_janus"
EXPERIMENT_NAME="${ARPO_EXPERIMENT_NAME:-janus_3b_math_v1}"
RESUME_MODE="${ARPO_RESUME_MODE:-disable}"
CONFIG_PATH="${PARENT_DIR}/scripts/config"
CONFIG_NAME="ppo_trainer.yaml"

NNODES=1
N_GPUS_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')
[ "$N_GPUS_PER_NODE" -ge 1 ] || N_GPUS_PER_NODE=1

# MATH problems are longer than GSM8K (~700 tok prompts, longer solutions)
MAX_PROMPT_LENGTH="${ARPO_MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${ARPO_MAX_RESPONSE_LENGTH:-1024}"
TRAIN_BATCH_SIZE="${ARPO_TRAIN_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${ARPO_PPO_MINI_BATCH_SIZE:-32}"
ARPO_VLLM_GPU_MEMORY_UTIL="${ARPO_VLLM_GPU_MEMORY_UTIL:-0.80}"
ARPO_VLLM_MAX_NUM_SEQS="${ARPO_VLLM_MAX_NUM_SEQS:-48}"
ARPO_VLLM_MAX_BATCH_TOKENS="${ARPO_VLLM_MAX_BATCH_TOKENS:-12288}"
TEST_FREQ="${ARPO_TEST_FREQ:-10}"

# ── Data: MATH train + MATH-500 test ────────────────────────────────────────
MATH_DIR="${PARENT_DIR}/rl_datasets/math"
TRAIN_FILES="${MATH_DIR}/train.parquet"
VALID_FILES="${PARENT_DIR}/rl_datasets/math500/test.parquet"

if [ ! -f "$TRAIN_FILES" ]; then
  echo "MATH train not found, attempting auto-prepare..."
  bash "${SCRIPT_DIR}/prep_math_train.sh" || { echo "auto-prepare failed; aborting"; exit 1; }
fi
[ -f "$VALID_FILES" ] || { echo "  MATH-500 test: python3 ${PARENT_DIR}/rl_datasets/download_math500.py"; exit 1; }

ACTOR_MODEL_PATH="${ARPO_ACTOR_MODEL_PATH:-$HOME/models/Qwen2.5-3B-Instruct}"
[ -d "$ACTOR_MODEL_PATH" ] || { echo "error: ARPO_ACTOR_MODEL_PATH not found"; exit 1; }

# ── JanusPO recipe (matches plan_c_asymkl_v2_ub25_v1) ──────────────────────
ROLLOUT_NAME="vllm"
ROLLOUT_MODE="sync"
ROLLOUT_N="${ARPO_ROLLOUT_N:-8}"
ROLLOUT_TEMP="${ARPO_ROLLOUT_TEMP:-0.9}"
ACTOR_LR="${ARPO_ACTOR_LR:-1e-6}"
INITIAL_ROLLOUTS=1
BRANCH_PROBABILITY=0.0
Entropy_weight="${ARPO_ENTROPY_WEIGHT:-0.05}"
KL_LOSS_COEF="${ARPO_KL_LOSS_COEF:-0.08}"

CLIP_RATIO_POS="${ARPO_CLIP_RATIO_POS:-0.20}"
CLIP_RATIO_NEG="${ARPO_CLIP_RATIO_NEG:-0.30}"
KL_BETA_POS="${ARPO_KL_BETA_POS:-0.5}"
KL_BETA_NEG="${ARPO_KL_BETA_NEG:-1.5}"

W_T_MODE="${ARPO_W_T_MODE:-exp_decay}"
W_T_BETA="${ARPO_W_T_BETA:-0.9}"
W_T_ALPHA="${ARPO_W_T_ALPHA:-0.5}"
REWARD_MANAGER="gtpo"

ENTROPY_COEFF="${ARPO_ENTROPY_COEFF:-0.005}"
ENTROPY_UPPER_BOUND="${ARPO_ENTROPY_UPPER_BOUND:-2.5}"

LOG_VAL_GENERATIONS="${ARPO_LOG_VAL_GENERATIONS:-8}"
TOTAL_EPOCHS="${ARPO_TOTAL_EPOCHS:-1}"
SAVE_FREQ="${ARPO_SAVE_FREQ:-50}"

SAVE_PATH="${ARPO_CHECKPOINT_DIR:-${PARENT_DIR}/checkpoints}/${EXPERIMENT_NAME}"
ROLLOUT_SAVE_PATH="${SAVE_PATH}/rollout"

WANDB_API_KEY="${WANDB_API_KEY:-}"
if [ -n "$WANDB_API_KEY" ]; then
  wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
  export WANDB_DIR="${SAVE_PATH}"
  LOGGER_CFG='[console,wandb]'
else
  LOGGER_CFG='[console]'
fi

mkdir -p "$SAVE_PATH" "$ROLLOUT_SAVE_PATH"

cat <<EOF
================================================================
 JanusPO on MATH (cross-domain validation of GSM8K winner)
  exp:                 ${EXPERIMENT_NAME}
  seed:                ${ARPO_SEED}
  model:               ${ACTOR_MODEL_PATH}
  train:               ${TRAIN_FILES}
  val:                 ${VALID_FILES}  (MATH-500)
  AsymClip:            ε_+=${CLIP_RATIO_POS}  ε_-=${CLIP_RATIO_NEG}
  AsymKL:              β_pos=${KL_BETA_POS}  β_neg=${KL_BETA_NEG}
  KL coef:             ${KL_LOSS_COEF}
  Entropy:             coeff=${ENTROPY_COEFF}  ub=${ENTROPY_UPPER_BOUND}
  Reward:              gtpo  w_t_mode=${W_T_MODE}
  lr=${ACTOR_LR}  rollout_n=${ROLLOUT_N}  T=${ROLLOUT_TEMP}  batch=${TRAIN_BATCH_SIZE}
  prompt/resp len:     ${MAX_PROMPT_LENGTH}/${MAX_RESPONSE_LENGTH}
================================================================
EOF

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VALID_FILES}" \
    data.prompt_key="prompt" \
    +data.seed="${ARPO_SEED}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path="${ACTOR_MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
    +actor_rollout_ref.actor.entropy_upper_bound="${ENTROPY_UPPER_BOUND}" \
    +actor_rollout_ref.actor.use_turn_clip=false \
    +actor_rollout_ref.actor.use_gspo_ratio=false \
    +actor_rollout_ref.actor.clip_ratio_pos="${CLIP_RATIO_POS}" \
    +actor_rollout_ref.actor.clip_ratio_neg="${CLIP_RATIO_NEG}" \
    +actor_rollout_ref.actor.kl_beta_pos="${KL_BETA_POS}" \
    +actor_rollout_ref.actor.kl_beta_neg="${KL_BETA_NEG}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="${ROLLOUT_NAME}" \
    actor_rollout_ref.rollout.mode="${ROLLOUT_MODE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ARPO_VLLM_GPU_MEMORY_UTIL}" \
    actor_rollout_ref.rollout.max_num_seqs="${ARPO_VLLM_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ARPO_VLLM_MAX_BATCH_TOKENS}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMP}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.initial_rollouts="${INITIAL_ROLLOUTS}" \
    actor_rollout_ref.rollout.beam_size=1 \
    actor_rollout_ref.rollout.branch_probability="${BRANCH_PROBABILITY}" \
    actor_rollout_ref.rollout.entropy_weight="${Entropy_weight}" \
    '~actor_rollout_ref.rollout.tools.tool_instances' \
    actor_rollout_ref.rollout.tools.verbose_logging=False \
    actor_rollout_ref.rollout.multi_turn.enable=false \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager="${REWARD_MANAGER}" \
    +reward_model.reward_kwargs.w_t_mode="${W_T_MODE}" \
    +reward_model.reward_kwargs.w_t_beta="${W_T_BETA}" \
    +reward_model.reward_kwargs.w_t_alpha="${W_T_ALPHA}" \
    trainer.critic_warmup=0 \
    trainer.logger="${LOGGER_CFG}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.log_val_generations="${LOG_VAL_GENERATIONS}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.default_local_dir="${SAVE_PATH}" \
    trainer.val_before_train=False \
    trainer.rollout_data_dir="${ROLLOUT_SAVE_PATH}" \
    hydra.run.dir="${SAVE_PATH}/outputs" 2>&1 | tee "${SAVE_PATH}/run.log"
