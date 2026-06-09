#!/usr/bin/env bash
# ============================================================================
# Vanilla GSPO baseline on Qwen2.5-3B-Instruct + GSM8K — apples-to-apples vs
# GRPO / DAPO / RLOO / JanusPO in the same codebase and hyperparameter regime.

# GSPO recipe (Zheng et al. 2025, https://arxiv.org/abs/2507.18071):
#   • adv_estimator:            grpo  (Zheng et al. keeps GRPO advantages)
#   • importance ratio:         SEQUENCE-level length-normalised
#                                  s_i = exp( (1/|y|) · Σ_t [logπ − logπ_old] )
#     (enabled via verl flag actor.use_gspo_ratio=True)
#   • PPO clipping:             SYMMETRIC ε = 0.20  (paper default)
#   • KL regularizer:           β_KL = 0.001  (same as clean GRPO)
#   • No entropy bonus, no upper entropy bound
#   • Naive reward manager (no turn-weighting)

# Why this baseline exists:
#   Zheng et al. show GSPO fixes a numerical pathology in the PPO ratio at
#   large batch sizes.  We isolate that contribution by holding every other
#   knob equal to clean GRPO and flipping only the ratio (+ symmetric clip).
#   If GSPO alone matches or beats JanusPO on 3B/batch-32, its sequence-level
#   ratio is the more important fix; otherwise the AsymKL+entropy-guard
#   safety rails remain load-bearing.

# Identical to the other 3B runs (§5.2 Table 2):
#   Qwen2.5-3B-Instruct, grad-ckpt on, GSM8K train/test,
#   lr 1e-6, train_batch=32, ppo_mini=32, rollout_n=8, T=0.9, 1 epoch.

# Usage:
#   conda activate arpo
#   export ARPO_ACTOR_MODEL_PATH="$HOME/models/Qwen2.5-3B-Instruct"
#   bash scripts/ARPO_3B_GSM8K_gspo.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"

if [ -f "${SCRIPT_DIR}/local.env" ]; then set -a; source "${SCRIPT_DIR}/local.env"; set +a; fi

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
ARPO_SEED="${ARPO_SEED:-1}"
export PYTHONHASHSEED="${ARPO_SEED}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export RAY_memory_usage_threshold=0.8
export RAY_memory_monitor_refresh_ms=0
if python3 -c "import xformers" 2>/dev/null; then export VLLM_ATTENTION_BACKEND=XFORMERS; fi
export PYTHONPATH="${PARENT_DIR}/verl_arpo_entropy:${PYTHONPATH:-}"
if ! python3 -c "import ray" 2>/dev/null; then echo "error:   conda activate arpo"; exit 1; fi
[ "${ARPO_ENABLE_TORCH_COMPILE:-0}" = "1" ] || export TORCHDYNAMO_DISABLE=1

PROJECT_NAME="gsm8k_grpo"
EXPERIMENT_NAME="${ARPO_EXPERIMENT_NAME:-gspo_3b_gsm8k_v1}"
RESUME_MODE="${ARPO_RESUME_MODE:-disable}"
CONFIG_PATH="${PARENT_DIR}/scripts/config"
CONFIG_NAME="ppo_trainer.yaml"

NNODES=1
N_GPUS_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')
[ "$N_GPUS_PER_NODE" -ge 1 ] || N_GPUS_PER_NODE=1

MAX_PROMPT_LENGTH="${ARPO_MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${ARPO_MAX_RESPONSE_LENGTH:-768}"
TRAIN_BATCH_SIZE="${ARPO_TRAIN_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${ARPO_PPO_MINI_BATCH_SIZE:-32}"
ARPO_VLLM_GPU_MEMORY_UTIL="${ARPO_VLLM_GPU_MEMORY_UTIL:-0.82}"
ARPO_VLLM_MAX_NUM_SEQS="${ARPO_VLLM_MAX_NUM_SEQS:-64}"
ARPO_VLLM_MAX_BATCH_TOKENS="${ARPO_VLLM_MAX_BATCH_TOKENS:-8192}"
TEST_FREQ="${ARPO_TEST_FREQ:-10}"

TRAIN_FILES="${PARENT_DIR}/rl_datasets/gsm8k/train.parquet"
VALID_FILES="${PARENT_DIR}/rl_datasets/gsm8k/test.parquet"

ACTOR_MODEL_PATH="${ARPO_ACTOR_MODEL_PATH:-$HOME/models/Qwen2.5-3B-Instruct}"
[ -d "$ACTOR_MODEL_PATH" ] || { echo "error: ARPO_ACTOR_MODEL_PATH not found"; exit 1; }

ROLLOUT_NAME="vllm"
ROLLOUT_MODE="sync"
ROLLOUT_N="${ARPO_ROLLOUT_N:-8}"
ROLLOUT_TEMP="${ARPO_ROLLOUT_TEMP:-0.9}"
ACTOR_LR="${ARPO_ACTOR_LR:-1e-6}"
INITIAL_ROLLOUTS=1
BRANCH_PROBABILITY=0.0

# GSPO-specific config: GRPO advantage + sequence-level ratio + sym clip 0.2 + tiny KL
GSPO_CLIP=0.20
GSPO_KL_COEF="${ARPO_KL_LOSS_COEF:-0.001}"
ENTROPY_COEFF=0.0
Entropy_weight=0.0
REWARD_MANAGER="naive"

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

[ -f "$TRAIN_FILES" ] || { echo "  GSM8K train; bash rl_datasets/download_gsm8k.sh"; exit 1; }
mkdir -p "$SAVE_PATH" "$ROLLOUT_SAVE_PATH"

cat <<EOF
================================================================
 GSPO baseline (apples-to-apples vs GRPO/DAPO/RLOO/JanusPO)
  exp:               ${EXPERIMENT_NAME}
  model:             ${ACTOR_MODEL_PATH}
  adv_estimator:     grpo  (Zheng et al. keep GRPO advantages)
  ratio:             sequence-level length-normalised (use_gspo_ratio=True)
  clip:              symmetric ε=${GSPO_CLIP}
  KL:                use_kl_loss=True  coef=${GSPO_KL_COEF}
  Reward:            naive
  lr=${ACTOR_LR}  rollout_n=${ROLLOUT_N}  T=${ROLLOUT_TEMP}  batch=${TRAIN_BATCH_SIZE}
  save_path:         ${SAVE_PATH}
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
    actor_rollout_ref.actor.kl_loss_coef="${GSPO_KL_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio="${GSPO_CLIP}" \
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
    +actor_rollout_ref.actor.use_turn_clip=false \
    +actor_rollout_ref.actor.use_gspo_ratio=true \
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
