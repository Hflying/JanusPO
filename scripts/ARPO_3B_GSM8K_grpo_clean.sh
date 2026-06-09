#!/usr/bin/env bash
#   GRPO  Qwen2.5-3B-Instruct + GSM8K  ARPO   +   KL  


#   1. branch_probability=0 →   vLLM   "#### N"  
#   2. kl_loss_coef=0.05   →   0.001   KL   pg_loss   2% 
#                             entropy   11   2.27   0.13  50x

#  :
#   conda activate arpo
#   bash rl_datasets/download_gsm8k.sh
#   export ARPO_ACTOR_MODEL_PATH="$HOME/models/Qwen2.5-3B-Instruct"
#   bash scripts/ARPO_3B_GSM8K_grpo_clean.sh

#  :
#   - step 1-10 score 50-65% Instruct  
#   - entropy   20   <0.3 
#   - grad_norm   <5
#   - kl_loss   0.5-3 

#  : actor/lr   0.000  %.3f  lr=1e-6  

#  : python3 scripts/diagnose_run.py checkpoints/grpo_clean_v2/run.log
#  : python3 scripts/gsm8k_log_best_val_step.py checkpoints/grpo_clean_v2/run.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"
echo "ARPO  : $PARENT_DIR"

if [ -f "${SCRIPT_DIR}/local.env" ]; then
  set -a
  source "${SCRIPT_DIR}/local.env"
  set +a
  echo "  ${SCRIPT_DIR}/local.env"
fi

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
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
  echo " :   conda activate arpo"
  exit 1
fi

if [ "${ARPO_ENABLE_TORCH_COMPILE:-0}" != "1" ]; then
  export TORCHDYNAMO_DISABLE=1
fi

# ──   ─────────────────────────────────────────────────────────────────
PROJECT_NAME="gsm8k_grpo"
EXPERIMENT_NAME="${ARPO_EXPERIMENT_NAME:-grpo_clean_v2}"
RESUME_MODE="${ARPO_RESUME_MODE:-disable}"
CONFIG_PATH="${PARENT_DIR}/scripts/config"
CONFIG_NAME="ppo_trainer.yaml"

# ── GPU   ─────────────────────────────────────────────────────────────────
NNODES=1
if command -v nvidia-smi &>/dev/null; then
  N_GPUS_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l)
else
  N_GPUS_PER_NODE=1
fi
N_GPUS_PER_NODE=$(echo "$N_GPUS_PER_NODE" | tr -d '[:space:]')
if ! [[ "$N_GPUS_PER_NODE" =~ ^[0-9]+$ ]] || [ "$N_GPUS_PER_NODE" -lt 1 ]; then
  N_GPUS_PER_NODE=1
fi
echo "  GPU  : ${N_GPUS_PER_NODE}"

# ──   ─────────────────────────────────────────────────────────────────
MAX_PROMPT_LENGTH="${ARPO_MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${ARPO_MAX_RESPONSE_LENGTH:-768}"

# ──   ─────────────────────────────────────────────────────────────────
if [ "$N_GPUS_PER_NODE" -eq 1 ]; then
  # ppo_mini_batch_size = train_batch_size →   1   stale  
  TRAIN_BATCH_SIZE="${ARPO_TRAIN_BATCH_SIZE:-32}"
  PPO_MINI_BATCH_SIZE="${ARPO_PPO_MINI_BATCH_SIZE:-32}"
  ARPO_VLLM_GPU_MEMORY_UTIL="${ARPO_VLLM_GPU_MEMORY_UTIL:-0.82}"
  ARPO_VLLM_MAX_NUM_SEQS="${ARPO_VLLM_MAX_NUM_SEQS:-64}"
  ARPO_VLLM_MAX_BATCH_TOKENS="${ARPO_VLLM_MAX_BATCH_TOKENS:-8192}"
  TEST_FREQ="${ARPO_TEST_FREQ:-10}"
  ARPO_ACTOR_PARAM_OFFLOAD="${ARPO_ACTOR_PARAM_OFFLOAD:-1}"
  ARPO_OPTIMIZER_OFFLOAD="${ARPO_OPTIMIZER_OFFLOAD:-1}"
  echo " : batch=${TRAIN_BATCH_SIZE} mini_batch=${PPO_MINI_BATCH_SIZE} "
else
  TRAIN_BATCH_SIZE="${ARPO_TRAIN_BATCH_SIZE:-128}"
  PPO_MINI_BATCH_SIZE="${ARPO_PPO_MINI_BATCH_SIZE:-128}"
  ARPO_VLLM_GPU_MEMORY_UTIL="${ARPO_VLLM_GPU_MEMORY_UTIL:-0.7}"
  ARPO_VLLM_MAX_NUM_SEQS="${ARPO_VLLM_MAX_NUM_SEQS:-1024}"
  ARPO_VLLM_MAX_BATCH_TOKENS="${ARPO_VLLM_MAX_BATCH_TOKENS:-8192}"
  TEST_FREQ="${ARPO_TEST_FREQ:-5}"
  ARPO_ACTOR_PARAM_OFFLOAD="${ARPO_ACTOR_PARAM_OFFLOAD:-0}"
  ARPO_OPTIMIZER_OFFLOAD="${ARPO_OPTIMIZER_OFFLOAD:-0}"
fi

# ──   ───────────────────────────────────────────────────────────────────
TRAIN_FILES="${PARENT_DIR}/rl_datasets/gsm8k/train.parquet"
VALID_FILES="${PARENT_DIR}/rl_datasets/gsm8k/test.parquet"

# ──   ─────────────────────────────────────────────────────────────────
ACTOR_MODEL_PATH="${ARPO_ACTOR_MODEL_PATH:-}"
if [ -z "$ACTOR_MODEL_PATH" ]; then
  _DEF="${HOME}/models/Qwen2.5-3B-Instruct"
  if [ -d "$_DEF" ] && [ -f "$_DEF/config.json" ]; then
    ACTOR_MODEL_PATH="$_DEF"
    echo "  ARPO_ACTOR_MODEL_PATH : ${ACTOR_MODEL_PATH}"
  fi
fi
if [ -z "$ACTOR_MODEL_PATH" ] || [ ! -d "$ACTOR_MODEL_PATH" ]; then
  echo " :   ARPO_ACTOR_MODEL_PATH"
  exit 1
fi

# ──   ─────────────────────────────────────────────────────────────────
ROLLOUT_NAME="vllm"
ROLLOUT_MODE="sync"
ROLLOUT_N="${ARPO_ROLLOUT_N:-8}"       # 8   GRPO  
INITIAL_ROLLOUTS=1
BRANCH_PROBABILITY=0.0                  #   ARPO  
Entropy_weight="${ARPO_ENTROPY_WEIGHT:-0.05}"  #   KL 
ROLLOUT_TEMP="${ARPO_ROLLOUT_TEMP:-0.9}"
BEAM_SIZE=1

# kl_loss_coef=0.05 pg_loss≈0.35  KL   0.05×KL≈0.15  loss   30%
#   0.001   KL   0.008 pg_loss   2%  entropy  
KL_LOSS_COEF="${ARPO_KL_LOSS_COEF:-0.05}"
ACTOR_LR="${ARPO_ACTOR_LR:-1e-6}"

LOG_VAL_GENERATIONS="${ARPO_LOG_VAL_GENERATIONS:-8}"
TOTAL_EPOCHS="${ARPO_TOTAL_EPOCHS:-1}"
SAVE_FREQ="${ARPO_SAVE_FREQ:-5}"
REWARD_MANAGER="naive"

# ── offload ──────────────────────────────────────────────────────────────────
if [ "$ARPO_ACTOR_PARAM_OFFLOAD" = "1" ]; then
  ACTOR_PARAM_OFFLOAD_ARG="actor_rollout_ref.actor.fsdp_config.param_offload=True"
else
  ACTOR_PARAM_OFFLOAD_ARG="actor_rollout_ref.actor.fsdp_config.param_offload=False"
fi
if [ "$ARPO_OPTIMIZER_OFFLOAD" = "1" ]; then
  ACTOR_OPT_OFFLOAD_ARG="actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
else
  ACTOR_OPT_OFFLOAD_ARG="actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
fi

# ──   ───────────────────────────────────────────────────────────────
SAVE_PATH="${ARPO_CHECKPOINT_DIR:-${PARENT_DIR}/checkpoints}/${EXPERIMENT_NAME}"
ROLLOUT_SAVE_PATH="${SAVE_PATH}/rollout"

WANDB_API_KEY="${WANDB_API_KEY:-}"
if [ -n "$WANDB_API_KEY" ]; then
  wandb login --relogin "$WANDB_API_KEY"
  export WANDB_DIR="${SAVE_PATH}"
  LOGGER_CFG='[console,wandb]'
else
  LOGGER_CFG='[console]'
  echo "  WANDB_API_KEY "
fi

if [ ! -f "$TRAIN_FILES" ] || [ ! -f "$VALID_FILES" ]; then
  echo " : bash ${PARENT_DIR}/rl_datasets/download_gsm8k.sh"
  exit 1
fi

mkdir -p "$SAVE_PATH" "$ROLLOUT_SAVE_PATH"

echo "========================================================"
echo "   GRPO +   KL   entropy  "
echo "   :  ${EXPERIMENT_NAME}"
echo "  KL  : ${KL_LOSS_COEF}  0.001  0.05  50x "
echo "   :    branch_probability=${BRANCH_PROBABILITY} "
echo "  rollout_n=${ROLLOUT_N}  entropy_weight=${Entropy_weight}"
echo "  mini_batch=${PPO_MINI_BATCH_SIZE} =batch "
echo "  max_response=${MAX_RESPONSE_LENGTH}  lr=${ACTOR_LR}"
echo "========================================================"
echo " : entropy  , KL   0.5-3, grad_norm<5, score  "
echo " : python3 scripts/diagnose_run.py ${SAVE_PATH}/run.log"

# ──   ─────────────────────────────────────────────────────────────────
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VALID_FILES}" \
    data.prompt_key="prompt" \
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
    +actor_rollout_ref.actor.use_turn_clip=false \
    "${ACTOR_PARAM_OFFLOAD_ARG}" \
    "${ACTOR_OPT_OFFLOAD_ARG}" \
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
    actor_rollout_ref.rollout.beam_size="${BEAM_SIZE}" \
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
