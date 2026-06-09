#!/usr/bin/env bash
# GTPO 7B GSM8K — plan_c_asymkl_v2 **  7B JanusPO  ** 

#   4   (optimizer × seed)   paper Section 5.4   controlled
# optimizer-precision ablation  4   7B run   2×2 paired  

#   PagedAdamW32bit (paged fp32) × seed 1   →  
#   PagedAdamW32bit (paged fp32) × seed 2   →   seed   fp32  
#   PagedAdamW8bit  (paged 8-bit) × seed 1  →   NaN regime + −3.1pp transfer
#   PagedAdamW8bit  (paged 8-bit) × seed 2  →   seed   NaN  


#   ARPO_USE_PAGED_ADAMW_32BIT=1 ARPO_USE_8BIT_ADAMW=0   fp32,  
#   ARPO_USE_8BIT_ADAMW=1        ARPO_USE_PAGED_ADAMW_32BIT=0   8-bit  
#   ARPO_SEED=2                  →   seed=2 dataloader shuffle

#  :
#   conda activate arpo
#   export ARPO_ACTOR_MODEL_PATH="$HOME/models/Qwen2.5-7B-Instruct"
#   #   1: fp32  
#   ARPO_EXPERIMENT_NAME=plan_c_asymkl_v2_7b_bf16_v1 \
#     bash scripts/ARPO_7B_GSM8K_plan_c_asymkl_v2_bf16.sh
#   #   2: 8-bit seed=2  
#   ARPO_USE_8BIT_ADAMW=1 ARPO_USE_PAGED_ADAMW_32BIT=0 ARPO_SEED=2 \
#     ARPO_EXPERIMENT_NAME=plan_c_asymkl_v2_7b_8bit_seed2_v1 \
#     bash scripts/ARPO_7B_GSM8K_plan_c_asymkl_v2_bf16.sh

#  : python3 scripts/diagnose_run.py checkpoints/<EXPERIMENT_NAME>/run.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"
echo "ARPO  : $PARENT_DIR"

if [ -f "${SCRIPT_DIR}/local.env" ]; then
  set -a; source "${SCRIPT_DIR}/local.env"; set +a
  echo "  ${SCRIPT_DIR}/local.env"
fi

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export HF_HOME="${PARENT_DIR}/rl_datasets/hf_cache"
export HF_DATASETS_CACHE="${PARENT_DIR}/rl_datasets/hf_cache/datasets"
mkdir -p "${HF_DATASETS_CACHE}"
export VLLM_USE_V1=0
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
  echo " :   conda activate arpo"; exit 1
fi

if [ "${ARPO_ENABLE_TORCH_COMPILE:-0}" != "1" ]; then
  export TORCHDYNAMO_DISABLE=1
fi

# ──   ─────────────────────────────────────────────────────────────────
PROJECT_NAME="gsm8k_grpo"
EXPERIMENT_NAME="${ARPO_EXPERIMENT_NAME:-plan_c_asymkl_v2_7b_bf16_v1}"
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

# ──   controlled 8-bit vs fp32  ─────────────
#  PagedAdamW32bit (paged fp32) —   NaN cross-domain transfer  
#  PagedAdamW8bit  (paged 8-bit) — paper Section 5.4 ablation   step 90+   NaN

#   1 paged32  
#   ARPO_USE_PAGED_ADAMW_32BIT=1  ARPO_USE_8BIT_ADAMW=0   ( )
#   ARPO_USE_8BIT_ADAMW=1         ARPO_USE_PAGED_ADAMW_32BIT=0  (8-bit  )
ARPO_USE_8BIT_ADAMW="${ARPO_USE_8BIT_ADAMW:-0}"
ARPO_USE_PAGED_ADAMW_32BIT="${ARPO_USE_PAGED_ADAMW_32BIT:-1}"

#  8-bit   force   paged32
if [ "$ARPO_USE_8BIT_ADAMW" = "1" ] && [ "$ARPO_USE_PAGED_ADAMW_32BIT" = "1" ]; then
  echo "[warn] USE_8BIT_ADAMW   USE_PAGED_ADAMW_32BIT   1  8-bit   paged32 "
  ARPO_USE_PAGED_ADAMW_32BIT=0
fi

# ──   / offload ────────────────────────────────────
if [ "$N_GPUS_PER_NODE" -eq 1 ]; then
  TRAIN_BATCH_SIZE="${ARPO_TRAIN_BATCH_SIZE:-32}"
  PPO_MINI_BATCH_SIZE="${ARPO_PPO_MINI_BATCH_SIZE:-16}"
  ARPO_VLLM_GPU_MEMORY_UTIL="${ARPO_VLLM_GPU_MEMORY_UTIL:-0.20}"
  ARPO_VLLM_MAX_NUM_SEQS="${ARPO_VLLM_MAX_NUM_SEQS:-32}"
  ARPO_VLLM_MAX_BATCH_TOKENS="${ARPO_VLLM_MAX_BATCH_TOKENS:-4096}"
  TEST_FREQ="${ARPO_TEST_FREQ:-10}"
  if [ "$ARPO_USE_8BIT_ADAMW" = "1" ]; then
    # PagedAdamW8bit   CUDA unified memory param_offload   FSDP  
    #   bnb   → IMA optimizer_offload  
    ARPO_ACTOR_PARAM_OFFLOAD="${ARPO_ACTOR_PARAM_OFFLOAD:-0}"
    ARPO_OPTIMIZER_OFFLOAD="${ARPO_OPTIMIZER_OFFLOAD:-0}"
  else
    # PagedAdamW32bit param_offload  optimizer_offload   verl
    # load_fsdp_optimizer   OOM 
    ARPO_ACTOR_PARAM_OFFLOAD="${ARPO_ACTOR_PARAM_OFFLOAD:-1}"
    ARPO_OPTIMIZER_OFFLOAD="${ARPO_OPTIMIZER_OFFLOAD:-0}"
  fi
else
  TRAIN_BATCH_SIZE="${ARPO_TRAIN_BATCH_SIZE:-64}"
  PPO_MINI_BATCH_SIZE="${ARPO_PPO_MINI_BATCH_SIZE:-32}"
  ARPO_VLLM_GPU_MEMORY_UTIL="${ARPO_VLLM_GPU_MEMORY_UTIL:-0.55}"
  ARPO_VLLM_MAX_NUM_SEQS="${ARPO_VLLM_MAX_NUM_SEQS:-256}"
  ARPO_VLLM_MAX_BATCH_TOKENS="${ARPO_VLLM_MAX_BATCH_TOKENS:-16384}"
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
  _DEF="${HOME}/models/Qwen2.5-7B-Instruct"
  if [ -d "$_DEF" ] && [ -f "$_DEF/config.json" ]; then
    ACTOR_MODEL_PATH="$_DEF"
    echo "  ARPO_ACTOR_MODEL_PATH : ${ACTOR_MODEL_PATH}"
  fi
fi
if [ -z "$ACTOR_MODEL_PATH" ] || [ ! -d "$ACTOR_MODEL_PATH" ]; then
  echo " :   ARPO_ACTOR_MODEL_PATH   Qwen2.5-7B-Instruct"; exit 1
fi

# ──   8-bit  ──────────────────────────
ROLLOUT_NAME="vllm"
ROLLOUT_MODE="sync"
ROLLOUT_N="${ARPO_ROLLOUT_N:-8}"
INITIAL_ROLLOUTS=1
BRANCH_PROBABILITY=0.0
Entropy_weight="${ARPO_ENTROPY_WEIGHT:-0.05}"
ROLLOUT_TEMP="${ARPO_ROLLOUT_TEMP:-0.9}"
BEAM_SIZE=1
KL_LOSS_COEF="${ARPO_KL_LOSS_COEF:-0.08}"
ACTOR_LR="${ARPO_ACTOR_LR:-1e-6}"

# ── GTPO Phase-1:   Clip ────────────────────────────────────────────────
ARPO_USE_ASYMCLIP="${ARPO_USE_ASYMCLIP:-1}"
CLIP_RATIO_POS="${ARPO_CLIP_RATIO_POS:-0.20}"
CLIP_RATIO_NEG="${ARPO_CLIP_RATIO_NEG:-0.30}"
ARPO_USE_GSPO=0

GSPO_ARG="+actor_rollout_ref.actor.use_gspo_ratio=false"
if [ "$ARPO_USE_ASYMCLIP" = "1" ]; then
  ASYMCLIP_ARG="+actor_rollout_ref.actor.clip_ratio_pos=${CLIP_RATIO_POS} +actor_rollout_ref.actor.clip_ratio_neg=${CLIP_RATIO_NEG}"
else
  ASYMCLIP_ARG=""
fi

# ── GTPO Phase-2:   β_KL ────────────────────────────────────────────────
ARPO_USE_ASYMKL="${ARPO_USE_ASYMKL:-1}"
KL_BETA_POS="${ARPO_KL_BETA_POS:-0.5}"
KL_BETA_NEG="${ARPO_KL_BETA_NEG:-1.5}"
if [ "$ARPO_USE_ASYMKL" = "1" ]; then
  ASYMKL_ARG="+actor_rollout_ref.actor.kl_beta_pos=${KL_BETA_POS} +actor_rollout_ref.actor.kl_beta_neg=${KL_BETA_NEG}"
else
  ASYMKL_ARG=""
fi

# ── GTPO Phase-3: w_t(n)   ────────────────────────────────────
W_T_MODE="${ARPO_W_T_MODE:-exp_decay}"
W_T_BETA="${ARPO_W_T_BETA:-0.9}"
W_T_ALPHA="${ARPO_W_T_ALPHA:-0.5}"
REWARD_MANAGER="gtpo"

# ── Plan C:   ──────────────────────────────────────────────────
ENTROPY_COEFF="${ARPO_ENTROPY_COEFF:-0.005}"
ENTROPY_UPPER_BOUND="${ARPO_ENTROPY_UPPER_BOUND:-2.5}"

LOG_VAL_GENERATIONS="${ARPO_LOG_VAL_GENERATIONS:-8}"
TOTAL_EPOCHS="${ARPO_TOTAL_EPOCHS:-1}"
SAVE_FREQ="${ARPO_SAVE_FREQ:-25}"

# ── offload bf16  ────────────────────────────────────────────
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

#   Hydra override  
if [ "$ARPO_USE_8BIT_ADAMW" = "1" ]; then
  ADAMW8BIT_ARG="+actor_rollout_ref.actor.optim.use_8bit_adamw=true +actor_rollout_ref.actor.optim.adamw_8bit_paged=true"
  PAGED32BIT_ARG="+actor_rollout_ref.actor.optim.use_paged_adamw_32bit=false"
  OPTIMIZER_LABEL="PagedAdamW8bit (paged 8-bit, controlled ablation)"
else
  ADAMW8BIT_ARG="+actor_rollout_ref.actor.optim.use_8bit_adamw=false"
  if [ "$ARPO_USE_PAGED_ADAMW_32BIT" = "1" ]; then
    PAGED32BIT_ARG="+actor_rollout_ref.actor.optim.use_paged_adamw_32bit=true"
    OPTIMIZER_LABEL="PagedAdamW32bit (paged fp32, default)"
  else
    PAGED32BIT_ARG="+actor_rollout_ref.actor.optim.use_paged_adamw_32bit=false"
    OPTIMIZER_LABEL="standard fp32 AdamW (no paging — only fits with optimizer_offload=1)"
  fi
fi

# ──   seed  controlled cross-seed replication ─────────────────────
# verl   train_dataloader_generator.manual_seed(data_config.get('seed', 1))
#   seed=1  ARPO_SEED=2   dataloader shuffle  
SEED_ARG=""
if [ -n "${ARPO_SEED:-}" ]; then
  SEED_ARG="+data.seed=${ARPO_SEED}"
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
  echo " : bash ${PARENT_DIR}/rl_datasets/download_gsm8k.sh"; exit 1
fi

mkdir -p "$SAVE_PATH" "$ROLLOUT_SAVE_PATH"

echo "========================================================"
echo " 7B JanusPO — controlled optimizer-precision study"
echo "   :               Qwen2.5-7B-Instruct"
echo "   :             ${EXPERIMENT_NAME}"
echo "  optimizer:          ${OPTIMIZER_LABEL}"
echo "  data seed:          ${ARPO_SEED:-1 (verl default)}"
echo "  param_offload:      ${ARPO_ACTOR_PARAM_OFFLOAD}"
echo "  optimizer_offload:  ${ARPO_OPTIMIZER_OFFLOAD}"
echo "  vLLM gpu_util:      ${ARPO_VLLM_GPU_MEMORY_UTIL}"
echo "  entropy_coeff:      ${ENTROPY_COEFF}  ub=${ENTROPY_UPPER_BOUND}"
echo "  AsymClip ε+/-:     ${CLIP_RATIO_POS} / ${CLIP_RATIO_NEG}"
echo "  AsymKL β+/-:       ${KL_BETA_POS} / ${KL_BETA_NEG}"
echo "  KL coef:            ${KL_LOSS_COEF}"
echo "  lr=${ACTOR_LR}  rollout_n=${ROLLOUT_N}  batch=${TRAIN_BATCH_SIZE}  mini=${PPO_MINI_BATCH_SIZE}"
echo "  save_freq=${SAVE_FREQ}  max_actor_ckpt_to_keep=3"
if [ "$ARPO_USE_8BIT_ADAMW" = "1" ]; then
  echo "    wallclock: ~6 min 8-bit   step 90+   NaN "
else
  echo "    wallclock: ~9-10 min fp32 paged  "
fi
echo "========================================================"
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
    ${SEED_ARG} \
    actor_rollout_ref.model.path="${ACTOR_MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
    +actor_rollout_ref.actor.entropy_upper_bound="${ENTROPY_UPPER_BOUND}" \
    +actor_rollout_ref.actor.use_turn_clip=false \
    ${GSPO_ARG} \
    ${ASYMCLIP_ARG} \
    ${ASYMKL_ARG} \
    "${ACTOR_PARAM_OFFLOAD_ARG}" \
    "${ACTOR_OPT_OFFLOAD_ARG}" \
    ${ADAMW8BIT_ARG} \
    ${PAGED32BIT_ARG} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((2 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="${ROLLOUT_NAME}" \
    actor_rollout_ref.rollout.mode="${ROLLOUT_MODE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ARPO_VLLM_GPU_MEMORY_UTIL}" \
    actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
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
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((2 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) \
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
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.log_val_generations="${LOG_VAL_GENERATIONS}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.default_local_dir="${SAVE_PATH}" \
    trainer.val_before_train=False \
    trainer.rollout_data_dir="${ROLLOUT_SAVE_PATH}" \
    hydra.run.dir="${SAVE_PATH}/outputs" 2>&1 | tee "${SAVE_PATH}/run.log"
