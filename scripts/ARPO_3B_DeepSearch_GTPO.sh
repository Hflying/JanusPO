#!/usr/bin/env bash
# GTPO   DeepSearch  Qwen2.5-3B-Instruct 

#   GTPO 
#   Phase-1:   Clip  ε_+=0.20  ε_-=0.30
#   Phase-2:   β_KL  β_pos=0.5  β_neg=1.5
#   Phase-3: w_t(n)  
#   H(n):      entropy_coeff 
#   Plan-C:  entropy_coeff=0.005 + entropy_upper_bound=2.5


#    : train_10k.parquet DR_grpo_mix 10k  + 
#    : gaia_test.parquet + hle_test.parquet


#    mock  ARPO_SEARCH_CACHE_PATH  
#     Bing   ARPO_USE_MOCK_SEARCH=0   api_key

#  :
#   conda activate arpo
#   export ARPO_ACTOR_MODEL_PATH="$HOME/models/Qwen2.5-3B-Instruct"
#   export ARPO_EXPERIMENT_NAME=gtpo_3b_deepsearch_v1
#   bash scripts/ARPO_3B_DeepSearch_GTPO.sh

#   (  GPU  ):
#   ARPO_TRAIN_BATCH_SIZE=8 ARPO_ROLLOUT_N=4 ARPO_TEST_FREQ=2 ARPO_SAVE_FREQ=2 bash ...

#  : python3 scripts/diagnose_run.py checkpoints/gtpo_3b_deepsearch_v1/run.log

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
PROJECT_NAME="deepsearch_grpo"
EXPERIMENT_NAME="${ARPO_EXPERIMENT_NAME:-gtpo_3b_deepsearch_v1}"
RESUME_MODE="${ARPO_RESUME_MODE:-disable}"
CONFIG_PATH="${PARENT_DIR}/scripts/config"
CONFIG_NAME="ppo_trainer_dr.yaml"

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
#   GPU (94GB)  sync_with_tool   vLLM   KV cache 
#   vLLM KV(0.30×94.5≈28G) + root (14G) + training(~35G) ≈ 77G < 80G ✅
#   MAX_RESPONSE_LENGTH=1500   backward  
#   GPU (4×):  response=8000  
MAX_PROMPT_LENGTH="${ARPO_MAX_PROMPT_LENGTH:-2000}"
MAX_RESPONSE_LENGTH="${ARPO_MAX_RESPONSE_LENGTH:-1500}"

# ──   ─────────────────────────────────────────────────────────────────
if [ "$N_GPUS_PER_NODE" -le 2 ]; then
  TRAIN_BATCH_SIZE="${ARPO_TRAIN_BATCH_SIZE:-8}"
  PPO_MINI_BATCH_SIZE="${ARPO_PPO_MINI_BATCH_SIZE:-8}"
  ARPO_VLLM_GPU_MEMORY_UTIL="${ARPO_VLLM_GPU_MEMORY_UTIL:-0.30}"
  ARPO_VLLM_MAX_NUM_SEQS="${ARPO_VLLM_MAX_NUM_SEQS:-32}"
  ARPO_VLLM_MAX_BATCH_TOKENS="${ARPO_VLLM_MAX_BATCH_TOKENS:-16000}"
  TEST_FREQ="${ARPO_TEST_FREQ:-5}"
  ARPO_ACTOR_PARAM_OFFLOAD="${ARPO_ACTOR_PARAM_OFFLOAD:-1}"
  ARPO_OPTIMIZER_OFFLOAD="${ARPO_OPTIMIZER_OFFLOAD:-1}"
else
  TRAIN_BATCH_SIZE="${ARPO_TRAIN_BATCH_SIZE:-64}"
  PPO_MINI_BATCH_SIZE="${ARPO_PPO_MINI_BATCH_SIZE:-16}"
  ARPO_VLLM_GPU_MEMORY_UTIL="${ARPO_VLLM_GPU_MEMORY_UTIL:-0.6}"
  ARPO_VLLM_MAX_NUM_SEQS="${ARPO_VLLM_MAX_NUM_SEQS:-256}"
  ARPO_VLLM_MAX_BATCH_TOKENS="${ARPO_VLLM_MAX_BATCH_TOKENS:-25000}"
  TEST_FREQ="${ARPO_TEST_FREQ:-5}"
  ARPO_ACTOR_PARAM_OFFLOAD="${ARPO_ACTOR_PARAM_OFFLOAD:-0}"
  ARPO_OPTIMIZER_OFFLOAD="${ARPO_OPTIMIZER_OFFLOAD:-0}"
fi

# ──  ────────────────────────────────────────────────────────────
TRAIN_FILES="${ARPO_TRAIN_FILES:-${PARENT_DIR}/rl_datasets/train_10k.parquet}"
VALID_FILES="${ARPO_VALID_FILES:-[\"${PARENT_DIR}/rl_datasets/gaia_test.parquet\",\"${PARENT_DIR}/rl_datasets/hle_test.parquet\"]}"

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
  echo " :   ARPO_ACTOR_MODEL_PATH  Qwen2.5-3B-Instruct  "; exit 1
fi

# ──   ──────────────────────────────────────────────────────────────
#  MockSearchTool 
#   ARPO_USE_MOCK_SEARCH=0   ARPO_BING_API_KEY
ARPO_USE_MOCK_SEARCH="${ARPO_USE_MOCK_SEARCH:-1}"
SEARCH_CACHE_PATH="${ARPO_SEARCH_CACHE_PATH:-${PARENT_DIR}/search_cache/search_cache.json}"
if [ "$ARPO_USE_MOCK_SEARCH" = "1" ]; then
  SEARCH_CLASS_PATH="verl.workers.agent.tools.mock_search_tool.MockSearchTool"
  echo " : MockSearchTool : ${SEARCH_CACHE_PATH} "
else
  SEARCH_CLASS_PATH="verl.workers.agent.tools.search_tool.BingSearchTool"
  echo " : BingSearchTool"
fi

# ──   ─────────────────────────────────────────────────────────────────
ROLLOUT_NAME="vllm"
ROLLOUT_MODE="sync_with_tool"
ROLLOUT_N="${ARPO_ROLLOUT_N:-8}"
INITIAL_ROLLOUTS="${ARPO_INITIAL_ROLLOUTS:-4}"
BEAM_SIZE="${ARPO_BEAM_SIZE:-2}"
BRANCH_PROBABILITY="${ARPO_BRANCH_PROBABILITY:-0.5}"
Entropy_weight="${ARPO_ENTROPY_WEIGHT:-0.2}"
ROLLOUT_TEMP="${ARPO_ROLLOUT_TEMP:-1.0}"
KL_LOSS_COEF="${ARPO_KL_LOSS_COEF:-0.08}"
ACTOR_LR="${ARPO_ACTOR_LR:-1e-6}"

# ── GTPO Phase-1:   Clip ────────────────────────────────────────────────
CLIP_RATIO_POS="${ARPO_CLIP_RATIO_POS:-0.20}"
CLIP_RATIO_NEG="${ARPO_CLIP_RATIO_NEG:-0.30}"
GSPO_ARG="+actor_rollout_ref.actor.use_gspo_ratio=false"
ASYMCLIP_ARG="+actor_rollout_ref.actor.clip_ratio_pos=${CLIP_RATIO_POS} +actor_rollout_ref.actor.clip_ratio_neg=${CLIP_RATIO_NEG}"

# ── GTPO Phase-2:   β_KL ────────────────────────────────────────────────
KL_BETA_POS="${ARPO_KL_BETA_POS:-0.5}"
KL_BETA_NEG="${ARPO_KL_BETA_NEG:-1.5}"
ASYMKL_ARG="+actor_rollout_ref.actor.kl_beta_pos=${KL_BETA_POS} +actor_rollout_ref.actor.kl_beta_neg=${KL_BETA_NEG}"

# ── GTPO Phase-3: w_t(n)   ────────────────────────────────────
W_T_MODE="${ARPO_W_T_MODE:-exp_decay}"
W_T_BETA="${ARPO_W_T_BETA:-0.9}"
W_T_ALPHA="${ARPO_W_T_ALPHA:-0.5}"
REWARD_MANAGER="gtpo"

# ── Plan-C:   ───────────────────────────────────────────────────
ENTROPY_COEFF="${ARPO_ENTROPY_COEFF:-0.005}"
ENTROPY_UPPER_BOUND="${ARPO_ENTROPY_UPPER_BOUND:-2.5}"

# ── GTPO H(n):   ────────────────────────────────────────────────
#   t   token   h(t) = (t/n_turns)^hn_alpha
#  n_turns=1 → h(1)=1.0 →   entropy_coeff 
USE_HN_ENTROPY="${ARPO_USE_HN_ENTROPY:-true}"
HN_ALPHA="${ARPO_HN_ALPHA:-1.0}"


TOOL_CALL_LIMIT="${ARPO_TOOL_CALL_LIMIT:-5}"

LOG_VAL_GENERATIONS="${ARPO_LOG_VAL_GENERATIONS:-4}"
TOTAL_EPOCHS="${ARPO_TOTAL_EPOCHS:-1}"
SAVE_FREQ="${ARPO_SAVE_FREQ:-5}"

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

if [ ! -f "$TRAIN_FILES" ]; then
  echo " :  : ${TRAIN_FILES}"; exit 1
fi

mkdir -p "$SAVE_PATH" "$ROLLOUT_SAVE_PATH"

echo "================================================================"
echo " GTPO   DeepSearch  "
echo "   :            ${EXPERIMENT_NAME}"
echo "   :          ${TRAIN_FILES}"
echo "   :          ${VALID_FILES}"
echo "   :              ${ACTOR_MODEL_PATH}"
echo "  rollout_n:         ${ROLLOUT_N}   initial_rollouts: ${INITIAL_ROLLOUTS}"
echo "  beam_size:         ${BEAM_SIZE}   branch_prob: ${BRANCH_PROBABILITY}"
echo "  GTPO  :"
echo "    AsymClip:        ε_+=${CLIP_RATIO_POS}  ε_-=${CLIP_RATIO_NEG}"
echo "    AsymKL:          β_pos=${KL_BETA_POS}  β_neg=${KL_BETA_NEG}"
echo "    w_t(n):          mode=${W_T_MODE}  β=${W_T_BETA}  α=${W_T_ALPHA}"
echo "    entropy_coeff:   ${ENTROPY_COEFF}  upper_bound: ${ENTROPY_UPPER_BOUND}"
echo "    H(n):            use=${USE_HN_ENTROPY}  alpha=${HN_ALPHA}  ←  "
echo "  lr=${ACTOR_LR}  kl_coef=${KL_LOSS_COEF}  entropy_weight=${Entropy_weight}"
echo "   :          ${SEARCH_CLASS_PATH}"
echo "  GPU: ${N_GPUS_PER_NODE}  seq: ${MAX_PROMPT_LENGTH}+${MAX_RESPONSE_LENGTH}"
echo "================================================================"
echo " tail run.log   diagnose_run.py :"
echo "  actor/hn_turns_mean    ←   > 1 "
echo "  actor/entropy_loss     ←   H(n)  "
echo "  val-core/gaia/...      ← GAIA  "
echo "  val-core/hle/...       ← HLE   "
echo "================================================================"

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
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ARPO_PPO_MAX_TOKEN_LEN_PER_GPU:-$((2 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)))}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
    +actor_rollout_ref.actor.entropy_upper_bound="${ENTROPY_UPPER_BOUND}" \
    +actor_rollout_ref.actor.use_hn_entropy="${USE_HN_ENTROPY}" \
    +actor_rollout_ref.actor.hn_alpha="${HN_ALPHA}" \
    +actor_rollout_ref.actor.use_turn_clip=false \
    ${GSPO_ARG} \
    ${ASYMCLIP_ARG} \
    ${ASYMKL_ARG} \
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
    +actor_rollout_ref.rollout.entropy_weight="${Entropy_weight}" \
    +actor_rollout_ref.rollout.tools.tool_instances.search.params.cache_file="${SEARCH_CACHE_PATH}" \
    +actor_rollout_ref.rollout.tools.tool_instances.search.class_path="${SEARCH_CLASS_PATH}" \
    +actor_rollout_ref.rollout.tools.call_limit="${TOOL_CALL_LIMIT}" \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    +actor_rollout_ref.rollout.multi_turn.tool_config_path="${PARENT_DIR}/scripts/config/arpo_multiturn_tool_stub.yaml" \
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
