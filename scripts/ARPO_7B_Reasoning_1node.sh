SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"
echo "ARPO  : $PARENT_DIR"

#   scripts/local.env.example 
if [ -f "${SCRIPT_DIR}/local.env" ]; then
  set -a
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/local.env"
  set +a
  echo "  ${SCRIPT_DIR}/local.env"
fi

# ============================ Environment Setup ============================
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VERL_LOGGING_LEVEL=DEBUG
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export RAY_memory_usage_threshold=0.8
export RAY_memory_monitor_refresh_ms=0

#   xformers   XFORMERS  vLLM  
if python3 -c "import xformers" 2>/dev/null; then
  export VLLM_ATTENTION_BACKEND=XFORMERS
fi

export PYTHONPATH="${PARENT_DIR}/verl_arpo_entropy:${PYTHONPATH:-}"

# PyTorch 2.6   torch._inductor   Triton   AttrsDescriptor Triton 3.5+   vLLM  aarch64/GH200  
#   Dynamo  torch+triton  : export ARPO_ENABLE_TORCH_COMPILE=1
if [ "${ARPO_ENABLE_TORCH_COMPILE:-0}" != "1" ]; then
  export TORCHDYNAMO_DISABLE=1
fi

# ============================ Basic Configuration ============================
PROJECT_NAME="reasoning_tasks"
EXPERIMENT_NAME="ARPO_global_16_init_8_beam_2_random_0_arpo_0.2_entropy"
CONFIG_PATH="${PARENT_DIR}/scripts/config"
CONFIG_NAME="ppo_trainer.yaml"

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

# ============================ Data Configuration ============================
PROMPT_KEY="prompt"
TRAIN_BATCH_SIZE="${ARPO_TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${ARPO_PPO_MINI_BATCH_SIZE:-16}"
MAX_PROMPT_LENGTH=1536
MAX_RESPONSE_LENGTH=4096

if [ "$N_GPUS_PER_NODE" -eq 1 ]; then
  TRAIN_BATCH_SIZE="${ARPO_TRAIN_BATCH_SIZE:-16}"
  PPO_MINI_BATCH_SIZE="${ARPO_PPO_MINI_BATCH_SIZE:-4}"
  echo "  batch: TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE} "
  # vLLM v1: KV   ≈ total_mem * gpu_memory_utilization - peak(FSDP+vLLM)  0.7   KV<=0 
  ARPO_VLLM_GPU_MEMORY_UTIL="${ARPO_VLLM_GPU_MEMORY_UTIL:-0.92}"
  ARPO_VLLM_MAX_NUM_SEQS="${ARPO_VLLM_MAX_NUM_SEQS:-256}"
  ARPO_ACTOR_PARAM_OFFLOAD="${ARPO_ACTOR_PARAM_OFFLOAD:-1}"
  ARPO_OPTIMIZER_OFFLOAD="${ARPO_OPTIMIZER_OFFLOAD:-1}"
  echo "  vLLM  : gpu_memory_utilization=${ARPO_VLLM_GPU_MEMORY_UTIL} max_num_seqs=${ARPO_VLLM_MAX_NUM_SEQS}  ARPO_VLLM_GPU_MEMORY_UTIL / ARPO_VLLM_MAX_NUM_SEQS  "
  echo "  FSDP  : ARPO_ACTOR_PARAM_OFFLOAD=${ARPO_ACTOR_PARAM_OFFLOAD} ARPO_OPTIMIZER_OFFLOAD=${ARPO_OPTIMIZER_OFFLOAD}  vLLM wake_up(kv_cache) OOM  0  "
else
  ARPO_VLLM_GPU_MEMORY_UTIL="${ARPO_VLLM_GPU_MEMORY_UTIL:-0.7}"
  ARPO_VLLM_MAX_NUM_SEQS="${ARPO_VLLM_MAX_NUM_SEQS:-1024}"
  ARPO_ACTOR_PARAM_OFFLOAD="${ARPO_ACTOR_PARAM_OFFLOAD:-0}"
  ARPO_OPTIMIZER_OFFLOAD="${ARPO_OPTIMIZER_OFFLOAD:-0}"
fi

TRAIN_FILES="${PARENT_DIR}/rl_datasets/train_10k.parquet"
VALID_FILES="${PARENT_DIR}/rl_datasets/valid.parquet"

# ============================ Model Configuration ============================
ACTOR_MODEL_PATH="${ARPO_ACTOR_MODEL_PATH:-}"
if [ -z "$ACTOR_MODEL_PATH" ] || [ ! -d "$ACTOR_MODEL_PATH" ]; then
  echo " : ARPO_ACTOR_MODEL_PATH  "
  echo " :   /path/to/model ...   HuggingFace   config.json "
  echo ""
  echo "  arpo   Qwen2.5-7B-Instruct "
  echo "  huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir \"\$HOME/models/Qwen2.5-7B-Instruct\""
  echo "  export ARPO_ACTOR_MODEL_PATH=\"\$HOME/models/Qwen2.5-7B-Instruct\""
  echo ""
  echo "  export   scripts/local.env  scripts/local.env.example "
  exit 1
fi
if [ ! -f "${ACTOR_MODEL_PATH}/config.json" ]; then
  echo " :   ${ACTOR_MODEL_PATH}   config.json  HF   tokenizer  "
  exit 1
fi

# ============================ Rollout Configuration ==========================
ROLLOUT_NAME="vllm"
ROLLOUT_MODE="sync_with_tool"
ROLLOUT_N=16
INITIAL_ROLLOUTS=8
BEAM_SIZE=2
BRANCH_PROBABILITY=0.5
Entropy_weight=0.2
ENABLE_MULTI_TURN=true

SEARCH_CACHE_PATH="${PARENT_DIR}/search_cache/search_cache.json"
mkdir -p "$(dirname "$SEARCH_CACHE_PATH")"
touch "$SEARCH_CACHE_PATH"

# RayPPOTrainer   multi_turn +   tool_config_path  vLLM   rollout.tools  stub  
MULTI_TURN_TOOL_CONFIG_STUB="${PARENT_DIR}/scripts/config/arpo_multiturn_tool_stub.yaml"

# ============================ Reward Model Configuration ==========================
REWARD_MANAGER="naive"
CUSTOM_REWARD_FUNCTION_PATH="${PARENT_DIR}/verl_arpo_entropy/verl/utils/reward_score/deep_research.py"
CUSTOM_REWARD_FUNCTION_NAME="compute_score"

# ============================ Training Configuration ============================
TOTAL_EPOCHS=2
SAVE_FREQ=5
TEST_FREQ=5

USE_TURN_CLIP=false
TURN_CLIP_RATIO=0.15
TURN_CLIP_RATIO_LOW=""
TURN_CLIP_RATIO_HIGH=""

# ============================ Path Configuration ============================
SAVE_PATH="${ARPO_CHECKPOINT_DIR:-${PARENT_DIR}/checkpoints}/${EXPERIMENT_NAME}"
ROLLOUT_SAVE_PATH="${SAVE_PATH}/rollout"

# ============================ WandB Configuration ============================
WANDB_API_KEY="${WANDB_API_KEY:-}"
if [ -n "$WANDB_API_KEY" ]; then
  wandb login --relogin "$WANDB_API_KEY"
  export WANDB_DIR="${SAVE_PATH}"
  LOGGER_CFG='[console,wandb]'
else
  LOGGER_CFG='[console]'
  echo "  WANDB_API_KEY "
fi

#   1   Bright Data 
if [ "${ARPO_USE_MOCK_SEARCH:-0}" = "1" ]; then
  SEARCH_CLASS_PATH="verl.workers.agent.tools.mock_search_tool.MockSearchTool"
  echo "  ARPO_USE_MOCK_SEARCH=1  MockSearchTool "
else
  SEARCH_CLASS_PATH="verl.workers.agent.tools.search_tool.BingSearchTool"
fi

# ============================ Data files check ============================
if [ ! -f "$TRAIN_FILES" ] || [ ! -f "$VALID_FILES" ]; then
  echo " : ${TRAIN_FILES}   ${VALID_FILES}"
  echo " : bash ${PARENT_DIR}/rl_datasets/download_reasoning_data.sh"
  exit 1
fi

if [ "${ARPO_USE_MOCK_SEARCH:-0}" != "1" ]; then
  if [ -z "${ARPO_SEARCH_API_KEY:-}" ] || [ -z "${ARPO_SEARCH_ZONE:-}" ]; then
    echo " :   ARPO_SEARCH_API_KEY / ARPO_SEARCH_ZONE "
    echo "        Bright Data : export ARPO_USE_MOCK_SEARCH=1"
  fi
fi

mkdir -p "$SAVE_PATH" "$ROLLOUT_SAVE_PATH"

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

# ============================ Start Training ============================
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VALID_FILES}" \
    data.prompt_key="${PROMPT_KEY}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path="${ACTOR_MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    +actor_rollout_ref.actor.use_turn_clip="${USE_TURN_CLIP}" \
    +actor_rollout_ref.actor.turn_clip_ratio="${TURN_CLIP_RATIO}" \
    $([ -n "$TURN_CLIP_RATIO_LOW" ] && echo "+actor_rollout_ref.actor.turn_clip_ratio_low=${TURN_CLIP_RATIO_LOW} \\") \
    $([ -n "$TURN_CLIP_RATIO_HIGH" ] && echo "+actor_rollout_ref.actor.turn_clip_ratio_high=${TURN_CLIP_RATIO_HIGH} \\") \
    "${ACTOR_PARAM_OFFLOAD_ARG}" \
    "${ACTOR_OPT_OFFLOAD_ARG}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="${ROLLOUT_NAME}" \
    actor_rollout_ref.rollout.mode="${ROLLOUT_MODE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ARPO_VLLM_GPU_MEMORY_UTIL}" \
    actor_rollout_ref.rollout.max_num_seqs="${ARPO_VLLM_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.initial_rollouts="${INITIAL_ROLLOUTS}" \
    actor_rollout_ref.rollout.beam_size="${BEAM_SIZE}" \
    actor_rollout_ref.rollout.branch_probability="${BRANCH_PROBABILITY}" \
    actor_rollout_ref.rollout.entropy_weight="${Entropy_weight}" \
    actor_rollout_ref.rollout.tools.tool_instances.search.params.cache_file="${SEARCH_CACHE_PATH}" \
    actor_rollout_ref.rollout.tools.tool_instances.search.class_path="${SEARCH_CLASS_PATH}" \
    actor_rollout_ref.rollout.multi_turn.enable="${ENABLE_MULTI_TURN}" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${MULTI_TURN_TOOL_CONFIG_STUB}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager="${REWARD_MANAGER}" \
    custom_reward_function.path="${CUSTOM_REWARD_FUNCTION_PATH}" \
    custom_reward_function.name="${CUSTOM_REWARD_FUNCTION_NAME}" \
    trainer.critic_warmup=0 \
    trainer.logger="${LOGGER_CFG}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.default_local_dir="${SAVE_PATH}" \
    trainer.val_before_train=False \
    trainer.rollout_data_dir="${ROLLOUT_SAVE_PATH}" \
    hydra.run.dir="${SAVE_PATH}/outputs" 2>&1 | tee "${SAVE_PATH}/run.log"
