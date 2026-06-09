

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
echo " : $SCRIPT_DIR"


# ============================   ============================

export PYTHONUNBUFFERED=1            
export HYDRA_FULL_ERROR=1           
export VLLM_ATTENTION_BACKEND=XFORMERS 
export VERL_LOGGING_LEVEL=DEBUG
export MKL_SERVICE_FORCE_INTEL=1    
export MKL_THREADING_LAYER=GNU       
export RAY_memory_usage_threshold=0.8  
export RAY_memory_monitor_refresh_ms=0 


export http_proxy=http://oversea-squid2.ko.txyun:11080 
export https_proxy=http://oversea-squid2.ko.txyun:11080 
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

#  Python 
export PYTHONPATH=<your_path_to_ARPO>/verl_arpo_entropy:$PYTHONPATH

# ============================   ============================

PROJECT_NAME="deep_research"
EXPERIMENT_NAME="qwen3_sft5.4w_global_16_init_8_beam_2_random_0.5_14B"


CONFIG_PATH="<your_path_to_ARPO>/scripts/config" #config , 
CONFIG_NAME="ppo_trainer_dr.yaml"
# /mmu_nlp_ssd/makai05/DeepResearch/train_rl/config/ppo_trainer_dr.yaml

NNODES=1                            
N_GPUS_PER_NODE=8                   

# ============================   ============================

PROMPT_KEY="prompt"                 #  
TRAIN_BATCH_SIZE=64                #  
PPO_MINI_BATCH_SIZE=8              # PPO 
MAX_PROMPT_LENGTH=2000             #  
MAX_RESPONSE_LENGTH=6192       #  


TRAIN_FILES="<your_path_to_ARPO>/rl_datasets/hard_search_1k.parquet"
VALID_FILES=["<your_path_to_ARPO>/rl_datasets/gaia_test.parquet","<your_path_to_ARPO>/rl_datasets/hle_test.parquet"]

# ============================   ============================
# Actor 
ACTOR_MODEL_PATH="<your_14B_model_path>"
# ============================ Rollout  ==========================
# Rollout 
ROLLOUT_NAME="vllm"                 #  vllm 
ROLLOUT_MODE="sync_with_tool"       #  
ROLLOUT_N=12                         #  
INITIAL_ROLLOUTS=6                 #  rollout 
BEAM_SIZE=2                        # beam size
BRANCH_PROBABILITY=0.5             # branch probability
Entropy_weight=0.2
# ============================ Rollout Tools  ==========================
SEARCH_CACHE_PATH="<your_path_to_ARPO>/search_cache/search_cache.json" # Modify

# ============================   ==========================

REWARD_MANAGER="naive"              #  
CUSTOM_REWARD_FUNCTION_PATH="<your_path_to_ARPO>/verl_arpo_entropy/verl/utils/reward_score/deep_research.py"
CUSTOM_REWARD_FUNCTION_NAME="compute_score"

# ============================   ============================

TOTAL_EPOCHS=5                      #  
SAVE_FREQ=5                        #  
TEST_FREQ=5                        #  

# ============================   ============================

SAVE_PATH="<your_checkpoint_save_dir>/rl/${EXPERIMENT_NAME}"
ROLLOUT_SAVE_PATH="${SAVE_PATH}/rollout"

# ============================ WandB  ============================
# WandB 
WANDB_API_KEY="<your_wandb_key>" # Modify your wandb key

# ============================   ============================
#  WandB API 
if [ "$WANDB_API_KEY" != "" ]; then
    wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi


if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p $SAVE_PATH
fi

#  rollout 
if [ ! -d "$ROLLOUT_SAVE_PATH" ]; then
    mkdir -p $ROLLOUT_SAVE_PATH
fi



if [ "${ARPO_USE_MOCK_SEARCH:-0}" = "1" ]; then
  SEARCH_CLASS_PATH="verl.workers.agent.tools.mock_search_tool.MockSearchTool"
else
  SEARCH_CLASS_PATH="verl.workers.agent.tools.search_tool.BingSearchTool"
fi


# ============================   ============================
python3 -m verl.trainer.main_ppo \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VALID_FILES} \
    data.prompt_key=${PROMPT_KEY} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    actor_rollout_ref.rollout.mode=${ROLLOUT_MODE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.initial_rollouts=${INITIAL_ROLLOUTS} \
    actor_rollout_ref.rollout.beam_size=${BEAM_SIZE} \
    actor_rollout_ref.rollout.branch_probability=${BRANCH_PROBABILITY} \
    actor_rollout_ref.rollout.entropy_weight=${Entropy_weight} \
    +actor_rollout_ref.rollout.tools.tool_instances.search.params.cache_file=${SEARCH_CACHE_PATH} \
    +actor_rollout_ref.rollout.tools.tool_instances.search.class_path=${SEARCH_CLASS_PATH} \
    actor_rollout_ref.rollout.multi_turn.enable=${ENABLE_MULTI_TURN} \
    +actor_rollout_ref.rollout.multi_turn.tool_config_path=${PARENT_DIR}/scripts/config/arpo_multiturn_tool_stub.yaml \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=${REWARD_MANAGER} \
    custom_reward_function.path=${CUSTOM_REWARD_FUNCTION_PATH} \
    custom_reward_function.name=${CUSTOM_REWARD_FUNCTION_NAME} \
    trainer.critic_warmup=0 \
    trainer.logger="[console, wandb]" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=${ROLLOUT_SAVE_PATH} \
    hydra.run.dir=${SAVE_PATH}/outputs 2>&1 | tee ${SAVE_PATH}/run.log 
