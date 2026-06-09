#!/usr/bin/env bash
# ===========================================================================
# Generic val-only evaluation on any parquet dataset using a HuggingFace-format
# checkpoint. Reuses verl's `trainer.val_only=True` path so we don't duplicate
# vLLM inference logic. Works for:
#   - MATH-500   (rl_datasets/math500/test.parquet)
#   - GSM8K      (rl_datasets/gsm8k/test.parquet)
#   - any parquet whose `data_source` field is wired in
#     verl.utils.reward_score.default_compute_score

# Inputs (env):
#   ARPO_EVAL_CKPT      *required*  HF-format model dir (config.json + weights)
#   ARPO_EVAL_PARQUET   *required*  test parquet path
#   ARPO_EVAL_NAME      optional    wandb run / log name          (default eval_<ts>)
#   ARPO_EVAL_TEMP      optional    rollout temperature           (default 0.0)
#   ARPO_EVAL_N         optional    num samples per prompt        (default 1)
#   ARPO_EVAL_MAX_RESP  optional    max response length           (default 1024)
#   ARPO_EVAL_MAX_PROMPT optional   max prompt length             (default 1024)
#   ARPO_EVAL_BATCH     optional    train_batch (only defines chunking) (default 64)
#   ARPO_VLLM_GPU_MEMORY_UTIL  optional                           (default 0.80)

# Examples:
#   # 3B champion on MATH-500
#   ARPO_EVAL_CKPT=${ARPO_ROOT}/checkpoints/plan_c_asymkl_v2_ub25_v1/hf_step200 \
#   ARPO_EVAL_PARQUET=${ARPO_ROOT}/rl_datasets/math500/test.parquet \
#   ARPO_EVAL_NAME=janus3b_math500_step200 \
#   bash scripts/eval_val_only.sh

#   # 7B champion on MATH-500
#   ARPO_EVAL_CKPT=${ARPO_ROOT}/checkpoints/plan_c_asymkl_v2_7b_v1/hf_step50 \
#   ARPO_EVAL_PARQUET=${ARPO_ROOT}/rl_datasets/math500/test.parquet \
#   ARPO_EVAL_NAME=janus7b_math500_step50 \
#   bash scripts/eval_val_only.sh

#   # zero-shot 3B instruct on MATH-500
#   ARPO_EVAL_CKPT=$HOME/models/Qwen2.5-3B-Instruct \
#   ARPO_EVAL_PARQUET=${ARPO_ROOT}/rl_datasets/math500/test.parquet \
#   ARPO_EVAL_NAME=qwen3b_zs_math500 \
#   bash scripts/eval_val_only.sh
# ===========================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"

if [ -f "${SCRIPT_DIR}/local.env" ]; then
  set -a; source "${SCRIPT_DIR}/local.env"; set +a
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

# ── Inputs ────────────────────────────────────────────────────────────────
: "${ARPO_EVAL_CKPT:?  ARPO_EVAL_CKPT   HF-format  }"
: "${ARPO_EVAL_PARQUET:?  ARPO_EVAL_PARQUET   test.parquet}"
if [ ! -d "$ARPO_EVAL_CKPT" ] || [ ! -f "$ARPO_EVAL_CKPT/config.json" ]; then
  echo " : ARPO_EVAL_CKPT=$ARPO_EVAL_CKPT   HF  "; exit 1
fi
if [ ! -f "$ARPO_EVAL_PARQUET" ]; then
  echo " :  : $ARPO_EVAL_PARQUET"; exit 1
fi

EVAL_NAME="${ARPO_EVAL_NAME:-eval_$(date +%Y%m%d_%H%M%S)}"
EVAL_TEMP="${ARPO_EVAL_TEMP:-0.0}"
EVAL_N="${ARPO_EVAL_N:-1}"
if awk -v t="$EVAL_TEMP" 'BEGIN{exit !(t>0)}'; then
  EVAL_DO_SAMPLE=True
  ROLLOUT_TEMP="$EVAL_TEMP"
else
  EVAL_DO_SAMPLE=False
  ROLLOUT_TEMP=1.0
fi
MAX_PROMPT_LENGTH="${ARPO_EVAL_MAX_PROMPT:-1024}"
MAX_RESPONSE_LENGTH="${ARPO_EVAL_MAX_RESP:-1024}"
TRAIN_BATCH_SIZE="${ARPO_EVAL_BATCH:-64}"
ARPO_VLLM_GPU_MEMORY_UTIL="${ARPO_VLLM_GPU_MEMORY_UTIL:-0.80}"
ARPO_VLLM_MAX_NUM_SEQS="${ARPO_VLLM_MAX_NUM_SEQS:-128}"
ARPO_VLLM_MAX_BATCH_TOKENS="${ARPO_VLLM_MAX_BATCH_TOKENS:-8192}"

SAVE_PATH="${ARPO_CHECKPOINT_DIR:-${PARENT_DIR}/checkpoints}/_eval/${EVAL_NAME}"
mkdir -p "$SAVE_PATH"
echo "========================================================"
echo " val-only eval"
echo "  ckpt    : ${ARPO_EVAL_CKPT}"
echo "  dataset : ${ARPO_EVAL_PARQUET}"
echo "  name    : ${EVAL_NAME}"
echo "  temp    : ${EVAL_TEMP}   n=${EVAL_N}   max_resp=${MAX_RESPONSE_LENGTH}"
echo "  out     : ${SAVE_PATH}/run.log"
echo "========================================================"

CONFIG_PATH="${PARENT_DIR}/scripts/config"
CONFIG_NAME="ppo_trainer.yaml"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="${ARPO_EVAL_PARQUET}" \
    data.val_files="${ARPO_EVAL_PARQUET}" \
    data.prompt_key="prompt" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path="${ARPO_EVAL_CKPT}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((2 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="vllm" \
    actor_rollout_ref.rollout.mode="sync" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ARPO_VLLM_GPU_MEMORY_UTIL}" \
    actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.rollout.max_num_seqs="${ARPO_VLLM_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ARPO_VLLM_MAX_BATCH_TOKENS}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMP}" \
    actor_rollout_ref.rollout.n="${EVAL_N}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample="${EVAL_DO_SAMPLE}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${EVAL_TEMP}" \
    actor_rollout_ref.rollout.val_kwargs.n="${EVAL_N}" \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.initial_rollouts=1 \
    actor_rollout_ref.rollout.beam_size=1 \
    actor_rollout_ref.rollout.branch_probability=0.0 \
    '~actor_rollout_ref.rollout.tools.tool_instances' \
    actor_rollout_ref.rollout.tools.verbose_logging=False \
    actor_rollout_ref.rollout.multi_turn.enable=false \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((2 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager="naive" \
    trainer.critic_warmup=0 \
    trainer.logger='[console]' \
    trainer.project_name="eval_val_only" \
    trainer.experiment_name="${EVAL_NAME}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.log_val_generations=16 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${SAVE_PATH}" \
    trainer.val_before_train=True \
    +trainer.val_only=True \
    trainer.rollout_data_dir="${SAVE_PATH}/rollout" \
    hydra.run.dir="${SAVE_PATH}/outputs" 2>&1 | tee "${SAVE_PATH}/run.log"

echo ""
echo "========================================================"
echo "   run.log  :"
grep -E "val/.*score|val_metrics|Initial validation" "${SAVE_PATH}/run.log" | tail -20
echo "========================================================"
