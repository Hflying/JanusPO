#!/usr/bin/env bash
# ===========================================================================
# Multi-turn DeepSearch zero-shot / final-checkpoint evaluation.

# Reuses the multi-turn vLLM tool-calling rollout pipeline of
#   ARPO_3B_DeepSearch_GTPO.sh
# but runs `trainer.val_only=True`, so it performs ONE validation pass
# (vLLM rollout with tool calling on every prompt) and exits before any
# RL training step. Required for §7.5 baseline numbers (SFT-only) and any
# zero-shot DeepSearch eval.

# Inputs (env):
#   ARPO_EVAL_CKPT       *required*  HF-format model dir
#   ARPO_EVAL_NAME       optional    eval name (default eval_<ts>)
#   ARPO_EVAL_TEMP       optional    sampling temperature      (default 0.6)
#   ARPO_EVAL_N          optional    n samples per prompt      (default 4)
#   ARPO_EVAL_VALID_FILES optional   parquet list, format same as
#                                   ARPO_VALID_FILES in GTPO script
#                                   (default: gaia_test + hle_test)
#   ARPO_EVAL_MAX_PROMPT optional    max prompt length         (default 2000)
#   ARPO_EVAL_MAX_RESP   optional    max response length       (default 1500)
#   ARPO_TOOL_CALL_LIMIT optional    tool calls per rollout    (default 5)
#   ARPO_USE_MOCK_SEARCH optional    use mock search           (default 1)
#   ARPO_VLLM_GPU_MEMORY_UTIL optional                          (default 0.55)

# Examples:
#   # SFT-only zero-shot baseline on GAIA + HLE
#   ARPO_EVAL_CKPT=${ARPO_ROOT}/checkpoints/deepsearch_sft_v1/global_step_273 \
#   ARPO_EVAL_NAME=sft_only_zs_gaia_hle \
#   bash scripts/eval_multiturn_deepsearch.sh

#   # Final RL ckpt eval
#   ARPO_EVAL_CKPT=${ARPO_ROOT}/checkpoints/gtpo_3b_deepsearch_v6_janus/hf_step_NNN \
#   ARPO_EVAL_NAME=janus_v6_step_NNN_gaia_hle \
#   bash scripts/eval_multiturn_deepsearch.sh
# ===========================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"
echo "ARPO root: $PARENT_DIR"

if [ -f "${SCRIPT_DIR}/local.env" ]; then
  set -a; source "${SCRIPT_DIR}/local.env"; set +a
fi

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export RAY_memory_usage_threshold=0.8
export RAY_memory_monitor_refresh_ms=0
export VLLM_USE_V1=0

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

# ── Required inputs ─────────────────────────────────────────────────────
: "${ARPO_EVAL_CKPT:?  ARPO_EVAL_CKPT   HF  }"
if [ ! -d "$ARPO_EVAL_CKPT" ] || [ ! -f "$ARPO_EVAL_CKPT/config.json" ]; then
  echo "error: ARPO_EVAL_CKPT=$ARPO_EVAL_CKPT   HF  "
  exit 1
fi

EVAL_NAME="${ARPO_EVAL_NAME:-eval_$(date +%Y%m%d_%H%M%S)}"
EVAL_TEMP="${ARPO_EVAL_TEMP:-0.6}"
EVAL_N="${ARPO_EVAL_N:-4}"
if awk -v t="$EVAL_TEMP" 'BEGIN{exit !(t>0)}'; then
  EVAL_DO_SAMPLE=True
  ROLLOUT_TEMP="$EVAL_TEMP"
else
  EVAL_DO_SAMPLE=False
  ROLLOUT_TEMP=1.0
fi

MAX_PROMPT_LENGTH="${ARPO_EVAL_MAX_PROMPT:-2000}"
MAX_RESPONSE_LENGTH="${ARPO_EVAL_MAX_RESP:-1500}"
TRAIN_BATCH_SIZE="${ARPO_EVAL_BATCH:-8}"

VALID_FILES_DEFAULT="[\"${PARENT_DIR}/rl_datasets/gaia_test.parquet\",\"${PARENT_DIR}/rl_datasets/hle_test.parquet\"]"
VALID_FILES="${ARPO_EVAL_VALID_FILES:-$VALID_FILES_DEFAULT}"

# ── Search tool ─────────────────────────────────────────────────────────
ARPO_USE_MOCK_SEARCH="${ARPO_USE_MOCK_SEARCH:-1}"
SEARCH_CACHE_PATH="${ARPO_SEARCH_CACHE_PATH:-${PARENT_DIR}/search_cache/search_cache.json}"
if [ "$ARPO_USE_MOCK_SEARCH" = "1" ]; then
  SEARCH_CLASS_PATH="verl.workers.agent.tools.mock_search_tool.MockSearchTool"
else
  SEARCH_CLASS_PATH="verl.workers.agent.tools.search_tool.BingSearchTool"
fi

# ── vLLM rollout knobs (tighter than training to leave room for tool exec) ──
ARPO_VLLM_GPU_MEMORY_UTIL="${ARPO_VLLM_GPU_MEMORY_UTIL:-0.55}"
ARPO_VLLM_MAX_NUM_SEQS="${ARPO_VLLM_MAX_NUM_SEQS:-32}"
ARPO_VLLM_MAX_BATCH_TOKENS="${ARPO_VLLM_MAX_BATCH_TOKENS:-16000}"
TOOL_CALL_LIMIT="${ARPO_TOOL_CALL_LIMIT:-5}"
INITIAL_ROLLOUTS="${ARPO_INITIAL_ROLLOUTS:-4}"
BEAM_SIZE="${ARPO_BEAM_SIZE:-2}"
BRANCH_PROBABILITY="${ARPO_BRANCH_PROBABILITY:-0.0}"  # eval = no ARPO branch injection
ENTROPY_WEIGHT="${ARPO_ENTROPY_WEIGHT:-0.2}"

SAVE_PATH="${ARPO_CHECKPOINT_DIR:-${PARENT_DIR}/checkpoints}/_eval/${EVAL_NAME}"
mkdir -p "$SAVE_PATH"
LOG_FILE="${SAVE_PATH}/run.log"

CONFIG_PATH="${PARENT_DIR}/scripts/config"
CONFIG_NAME="ppo_trainer_dr.yaml"

cat <<EOF
================================================================
 multi-turn DeepSearch val-only eval
  ckpt          : ${ARPO_EVAL_CKPT}
  valid_files   : ${VALID_FILES}
  eval name     : ${EVAL_NAME}
  temp/n        : T=${EVAL_TEMP}  n=${EVAL_N}  do_sample=${EVAL_DO_SAMPLE}
  seq           : prompt=${MAX_PROMPT_LENGTH}  response=${MAX_RESPONSE_LENGTH}
  tool          : ${SEARCH_CLASS_PATH}  call_limit=${TOOL_CALL_LIMIT}
  vLLM mem_util : ${ARPO_VLLM_GPU_MEMORY_UTIL}
  out           : ${LOG_FILE}
================================================================
EOF

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="${PARENT_DIR}/rl_datasets/train_10k.parquet" \
    data.val_files="${VALID_FILES}" \
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
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((2 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="vllm" \
    actor_rollout_ref.rollout.mode="sync_with_tool" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ARPO_VLLM_GPU_MEMORY_UTIL}" \
    actor_rollout_ref.rollout.max_num_seqs="${ARPO_VLLM_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ARPO_VLLM_MAX_BATCH_TOKENS}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMP}" \
    actor_rollout_ref.rollout.n="${EVAL_N}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample="${EVAL_DO_SAMPLE}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${EVAL_TEMP}" \
    actor_rollout_ref.rollout.val_kwargs.n="${EVAL_N}" \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.initial_rollouts="${INITIAL_ROLLOUTS}" \
    actor_rollout_ref.rollout.beam_size="${BEAM_SIZE}" \
    actor_rollout_ref.rollout.branch_probability="${BRANCH_PROBABILITY}" \
    +actor_rollout_ref.rollout.entropy_weight="${ENTROPY_WEIGHT}" \
    +actor_rollout_ref.rollout.tools.tool_instances.search.params.cache_file="${SEARCH_CACHE_PATH}" \
    +actor_rollout_ref.rollout.tools.tool_instances.search.class_path="${SEARCH_CLASS_PATH}" \
    +actor_rollout_ref.rollout.tools.call_limit="${TOOL_CALL_LIMIT}" \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    +actor_rollout_ref.rollout.multi_turn.tool_config_path="${PARENT_DIR}/scripts/config/arpo_multiturn_tool_stub.yaml" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((2 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager="naive" \
    trainer.critic_warmup=0 \
    trainer.logger='[console]' \
    trainer.project_name="eval_multiturn" \
    trainer.experiment_name="${EVAL_NAME}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.log_val_generations=8 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${SAVE_PATH}" \
    trainer.val_before_train=True \
    +trainer.val_only=True \
    trainer.rollout_data_dir="${SAVE_PATH}/rollout" \
    hydra.run.dir="${SAVE_PATH}/outputs" 2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "================================================================"
echo "  grep :"
grep -oE "'val-core/[^']*reward/(mean|acc)@[0-9]+':\s*[0-9.]+" "$LOG_FILE" | sort -u | tail -20
echo "================================================================"

exit "$EXIT_CODE"
