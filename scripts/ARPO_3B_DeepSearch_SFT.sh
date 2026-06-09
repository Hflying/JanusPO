#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# ARPO 3B DeepSearch SFT  
#   RL rollout   rejection-sampled   3B   SFT  
#         DeepSearch  <think>/<search>/<result>/<answer>/<boxed{}> 
#         GTPO RL  gtpo_3b_deepsearch_v4 


#   #   SFT  
#   cd ARPO/scripts
#   python build_sft_data.py \
#       --rollout_dir  ../checkpoints/gtpo_3b_deepsearch_v3/rollout \
#       --output_dir   ../rl_datasets/sft \
#       --score_thresh 0.3 \
#       --dedup

#   #   SFT  
#   bash ARPO_3B_DeepSearch_SFT.sh

#  1× H100 80GB 
#   ~200   × 2 epoch → ~100   × 6 s/  ≈ 10  
#   ~500   × 2 epoch → ~250   × 6 s/  ≈ 25  
#    5k  2 epoch → ~2500   ≈ 4  
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ──   ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "${SCRIPT_DIR}")"

#   RL checkpoint  gtpo_3b_deepsearch_v3/global_step_xxx 
MODEL_PATH="${ARPO_MODEL_PATH:-${HOME}/models/Qwen2.5-3B-Instruct}"
#  HuggingFace   "../xxx"  
MODEL_PATH="$(cd "${MODEL_PATH}" 2>/dev/null && pwd || realpath "${MODEL_PATH}" 2>/dev/null || echo "${MODEL_PATH}")"

# SFT  
# - rl_datasets/sft_mix/  ← prepare_sft_mix.py   ThornZ 3k + rollout 94 
# - rl_datasets/sft/      ← build_sft_data.py   rollout  
SFT_DATA_DIR="${ARPO_SFT_DATA_DIR:-${PARENT_DIR}/rl_datasets/sft_mix}"
TRAIN_FILE="${SFT_DATA_DIR}/train.parquet"
VAL_FILE="${SFT_DATA_DIR}/val.parquet"

#   &  
EXPERIMENT_NAME="${ARPO_SFT_EXPERIMENT_NAME:-deepsearch_sft_v1}"
OUTPUT_DIR="${PARENT_DIR}/checkpoints/${EXPERIMENT_NAME}"

# ── GPU   ──────────────────────────────────────────────────────────────────
N_GPUS="${ARPO_N_GPUS:-1}"

# ──   ────────────────────────────────────────────────────────────────────
#   batch size = TRAIN_BATCH_SIZE  N_GPUS  
TRAIN_BATCH_SIZE="${ARPO_SFT_TRAIN_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE_PER_GPU="${ARPO_SFT_MICRO_BATCH:-4}"
MAX_LENGTH="${ARPO_SFT_MAX_LENGTH:-4096}"
TOTAL_EPOCHS="${ARPO_SFT_TOTAL_EPOCHS:-3}"
LR="${ARPO_SFT_LR:-1e-5}"
WARMUP_RATIO="${ARPO_SFT_WARMUP_RATIO:-0.1}"
SAVE_FREQ="${ARPO_SFT_SAVE_FREQ:-50}"     # -1 =  

# ──   ──────────────────────────────────────────────────────────────
if [ ! -d "${MODEL_PATH}" ]; then
    echo "❌   ${MODEL_PATH}"
    echo "      ARPO_MODEL_PATH  "
    echo "      unset ARPO_MODEL_PATH"
    echo "      #  "
    echo "      export ARPO_MODEL_PATH=\${HOME}/models/Qwen2.5-3B-Instruct"
    exit 1
fi

# ──   ──────────────────────────────────────────────────────────────
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "❌    SFT  ${TRAIN_FILE}"
    echo "      build_sft_data.py   ARPO_SFT_DATA_DIR"
    exit 1
fi


if [ ! -f "${VAL_FILE}" ]; then
    echo "⚠️     ${VAL_FILE} "
    VAL_FILE="${TRAIN_FILE}"
fi

echo "════════════════════════════════════════════════════════════"
echo "  ARPO 3B DeepSearch SFT  "
echo "   :       ${MODEL_PATH}"
echo "   :   ${TRAIN_FILE}"
echo "   :   ${VAL_FILE}"
echo "   :   ${OUTPUT_DIR}"
echo "  GPU  :   ${N_GPUS}"
echo "  batch( ): ${TRAIN_BATCH_SIZE}  micro/GPU: ${MICRO_BATCH_SIZE_PER_GPU}"
echo "  max_len:    ${MAX_LENGTH}  epochs: ${TOTAL_EPOCHS}  lr: ${LR}"
echo "════════════════════════════════════════════════════════════"

# ──   verl   hydra config_path   sft_trainer.yaml ──────────
cd "${PARENT_DIR}/verl_arpo_entropy"

#   Hydra  
export HYDRA_FULL_ERROR=1

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${N_GPUS}" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.max_length="${MAX_LENGTH}" \
    data.truncation=right \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    model.partial_pretrain="${MODEL_PATH}" \
    model.enable_gradient_checkpointing=true \
    model.fsdp_config.model_dtype=bf16 \
    model.fsdp_config.cpu_offload=false \
    optim.lr="${LR}" \
    optim.warmup_steps_ratio="${WARMUP_RATIO}" \
    optim.lr_scheduler=cosine \
    optim.clip_grad=1.0 \
    trainer.project_name=arpo-deepsearch-sft \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq=10 \
    "trainer.logger=[console]"

echo "✅  SFT  ${OUTPUT_DIR}"
echo ""
echo "  SFT checkpoint   GTPO RL  "
echo "  export ARPO_MODEL_PATH=${OUTPUT_DIR}/global_step_LAST"
echo "  bash ARPO_3B_DeepSearch_GTPO.sh"
