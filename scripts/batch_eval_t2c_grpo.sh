#!/usr/bin/env bash
# ===========================================================================
# Sequentially run eval_rl_ckpt.sh for several v8_grpo_paired (T2-C) ckpts
# and aggregate the paper §7.5 Table 3 row 2 numbers.
# Targets: 25, 50, 75, 100  (mirror v8 trajectory; step 50 is the key
# paired comparison point — same SFT init, same hparams, only GTPO
# components differ).
# Each eval ~12 min (~5 min FSDP→HF + ~7 min multi-turn val_only n=4).
# Total wallclock: ~50 min.
# Usage:
#   tmux new -s eval_t2c 'bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate arpo && cd ${ARPO_ROOT} && bash scripts/batch_eval_t2c_grpo.sh"'
# ===========================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"

EXP="gtpo_3b_deepsearch_v8_grpo_paired"
STEPS=(25 50 75 100)

mkdir -p "${PARENT_DIR}/eval_results"
AGG="${PARENT_DIR}/eval_results/${EXP}_batch_summary.txt"

{
  echo "# T2-C (vanilla GRPO) batch eval — paper §7.5 Table 3 row 2"
  echo "# generated: $(date -u +%FT%TZ)"
  echo "# config:    best@4, T=0.6, n=4, mock_search (matches T2-A & v8 exactly)"
  echo
  echo "# baselines for comparison:"
  echo "#   T2-A SFT only (step 273)   GAIA f1 best@4 = 0.00%   HLE f1 best@4 = 0.30%"
  echo "#                              GAIA reward    = -0.981  HLE reward    = -0.976"
  echo "#   T2-B v8 JanusPO step 50    GAIA f1 best@4 = 0.00%   HLE f1 best@4 = 0.34%"
  echo
} > "${AGG}"

for STEP in "${STEPS[@]}"; do
  CKPT_DIR="${PARENT_DIR}/checkpoints/${EXP}/global_step_${STEP}"
  if [ ! -d "${CKPT_DIR}/actor" ]; then
    echo "[skip] step_${STEP}: missing actor dir"
    echo "# step ${STEP}: SKIPPED (missing actor dir)" >> "${AGG}"
    continue
  fi

  echo "================================================================"
  echo " EVAL T2-C ckpt step_${STEP}   ($(date -u +%T))"
  echo "================================================================"
  ARPO_RL_EXP="${EXP}" ARPO_RL_STEP="${STEP}" \
    bash "${SCRIPT_DIR}/eval_rl_ckpt.sh" 2>&1 | tail -80
  EC=$?
  if [ $EC -ne 0 ]; then
    echo "[err] eval_rl_ckpt.sh exited $EC for step ${STEP}"
    echo "# step ${STEP}: ERROR exit=${EC}" >> "${AGG}"
    continue
  fi

  SUM="${PARENT_DIR}/eval_results/${EXP}_step_${STEP}_summary.txt"
  if [ -f "${SUM}" ]; then
    {
      echo "================================================================"
      echo "# step ${STEP} (eval @ $(date -u +%T))"
      grep -E "f1 best@4|reward best@4|f1 mean@4" "${SUM}" | head -12
      echo
    } >> "${AGG}"
  fi
done

echo
echo "================================================================"
echo " ALL DONE — aggregate at: ${AGG}"
echo "================================================================"
echo
cat "${AGG}"
