#!/usr/bin/env bash
# ===========================================================================
# Sequentially run eval_rl_ckpt.sh for several v8_janus ckpts and aggregate
# the paper §7.5 Table 3 numbers.
# Targets: 25, 50, 100, 150  (the 4 representative steps spanning the
# healthy → peak → start-of-decay window). Each eval takes ~12 min
# (~5 min FSDP→HF convert + ~7 min multi-turn val_only with n=4 sampling).
# Total wallclock: ~50 min.
# Usage:
#   tmux new -s eval_v8 'bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate arpo && cd ${ARPO_ROOT} && bash scripts/batch_eval_v8.sh"'
# ===========================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"

EXP="gtpo_3b_deepsearch_v8_janus"
STEPS=(25 50 100 150)

mkdir -p "${PARENT_DIR}/eval_results"
AGG="${PARENT_DIR}/eval_results/${EXP}_batch_summary.txt"

{
  echo "# v8 batch eval — paper §7.5 Table 3 trajectory"
  echo "# generated: $(date -u +%FT%TZ)"
  echo "# config:    best@4, T=0.6, n=4, mock_search (matches T2-A SFT-only baseline exactly)"
  echo
  echo "# baseline (from T2-A run earlier):"
  echo "#   SFT only (step 273)   GAIA f1 best@4 = 0.00%   HLE f1 best@4 = 0.30%"
  echo "#                         GAIA reward best@4 = -0.981   HLE reward best@4 = -0.976"
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
  echo " EVAL ckpt step_${STEP}   ($(date -u +%T))"
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
