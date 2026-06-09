#!/usr/bin/env bash
# ===========================================================================
# Round-3 verification: n=8 best@8 on the two surviving "peak" datapoints
# from round-2 best@4:
#   - v8 step 75:   GAIA 0.64% / HLE 0.80%   (n=4)
#   - T2-C step 150: GAIA 0.90% / HLE 0.00%  (n=4)
# round-2 already showed that v8 step 50 (HLE 0.34%) and T2-C step 25
# (GAIA 0.44%) collapse to 0% under n=8 → those were sampling flukes.
# This run answers: do the two newly-discovered peaks survive?
# Each n=8 eval ~14 min, 2 ckpts → ~28 min total.
# Plus we rerun SFT-only at n=8 for an apples-to-apples reward-level baseline.
# Usage:
#   tmux new -d -s eval_r3 'bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate arpo && cd ${ARPO_ROOT} && bash scripts/batch_eval_round3.sh"'
# ===========================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"

mkdir -p "${PARENT_DIR}/eval_results"
AGG="${PARENT_DIR}/eval_results/round3_summary.txt"

V8_EXP="gtpo_3b_deepsearch_v8_janus"
GRPO_EXP="gtpo_3b_deepsearch_v8_grpo_paired"

{
  echo "# Round-3 verification — n=8 outlier check on round-2 peaks"
  echo "# generated: $(date -u +%FT%TZ)"
  echo "# config:    T=0.6, mock_search, n=8 (best@8 reward + f1)"
  echo
  echo "# Round-2 n=8 baselines:"
  echo "#   v8 step 50 n=8   GAIA reward = -0.938  HLE reward = -0.873   f1 ALL 0%"
  echo "#   GRPO step 25 n=8 GAIA reward = -0.931  HLE reward = -0.967   f1 ALL 0%"
  echo
  echo "# Round-2 n=4 peaks to verify:"
  echo "#   v8 step 75   GAIA f1 best@4 = 0.64%  HLE f1 best@4 = 0.80%  (★ key)"
  echo "#   GRPO step 150 GAIA f1 best@4 = 0.90% HLE f1 best@4 = 0.00%"
  echo
} > "${AGG}"

run_one_eval () {
  local EXP="$1" STEP="$2" N="$3" SUFFIX="$4"
  local CKPT_DIR="${PARENT_DIR}/checkpoints/${EXP}/global_step_${STEP}"
  if [ ! -d "${CKPT_DIR}/actor" ]; then
    echo "[skip] ${EXP} step_${STEP}: missing actor dir"
    echo "# ${EXP} step ${STEP}: SKIPPED" >> "${AGG}"
    return
  fi
  local NAME="${EXP}_step_${STEP}${SUFFIX}"
  echo "================================================================"
  echo " EVAL ${NAME}  (n=${N})  $(date -u +%T)"
  echo "================================================================"
  ARPO_RL_EXP="${EXP}" ARPO_RL_STEP="${STEP}" \
  ARPO_EVAL_N="${N}" ARPO_RL_EVAL_SUFFIX="${SUFFIX}" \
    bash "${SCRIPT_DIR}/eval_rl_ckpt.sh" 2>&1 | tail -80
  local EC=$?
  if [ $EC -ne 0 ]; then
    echo "# ${NAME}: ERROR exit=${EC}" >> "${AGG}"
    return
  fi
  local SUM="${PARENT_DIR}/eval_results/${NAME}_summary.txt"
  if [ -f "${SUM}" ]; then
    {
      echo "================================================================"
      echo "# ${NAME}  (n=${N}, eval @ $(date -u +%T))"
      grep -E "f1_score/(best@8|mean@8|best@4)/mean: " "${SUM}" | head -10
      grep -E "reward/(best@8|mean@8|best@4)/mean: "    "${SUM}" | head -10
      echo
    } >> "${AGG}"
  fi
}

# n=8 verification of round-2 peaks
run_one_eval "${V8_EXP}"   75  8 "_n8"
run_one_eval "${GRPO_EXP}" 150 8 "_n8"

echo
echo "================================================================"
echo " ROUND-3 ALL DONE — aggregate at: ${AGG}"
echo "================================================================"
echo
cat "${AGG}"
