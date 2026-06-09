#!/usr/bin/env bash
# ===========================================================================
# Round-2 evaluation: complete the trajectory + verify the two single-ckpt
# peaks (v8@50 HLE 0.34%, T2-C@25 GAIA 0.44%) under best@8 sampling.
# Plan (~80-100 min total):
#   1) v8 supplements: step 75, 125  (best@4, n=4 — match existing rows)
#   2) T2-C supplements: step 125, 150  (best@4, n=4)
#   3) v8 step 50 re-eval at n=8  (key paper datapoint, lower outlier risk)
#   4) T2-C step 25 re-eval at n=8 (key paper datapoint, lower outlier risk)
# Each n=4 eval ~10 min, each n=8 eval ~14 min, plus HF convert ~5 min for
# any uncached ckpt.
# Usage:
#   tmux new -d -s eval_r2 'bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate arpo && cd ${ARPO_ROOT} && bash scripts/batch_eval_round2.sh"'
# ===========================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"

mkdir -p "${PARENT_DIR}/eval_results"
AGG="${PARENT_DIR}/eval_results/round2_summary.txt"

V8_EXP="gtpo_3b_deepsearch_v8_janus"
GRPO_EXP="gtpo_3b_deepsearch_v8_grpo_paired"

{
  echo "# Round-2 batch eval — paper §7.5 supplements"
  echo "# generated: $(date -u +%FT%TZ)"
  echo "# config:    T=0.6, mock_search; n varies (best@4 vs best@8)"
  echo
  echo "# Existing baselines:"
  echo "#   T2-A SFT only step 273   GAIA f1 best@4 = 0.00%   HLE f1 best@4 = 0.30%"
  echo "#   T2-B v8  JanusPO step 50  GAIA f1 best@4 = 0.00%   HLE f1 best@4 = 0.34%"
  echo "#   T2-C GRPO step 25         GAIA f1 best@4 = 0.44%   HLE f1 best@4 = 0.00%"
  echo
} > "${AGG}"

run_one_eval () {
  # args: exp step n suffix
  local EXP="$1" STEP="$2" N="$3" SUFFIX="$4"
  local CKPT_DIR="${PARENT_DIR}/checkpoints/${EXP}/global_step_${STEP}"
  if [ ! -d "${CKPT_DIR}/actor" ]; then
    echo "[skip] ${EXP} step_${STEP}: missing actor dir"
    echo "# ${EXP} step ${STEP}: SKIPPED (missing actor dir)" >> "${AGG}"
    return
  fi

  local NAME="${EXP}_step_${STEP}${SUFFIX}"
  echo "================================================================"
  echo " EVAL ${NAME}  (n=${N})  $(date -u +%T)"
  echo "================================================================"
  ARPO_RL_EXP="${EXP}" \
  ARPO_RL_STEP="${STEP}" \
  ARPO_EVAL_N="${N}" \
  ARPO_RL_EVAL_SUFFIX="${SUFFIX}" \
    bash "${SCRIPT_DIR}/eval_rl_ckpt.sh" 2>&1 | tail -80
  local EC=$?
  if [ $EC -ne 0 ]; then
    echo "[err] eval_rl_ckpt.sh exited $EC for ${NAME}"
    echo "# ${NAME}: ERROR exit=${EC}" >> "${AGG}"
    return
  fi

  local SUM="${PARENT_DIR}/eval_results/${NAME}_summary.txt"
  if [ -f "${SUM}" ]; then
    {
      echo "================================================================"
      echo "# ${NAME}  (n=${N}, eval @ $(date -u +%T))"
      grep -E "f1 best@|f1 mean@|reward best@|reward mean@" "${SUM}" | head -20
      echo
    } >> "${AGG}"
  fi
}

# 1-2) Trajectory supplements (n=4)
run_one_eval "${V8_EXP}"   75  4 ""
run_one_eval "${V8_EXP}"   125 4 ""
run_one_eval "${GRPO_EXP}" 125 4 ""
run_one_eval "${GRPO_EXP}" 150 4 ""

# 3-4) Key-ckpt outlier check (n=8 — independent rollouts, suffixed name)
run_one_eval "${V8_EXP}"   50 8 "_n8"
run_one_eval "${GRPO_EXP}" 25 8 "_n8"

echo
echo "================================================================"
echo " ROUND-2 ALL DONE — aggregate at: ${AGG}"
echo "================================================================"
echo
cat "${AGG}"
