#!/usr/bin/env bash
# ===========================================================================
# End-to-end RL ckpt evaluation for paper §7.5 Table 3:
#   verl FSDP ckpt  →  HF model dir  →  multi-turn val_only eval

# Inputs (env):
#   ARPO_RL_EXP        *required*  experiment name, e.g. gtpo_3b_deepsearch_v6_janus
#   ARPO_RL_STEP       *required*  integer step, e.g. 25, 100, 250
#                                  (the dir global_step_<STEP>/actor must exist)
#   ARPO_RL_BASE_MODEL optional    base HF model for tokenizer & config
#                                  (default: SFT ckpt
#                                  checkpoints/deepsearch_sft_v1/global_step_273)
#   ARPO_RL_REUSE_HF   optional    if 1, skip convert when target hf dir exists
#                                  (default 1)
#   any ARPO_EVAL_*    optional    forwarded to eval_multiturn_deepsearch.sh
#                                  (e.g. ARPO_EVAL_TEMP, ARPO_EVAL_N,
#                                  ARPO_EVAL_VALID_FILES, ARPO_USE_MOCK_SEARCH)

# Outputs:
#   checkpoints/<exp>/global_step_<step>/hf/    ← merged HF dir
#   checkpoints/_eval/<exp>_step_<step>/run.log ← eval log
#   eval_results/<exp>_step_<step>_summary.txt  ← extracted GAIA/HLE numbers

# Examples:
#   # Eval the v6_janus ckpt at step 100
#   ARPO_RL_EXP=gtpo_3b_deepsearch_v6_janus ARPO_RL_STEP=100 \
#     bash scripts/eval_rl_ckpt.sh

#   # Eval the GRPO baseline ckpt
#   ARPO_RL_EXP=gtpo_3b_deepsearch_v6_grpo ARPO_RL_STEP=100 \
#     bash scripts/eval_rl_ckpt.sh

# Note: requires the GPU. Do not run while a training job is occupying memory.
# ===========================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"

: "${ARPO_RL_EXP:?  ARPO_RL_EXP  gtpo_3b_deepsearch_v6_janus }"
: "${ARPO_RL_STEP:?  ARPO_RL_STEP  100 }"

CKPT_ROOT="${PARENT_DIR}/checkpoints/${ARPO_RL_EXP}"
ACTOR_DIR="${CKPT_ROOT}/global_step_${ARPO_RL_STEP}/actor"
HF_DIR="${CKPT_ROOT}/global_step_${ARPO_RL_STEP}/hf"

if [ ! -d "$ACTOR_DIR" ]; then
  echo "[err] verl FSDP ckpt missing: ${ACTOR_DIR}"
  echo "      available steps under ${CKPT_ROOT}:"
  ls -1 "${CKPT_ROOT}" 2>/dev/null | grep '^global_step_' | sort -V || true
  exit 1
fi

BASE_MODEL="${ARPO_RL_BASE_MODEL:-${PARENT_DIR}/checkpoints/deepsearch_sft_v1/global_step_273}"
if [ ! -f "${BASE_MODEL}/config.json" ]; then
  echo "[err] base model not a valid HF dir: ${BASE_MODEL}"; exit 1
fi

REUSE_HF="${ARPO_RL_REUSE_HF:-1}"

# ── Step 1: FSDP → HF ──────────────────────────────────────────────────────
need_convert=1
if [ "$REUSE_HF" = "1" ] && [ -f "${HF_DIR}/config.json" ]; then
  echo "[skip-convert] reuse existing ${HF_DIR}"
  need_convert=0
fi

if [ "$need_convert" = "1" ]; then
  mkdir -p "${HF_DIR}"
  echo "================================================================"
  echo " FSDP → HF conversion"
  echo "  src:      ${ACTOR_DIR}"
  echo "  base:     ${BASE_MODEL}"
  echo "  dst:      ${HF_DIR}"
  echo "================================================================"
  # convert script imports `verl.*`; inject the in-repo verl onto PYTHONPATH
  PYTHONPATH="${PARENT_DIR}/verl_arpo_entropy:${PYTHONPATH:-}" \
  python3 "${PARENT_DIR}/merge_ckpt/convert_checkpoint_from_verl_to_hf.py" merge \
      --backend "fsdp" \
      --hf_model_path "${BASE_MODEL}" \
      --local_dir "${ACTOR_DIR}" \
      --target_dir "${HF_DIR}" || {
    echo "[err] conversion failed"; exit 2;
  }
  if [ ! -f "${HF_DIR}/config.json" ]; then
    echo "[err] merge produced no config.json in ${HF_DIR}"; exit 2
  fi
fi

# ── Step 2: multi-turn val_only eval ──────────────────────────────────────
# Optional ARPO_RL_EVAL_SUFFIX lets caller distinguish multiple eval runs of
# the same ckpt (e.g. n=4 vs n=8) so summaries do not overwrite each other.
EVAL_NAME="${ARPO_RL_EXP}_step_${ARPO_RL_STEP}${ARPO_RL_EVAL_SUFFIX:-}"
export ARPO_EVAL_CKPT="${HF_DIR}"
export ARPO_EVAL_NAME="${EVAL_NAME}"
# eval defaults match T2-A so all rows of Table 3 are comparable
export ARPO_EVAL_TEMP="${ARPO_EVAL_TEMP:-0.6}"
export ARPO_EVAL_N="${ARPO_EVAL_N:-4}"
export ARPO_EVAL_MAX_PROMPT="${ARPO_EVAL_MAX_PROMPT:-2000}"
export ARPO_EVAL_MAX_RESP="${ARPO_EVAL_MAX_RESP:-1500}"

bash "${SCRIPT_DIR}/eval_multiturn_deepsearch.sh"
EXIT_CODE=$?

if [ "$EXIT_CODE" -ne 0 ]; then
  echo "[err] eval_multiturn_deepsearch.sh exited with $EXIT_CODE"
  exit "$EXIT_CODE"
fi

# ── Step 3: extract paper numbers ──────────────────────────────────────────
EVAL_LOG="${PARENT_DIR}/checkpoints/_eval/${EVAL_NAME}/run.log"
SUMMARY_DIR="${PARENT_DIR}/eval_results"
SUMMARY_FILE="${SUMMARY_DIR}/${EVAL_NAME}_summary.txt"
mkdir -p "${SUMMARY_DIR}"

if [ ! -f "${EVAL_LOG}" ]; then
  echo "[warn] eval log not found at ${EVAL_LOG}; cannot extract metrics"
  exit 0
fi

python3 - <<PY > "${SUMMARY_FILE}"
import re, json, sys
from pathlib import Path
log = Path("${EVAL_LOG}").read_text(errors="ignore")
keep = []
keys = re.findall(r"'(val-(?:core|aux)/[^']+)':\s*([-\d\.eE]+)", log)
seen = set()
for k, v in keys:
    if k in seen: continue
    seen.add(k)
    keep.append((k, float(v)))
def get(name):
    for k, v in keep:
        if k == name: return v
    return None
# best@4 reward & f1, plus mean@4 for completeness
# task tags follow the eval_multiturn pipeline naming (DR_<task>)
header = "# RL ckpt eval summary  exp=${ARPO_RL_EXP}  step=${ARPO_RL_STEP}\n"
print(header)
print("# raw extracted (sorted)")
for k, v in keep:
    if k.endswith("/std@4") or k.endswith("/std"): continue
    print(f"  {k}: {v}")
print()
print("# ---- paper §7.5 Table 3 row ----")
for task in ("DR_gaia", "DR_hle_100"):
    f1_best4 = get(f"val-aux/{task}/f1_score/best@4/mean")
    f1_mean4 = get(f"val-aux/{task}/f1_score/mean@4")
    rwd_best4 = get(f"val-aux/{task}/reward/best@4/mean")
    rwd_mean4 = get(f"val-aux/{task}/reward/mean@4")
    pretty = "GAIA" if task == "DR_gaia" else "HLE"
    print(f"# {pretty}:")
    if f1_best4 is not None:
        print(f"#   f1 best@4   = {100*f1_best4:.2f}%   <- reported in paper")
    if f1_mean4 is not None:
        print(f"#   f1 mean@4   = {100*f1_mean4:.2f}%")
    if rwd_best4 is not None:
        print(f"#   reward best@4 = {rwd_best4:+.3f}")
    if rwd_mean4 is not None:
        print(f"#   reward mean@4 = {rwd_mean4:+.3f}")
PY

echo "================================================================"
echo " Eval done"
echo "  hf  ckpt:  ${HF_DIR}"
echo "  summary :  ${SUMMARY_FILE}"
echo "================================================================"
echo ""
cat "${SUMMARY_FILE}"
