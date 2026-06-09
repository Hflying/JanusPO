#!/usr/bin/env bash
# ============================================================================
# Sequential orchestrator for paper §6.2 strengthening:
#   phase 1 (≈1h):  MATH-500 cross-benchmark zero-shot on 5 ckpts
#   phase 2 (≈5h):  RLOO baseline GSM8K training (1 seed)
# The phases share the GH200 96 GB GPU so we MUST serialize.
# Total wall-clock ≈ 6h.
# The trainer launch uses `set -o pipefail` + explicit PIPESTATUS check so
# we don't repeat the previous silent-success bug where `python | tee`
# masked the trainer's non-zero exit.
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"

LOG_DIR="${PARENT_DIR}/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
ORCH_LOG="${LOG_DIR}/orch_math_rloo_${TS}.log"

log() { printf '[orch %s] %s\n' "$(date +%F" "%T)" "$*" | tee -a "$ORCH_LOG"; }

log "========================================================"
log " orchestrator: MATH-500 evals → RLOO training"
log " host: $(hostname)   gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
log " log:    ${ORCH_LOG}"
log "========================================================"

# sanity
if ! python3 -c "import ray" 2>/dev/null; then
  log "FATAL: please run 'conda activate arpo' before invoking this script."
  exit 2
fi
ACTOR_MODEL_PATH="${ARPO_ACTOR_MODEL_PATH:-$HOME/models/Qwen2.5-3B-Instruct}"
if [ ! -d "$ACTOR_MODEL_PATH" ]; then
  log "FATAL: ACTOR_MODEL_PATH not found: ${ACTOR_MODEL_PATH}"
  exit 2
fi
export ARPO_ACTOR_MODEL_PATH="$ACTOR_MODEL_PATH"

free_mem_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
log "[pre] free GPU memory: ${free_mem_mb} MiB"

# ── Phase 1: MATH-500 cross-benchmark eval ─────────────────────────────────
log "[phase 1] MATH-500 zero-shot cross-benchmark"
p1_start=$(date +%s)
set +e
bash "${SCRIPT_DIR}/eval_math500_all_ckpts.sh" >> "$ORCH_LOG" 2>&1
p1_ec=$?
set -e
p1_end=$(date +%s); p1_dur=$((p1_end - p1_start))

if [ $p1_ec -eq 0 ]; then
  log "[phase 1] MATH-500 evals OK in $((p1_dur / 60)) min"
  : > "${LOG_DIR}/math500_eval.done"
else
  log "[phase 1] MATH-500 evals exited with code ${p1_ec} after $((p1_dur / 60)) min"
  echo "exit_code=${p1_ec}" > "${LOG_DIR}/math500_eval.failed"
  log "[phase 1] continuing to RLOO phase anyway (independent experiment)"
fi

# GPU reclaim gap
log "[gap] cleaning up ray/GPU, sleeping 60s"
ray stop --force >> "$ORCH_LOG" 2>&1 || true
pkill -f 'verl.trainer.main_ppo' 2>/dev/null || true
sleep 60
free_mem_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
log "[gap] free GPU memory after cleanup: ${free_mem_mb} MiB"

# ── Phase 2: RLOO baseline training ────────────────────────────────────────
RLOO_EXP="${ARPO_RLOO_EXP:-rloo_3b_gsm8k_v1}"
RLOO_DONE="${LOG_DIR}/rloo_${RLOO_EXP}.done"
RLOO_FAIL="${LOG_DIR}/rloo_${RLOO_EXP}.failed"
rm -f "$RLOO_DONE" "$RLOO_FAIL"

log "[phase 2] RLOO start  exp=${RLOO_EXP}"
p2_start=$(date +%s)

# IMPORTANT: the trainer pipes into tee. Use PIPESTATUS to catch trainer exit.
set +e
(
  set -o pipefail
  ARPO_EXPERIMENT_NAME="${RLOO_EXP}" \
    bash "${SCRIPT_DIR}/ARPO_3B_GSM8K_rloo.sh"
)
p2_ec=$?
set -e
p2_end=$(date +%s); p2_dur=$((p2_end - p2_start))

if [ $p2_ec -eq 0 ]; then
  log "[phase 2] RLOO finished OK in $((p2_dur / 60)) min"
  : > "$RLOO_DONE"
else
  log "[phase 2] RLOO exited with code ${p2_ec} after $((p2_dur / 60)) min"
  echo "exit_code=${p2_ec}" > "$RLOO_FAIL"
fi

total=$(( $(date +%s) - p1_start ))
log "========================================================"
log " done in $((total / 60)) min ($((total / 3600))h $(( (total % 3600) / 60 ))m)"
log "  phase 1 (MATH-500 eval): exit=${p1_ec}, $((p1_dur / 60)) min"
log "  phase 2 (RLOO):          exit=${p2_ec}, $((p2_dur / 60)) min"
log "========================================================"

[ $p1_ec -eq 0 ] && [ $p2_ec -eq 0 ] && exit 0
exit 1
