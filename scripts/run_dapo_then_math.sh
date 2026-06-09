#!/usr/bin/env bash
# ============================================================================
# Sequential orchestrator: DAPO ablation (≈5h) → MATH JanusPO training (≈5h)
# Both runs need the full GH200 GPU (~80GB VRAM each), so we MUST run them
# serially. Total wall-clock: ~10–11 hours.
# Logs:
#   logs/orchestrator_<ts>.log               this script's high-level log
#   checkpoints/dapo_3b_gsm8k_v1/run.log     DAPO trainer log
#   checkpoints/janus_3b_math_v1/run.log     MATH trainer log
# Usage:
#   conda activate arpo
#   export ARPO_ACTOR_MODEL_PATH="$HOME/models/Qwen2.5-3B-Instruct"
#   bash scripts/run_dapo_then_math.sh                      # foreground
#   nohup bash scripts/run_dapo_then_math.sh > logs/orch.out 2>&1 &
#   # or use scripts/launch_dapo_then_math_tmux.sh for a tmux session
# Failure handling:
#   • If DAPO fails (non-zero exit), MATH still runs (independent experiments).
#   • Each phase has its own marker file under logs/ so you can tell which
#     stage finished and when.
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"

LOG_DIR="${PARENT_DIR}/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
ORCH_LOG="${LOG_DIR}/orchestrator_${TS}.log"

log() { printf '[orch %s] %s\n' "$(date +%F" "%T)" "$*" | tee -a "$ORCH_LOG"; }

log "========================================================"
log " sequential orchestrator: DAPO  →  MATH"
log " host: $(hostname)   gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
log " parent: ${PARENT_DIR}"
log " log:    ${ORCH_LOG}"
log "========================================================"

# ── Phase 0: sanity ─────────────────────────────────────────────────────────
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

# Pre-prepare MATH train data BEFORE DAPO starts so we fail fast on data issues.
log "[pre] preparing MATH train.parquet (auto-skipped if present)…"
if ! bash "${SCRIPT_DIR}/prep_math_train.sh" >> "$ORCH_LOG" 2>&1; then
  log "WARN: MATH data prep failed; will retry before MATH phase."
fi

# Free VRAM check
free_mem_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
log "[pre] free GPU memory: ${free_mem_mb} MiB"
if [ "${free_mem_mb:-0}" -lt 60000 ]; then
  log "WARN: less than 60 GiB free on GPU; another job may be running."
  log "      we will still proceed; abort with Ctrl-C if undesired."
  sleep 5
fi

# ── Phase 1: DAPO ablation (~5h) ────────────────────────────────────────────
DAPO_EXP="${ARPO_DAPO_EXP:-dapo_3b_gsm8k_v1}"
DAPO_MARKER="${LOG_DIR}/dapo_${DAPO_EXP}.done"
DAPO_FAIL_MARKER="${LOG_DIR}/dapo_${DAPO_EXP}.failed"
rm -f "$DAPO_MARKER" "$DAPO_FAIL_MARKER"

log "[phase 1] DAPO start  exp=${DAPO_EXP}"
phase1_start=$(date +%s)
ARPO_EXPERIMENT_NAME="${DAPO_EXP}" \
  bash "${SCRIPT_DIR}/ARPO_3B_GSM8K_dapo.sh" \
  >> "$ORCH_LOG" 2>&1
phase1_ec=$?
phase1_end=$(date +%s)
phase1_dur=$((phase1_end - phase1_start))

if [ $phase1_ec -eq 0 ]; then
  log "[phase 1] DAPO finished OK in $((phase1_dur / 60)) min"
  : > "$DAPO_MARKER"
else
  log "[phase 1] DAPO exited with code ${phase1_ec} after $((phase1_dur / 60)) min"
  echo "exit_code=${phase1_ec}" > "$DAPO_FAIL_MARKER"
  log "[phase 1] continuing to MATH phase regardless (independent experiment)"
fi

# Best-effort: free residual ray + GPU before next phase
log "[gap] cleaning up ray + sleeping 60 s for GPU reclaim…"
ray stop --force >> "$ORCH_LOG" 2>&1 || true
pkill -f 'verl.trainer.main_ppo' 2>/dev/null || true
sleep 60
free_mem_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
log "[gap] free GPU memory after cleanup: ${free_mem_mb} MiB"

# ── Phase 2: MATH JanusPO (~5h) ─────────────────────────────────────────────
MATH_EXP="${ARPO_MATH_EXP:-janus_3b_math_v1}"
MATH_MARKER="${LOG_DIR}/math_${MATH_EXP}.done"
MATH_FAIL_MARKER="${LOG_DIR}/math_${MATH_EXP}.failed"
rm -f "$MATH_MARKER" "$MATH_FAIL_MARKER"

log "[phase 2] MATH JanusPO start  exp=${MATH_EXP}"
phase2_start=$(date +%s)
ARPO_EXPERIMENT_NAME="${MATH_EXP}" \
  bash "${SCRIPT_DIR}/ARPO_3B_MATH_janus.sh" \
  >> "$ORCH_LOG" 2>&1
phase2_ec=$?
phase2_end=$(date +%s)
phase2_dur=$((phase2_end - phase2_start))

if [ $phase2_ec -eq 0 ]; then
  log "[phase 2] MATH finished OK in $((phase2_dur / 60)) min"
  : > "$MATH_MARKER"
else
  log "[phase 2] MATH exited with code ${phase2_ec} after $((phase2_dur / 60)) min"
  echo "exit_code=${phase2_ec}" > "$MATH_FAIL_MARKER"
fi

total_dur=$(( $(date +%s) - phase1_start ))
log "========================================================"
log " orchestrator done in $((total_dur / 60)) min ($((total_dur / 3600))h $(( (total_dur % 3600) / 60 ))m)"
log "  phase 1 (DAPO): exit=${phase1_ec}, $((phase1_dur / 60)) min"
log "  phase 2 (MATH): exit=${phase2_ec}, $((phase2_dur / 60)) min"
log "========================================================"

# overall exit: 0 only if both phases succeeded
if [ $phase1_ec -eq 0 ] && [ $phase2_ec -eq 0 ]; then
  exit 0
fi
exit 1
