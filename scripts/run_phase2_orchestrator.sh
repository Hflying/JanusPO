#!/usr/bin/env bash
# ============================================================================
# Serial orchestrator for two paper experiments:
#   (a) JanusPO-minus-entropy ablation (is the entropy guard load-bearing?)
#   (b) DAPO β_KL sweep (at what β does DAPO stop collapsing?)
# Each run: 3B Qwen2.5-3B-Instruct + GSM8K + 233 steps + save_freq=50.
# Logs go to checkpoints/<exp_name>/run.log; we only need val curves.
# ============================================================================
set -eu
cd 
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate arpo
export ARPO_ACTOR_MODEL_PATH="$HOME/models/Qwen2.5-3B-Instruct"

ORCH_LOG="logs/orch_phase2_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
log() { echo "[orch $(date '+%F %T')] $*" | tee -a "$ORCH_LOG"; }

run_and_wait() {
  local exp_name=$1; shift
  local script=$1; shift
  local ckpt_dir="checkpoints/${exp_name}"
  if [ -f "${ckpt_dir}/run.log" ] && grep -q "val-core/openai/gsm8k/reward/mean@1:[0-9]" "${ckpt_dir}/run.log" 2>/dev/null; then
    local nsteps=$(grep -oE "step:[0-9]+ -" "${ckpt_dir}/run.log" | tail -1 | grep -oE "[0-9]+")
    if [ -n "$nsteps" ] && [ "$nsteps" -ge 230 ]; then
      log "[skip] $exp_name already has ${nsteps} steps, skipping"
      return 0
    fi
  fi
  log "============================================================"
  log "STARTING: $exp_name"
  log "  script=$script"
  log "  env overrides: $*"
  log "============================================================"
  # launch with the given env overrides
  ( "$@" bash "$script" ) >> "$ORCH_LOG" 2>&1
  local ec=$?
  log "FINISHED: $exp_name  exit=$ec"
  # cleanup ray/vllm between runs
  ray stop --force >/dev/null 2>&1 || true
  pkill -f 'verl.trainer.main_ppo|vllm' 2>/dev/null || true
  sleep 20
  return $ec
}

log "GPU state before start:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee -a "$ORCH_LOG"

# ── Experiment 3: JanusPO with entropy_coeff = 0 (no entropy guard) ─────────
# Everything else identical to the main JanusPO run.
run_and_wait "janus_no_entropy_v1" "scripts/ARPO_3B_GSM8K_plan_c_asymkl_v2.sh" \
  env \
  ARPO_EXPERIMENT_NAME=janus_no_entropy_v1 \
  ARPO_ENTROPY_COEFF=0.0 \
  ARPO_ENTROPY_WEIGHT=0.0 \
  ARPO_ENTROPY_UPPER_BOUND=999 \
  ARPO_SAVE_FREQ=50

# ── Experiment 2: DAPO β_KL sweep (3 points: 1e-4, 1e-3, 1e-2) ──────────────
# DAPO baseline (β=0) is already in dapo_3b_gsm8k_v1 so we skip it.
# Reusing the DAPO script but overriding KL_LOSS_COEF and flipping use_kl_loss.
for beta in 1e-4 1e-3 1e-2; do
  # use underscore-clean name for filesystem
  tag=$(echo "$beta" | sed 's/-/m/g')
  run_and_wait "dapo_betakl_${tag}_v1" "scripts/ARPO_3B_GSM8K_dapo.sh" \
    env \
    ARPO_EXPERIMENT_NAME=dapo_betakl_${tag}_v1 \
    ARPO_KL_LOSS_COEF=${beta} \
    ARPO_USE_KL_LOSS=True \
    ARPO_SAVE_FREQ=50
done

log "============================================================"
log "ALL 4 RUNS FINISHED"
log "============================================================"
touch logs/phase2_all.done
