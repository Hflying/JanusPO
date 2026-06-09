#!/usr/bin/env bash
# Quick progress dashboard for the math500+rloo orchestrator.
# Usage: bash scripts/progress_rloo.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "===== orchestrator log tail ====="
LOG=$(ls -1t logs/orch_math_rloo_*.log 2>/dev/null | head -1)
if [ -n "$LOG" ]; then
  echo "  $LOG"
  tail -15 "$LOG"
else
  echo "  (no orchestrator log yet)"
fi

echo
echo "===== RLOO training progress ====="
RUN=checkpoints/rloo_3b_gsm8k_v1/run.log
if [ -f "$RUN" ]; then
  tail -1 "$RUN" | grep -oE "Training Progress:[^[]*" || echo "  (no progress bar yet)"
  echo
  echo "  val trajectory so far:"
  grep -oE "step:[0-9]+ -.*val-core/openai/gsm8k/reward/mean@1:[0-9.]+" "$RUN" \
    | awk -F 'step:|val-core/openai/gsm8k/reward/mean@1:' \
        '{printf "    step=%4d  val=%.4f\n", $2, $3}' | tail -12
fi

echo
echo "===== MATH-500 eval status ====="
for label in qwen3b_zs grpo_step30 dapo_step50 dapo_step233 janus_step200; do
  d=checkpoints/_eval/${label}_math500/run.log
  if [ -f "$d" ]; then
    score=$(grep -oE "val-core/.*reward/mean@1:[0-9.]+" "$d" | tail -1 | awk -F: '{print $2}')
    [ -n "$score" ] && echo "   DONE  ${label}  MATH500=${score}" \
                    || echo "   RUN   ${label}  (no score yet; log size $(stat -c %s "$d" 2>/dev/null))"
  else
    echo "   WAIT  ${label}"
  fi
done

echo
echo "===== GPU =====";  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>&1
echo "===== tmux session ====="; tmux has-session -t math500_rloo 2>/dev/null && echo "  alive" || echo "  (dead)"
