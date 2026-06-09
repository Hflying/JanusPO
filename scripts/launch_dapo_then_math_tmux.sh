#!/usr/bin/env bash
# Launch the sequential DAPO → MATH orchestrator inside a detached tmux session
# so it survives shell disconnects.
# Usage:
#   conda activate arpo
#   bash scripts/launch_dapo_then_math_tmux.sh
# Watch:   tmux attach -t dapo_then_math
# Detach:  Ctrl-b d
# Kill:    tmux kill-session -t dapo_then_math
set -euo pipefail

SESSION="${ARPO_TMUX_SESSION:-dapo_then_math}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

if ! command -v tmux >/dev/null; then
  echo "tmux not installed; falling back to nohup."
  cd "$PARENT_DIR"
  mkdir -p logs
  nohup bash "${SCRIPT_DIR}/run_dapo_then_math.sh" > "logs/orch.nohup.out" 2>&1 &
  echo "launched in nohup, pid=$!"
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
  exit 1
fi

# Forward env vars the trainer expects.
ENV_PRELUDE=""
for v in ARPO_ACTOR_MODEL_PATH ARPO_DAPO_EXP ARPO_MATH_EXP WANDB_API_KEY ARPO_SEED; do
  if [ -n "${!v:-}" ]; then
    ENV_PRELUDE+="export ${v}=$(printf %q "${!v}"); "
  fi
done

CMD="cd $(printf %q "$PARENT_DIR") && \
source $HOME/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true; \
source $HOME/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true; \
conda activate arpo; \
${ENV_PRELUDE} \
bash $(printf %q "${SCRIPT_DIR}/run_dapo_then_math.sh"); \
ec=\$?; \
echo \"orchestrator exited with \$ec\"; \
exec bash"

tmux new-session -d -s "$SESSION" "$CMD"
echo "tmux session '$SESSION' launched."
echo "  attach:   tmux attach -t $SESSION"
echo "  log file: $(ls -1t ${PARENT_DIR}/logs/orchestrator_*.log 2>/dev/null | head -1 || echo '(will appear shortly)')"
