#!/usr/bin/env bash
# Deploy to GPU from laptop.
#
# Usage:
#   bash scripts/deploy_gpu.sh [host-alias]
#
# Optional environment variables:
#   REPO_URL=...              Override GitHub repo URL
#   PROJECT_DIR=...           Remote project directory, e.g. /home/nodozi/toxic_homework
#   RUN_ENV_SETUP=1           Run scripts/setup_envs.sh after GPU bootstrap
#
# Example:
#   REPO_URL=https://github.com/NnamdiOdozi/toxic_homework.git \
#   PROJECT_DIR=/home/nodozi/toxic_homework \
#   bash scripts/deploy_gpu.sh nebius-vm

set -euo pipefail

HOST="${1:-nebius-vm}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_ENV_FILE="/tmp/$(basename "$LOCAL_DIR").env"

LOG_DIR="$LOCAL_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/deploy_gpu_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging to: $LOG_FILE"

TMUX_SESSION="gpu-setup"
AUTOSSH_PIDFILE="/tmp/autossh_${HOST}.pid"

REPO_URL="${REPO_URL:-$(git -C "$LOCAL_DIR" config --get remote.origin.url || true)}"
# Always whatever's actually checked out locally (picks up .envrc's export
# if direnv already set it, otherwise derives it fresh) — never hand-typed,
# so it can't drift from the branch you're actually working on.
GIT_BRANCH="${GIT_BRANCH:-$(git -C "$LOCAL_DIR" rev-parse --abbrev-ref HEAD)}"
PROJECT_DIR="${PROJECT_DIR:-}"
RUN_ENV_SETUP="${RUN_ENV_SETUP:-1}"

if [ -z "$REPO_URL" ]; then
  echo "ERROR: REPO_URL is not set and could not be inferred from local git remote."
  echo "Run with:"
  echo "  REPO_URL=https://github.com/<user>/<repo>.git bash scripts/deploy_gpu.sh $HOST"
  exit 1
fi

echo "=== Local project root ==="
echo "$LOCAL_DIR"
echo

echo "=== Remote host ==="
echo "$HOST"
echo

echo "=== Repo URL ==="
echo "$REPO_URL"
echo

echo "=== Git branch ==="
echo "$GIT_BRANCH"
echo

echo "=== Copying setup_gpu.sh to $HOST:/tmp ==="
scp "$LOCAL_DIR/scripts/setup_gpu.sh" "$HOST:/tmp/setup_gpu.sh"

if [ -f "$LOCAL_DIR/.env" ]; then
  echo "=== Copying .env to $HOST:$REMOTE_ENV_FILE ==="
  scp "$LOCAL_DIR/.env" "$HOST:$REMOTE_ENV_FILE"
else
  echo "=== No local .env found; skipping .env copy ==="
fi

echo "=== Ensuring tmux is installed on $HOST ==="
ssh "$HOST" "command -v tmux >/dev/null || { sudo apt-get update -y && sudo apt-get install -y tmux; }"

echo "=== Preparing remote runner ==="

REMOTE_RUNNER="/tmp/run_setup_gpu.sh"

ssh "$HOST" "cat > $REMOTE_RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail

export REPO_URL="$REPO_URL"
export GIT_BRANCH="$GIT_BRANCH"
export RUN_ENV_SETUP="$RUN_ENV_SETUP"

if [ -n "$PROJECT_DIR" ]; then
  export PROJECT_DIR="$PROJECT_DIR"
fi

bash /tmp/setup_gpu.sh
EOF

ssh "$HOST" "chmod +x $REMOTE_RUNNER"

echo "=== Running setup_gpu.sh inside tmux session '$TMUX_SESSION' ==="
ssh "$HOST" "tmux kill-session -t $TMUX_SESSION 2>/dev/null || true"
ssh "$HOST" "tmux new-session -d -s $TMUX_SESSION 'bash $REMOTE_RUNNER; exec bash --login'"

echo "=== Starting lightweight port forwarding ==="

if [ -f "$AUTOSSH_PIDFILE" ]; then
  kill "$(cat "$AUTOSSH_PIDFILE")" 2>/dev/null || true
  rm -f "$AUTOSSH_PIDFILE"
  sleep 1
fi

if command -v autossh >/dev/null 2>&1; then
  AUTOSSH_PIDFILE="$AUTOSSH_PIDFILE" autossh -f -M 0 -N \
    -o "ServerAliveInterval=30" \
    -o "ServerAliveCountMax=3" \
    -o "ExitOnForwardFailure=yes" \
    -L 8000:localhost:8000 \
    -L 8888:localhost:8888 \
    "$HOST"

  pgrep -f "autossh.*-N.*$HOST" | tail -1 > "$AUTOSSH_PIDFILE" || true

  echo "Port forwarding active."
  echo "  vLLM:     http://localhost:8000"
  echo "  Jupyter:  http://localhost:8888"
  echo "  Stop tunnel later with: kill \$(cat $AUTOSSH_PIDFILE)"
else
  echo "autossh not found locally; skipping persistent port forwarding."
  echo "Manual fallback:"
  echo "  ssh -L 8000:localhost:8000 -L 8888:localhost:8888 $HOST"
fi

echo
echo "=== Attaching to tmux session on GPU ==="
echo "Detach with: Ctrl+B then D"
echo "Reattach later with:"
echo "  ssh $HOST -t 'tmux attach -t $TMUX_SESSION'"
echo

ssh "$HOST" -t "tmux attach -t $TMUX_SESSION"