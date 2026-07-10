#!/usr/bin/env bash
# GPU VM bootstrapper.
#
# This script is run on the remote GPU VM.
#
# It should stay fairly project-agnostic:
#   - clone or pull repo
#   - install basic system tools
#   - install uv and Python 3.12
#   - call repo/scripts/setup_envs.sh if present
#
# Required:
#   REPO_URL must be set unless you hard-code it below.
#
# Optional:
#   PROJECT_DIR=/path/to/remote/project
RUN_ENV_SETUP=0

set -euo pipefail

#: "${REPO_URL:?ERROR: REPO_URL must be set. Example: REPO_URL=https://github.com/user/repo.git}"
REPO_URL="${REPO_URL:-https://github.com/NnamdiOdozi/llm_experiments_lab.git}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/llm_experiments_lab}"
REMOTE_ENV_FILE="/tmp/$(basename "$PROJECT_DIR").env"
RUN_ENV_SETUP="${RUN_ENV_SETUP:-1}"

EARLY_LOG_DIR="/tmp/setup_gpu_logs"
mkdir -p "$EARLY_LOG_DIR"
EARLY_LOG_FILE="$EARLY_LOG_DIR/setup_gpu_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$EARLY_LOG_FILE") 2>&1

echo "Logging to: $EARLY_LOG_FILE (will move to project dir after clone)"


echo "=== GPU setup starting ==="
echo "Repo URL:     $REPO_URL"
echo "Project dir:  $PROJECT_DIR"
echo "Run envs:     $RUN_ENV_SETUP"
echo

echo "=== Git config ==="
git config --global user.name "NnamdiOdozi"
git config --global user.email "NnamdiOdozi@users.noreply.github.com"
git config --global pull.rebase true

echo "=== Git shortcuts (ac, acp) ==="
GIT_SHORTCUT_BLOCK='
# Git shortcuts
# git ac  -m "message" = add all, commit (no push)
# git acp -m "message" = add all, commit, push (handles first push automatically)
git() {
    if [[ "$1" == "ac" ]]; then
        shift
        command git add . && command git commit "$@"
    elif [[ "$1" == "acp" ]]; then
        shift
        command git add . && command git commit "$@" && {
            if command git rev-parse --abbrev-ref --symbolic-full-name @{u} &>/dev/null; then
                command git push
            else
                command git push -u origin "$(command git rev-parse --abbrev-ref HEAD)"
            fi
        }
    else
        command git "$@"
    fi
}'

if ! grep -q 'git acp' "$HOME/.bashrc" 2>/dev/null; then
  echo "$GIT_SHORTCUT_BLOCK" >> "$HOME/.bashrc"
  echo "Git shortcuts added to ~/.bashrc"
else
  echo "Git shortcuts already in ~/.bashrc"
fi

echo "=== Install system basics ==="
sudo apt-get update -y
sudo apt-get install -y \
  git \
  curl \
  ca-certificates \
  build-essential \
  python3-dev \
  tmux \
  htop

echo "=== Install direnv ==="
mkdir -p "$HOME/.local/bin"
export bin_path="$HOME/.local/bin"
if ! command -v direnv >/dev/null 2>&1; then
  curl -sfL https://direnv.net/install.sh | bash
fi
if ! grep -q 'direnv hook bash' "$HOME/.bashrc" 2>/dev/null; then
  echo 'eval "$(direnv hook bash)"' >> "$HOME/.bashrc"
  echo "direnv hook added to ~/.bashrc"
fi
echo "direnv: $(direnv version 2>&1 || echo 'not installed')"

echo "=== Install AWS CLI (for Nebius S3) ==="
if ! command -v aws >/dev/null 2>&1; then
  curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  unzip -q /tmp/awscliv2.zip -d /tmp/
  sudo /tmp/aws/install
  rm -rf /tmp/aws /tmp/awscliv2.zip
fi
echo "AWS CLI: $(aws --version 2>&1 || echo 'not installed')"

echo "=== Configure AWS CLI for Nebius S3 ==="
AWS_REGION="${AWS_REGION:-eu-north1}"
AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:-https://storage.eu-north1.nebius.cloud}"
aws configure set region "$AWS_REGION" --profile nebius
aws configure set endpoint_url "$AWS_ENDPOINT_URL" --profile nebius
if [ -n "${AWS_ACCESS_KEY_ID:-}" ]; then
  aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID" --profile nebius
  aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY" --profile nebius
  echo "AWS credentials written to nebius profile"
fi
echo "AWS_PROFILE=nebius" >> "$HOME/.bashrc"
echo "AWS endpoint set to $AWS_ENDPOINT_URL ($AWS_REGION), profile=nebius"

echo "=== Install Node.js 22 and Claude Code ==="

if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi

if ! command -v claude >/dev/null 2>&1; then
  sudo npm install -g @anthropic-ai/claude-code
fi

echo "Node version: $(node --version)"
echo "npm version:  $(npm --version)"
echo "Claude Code:  $(command -v claude)"

echo "=== Install uv if missing ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="$HOME/.local/bin:$PATH"

if [ -f "$HOME/.local/bin/env" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.local/bin/env"
fi

echo "=== uv version ==="
uv --version

echo "=== Install Python 3.12 via uv ==="
uv python install 3.12

echo "=== Clone or pull repo ==="
if [ -d "$PROJECT_DIR/.git" ]; then
  cd "$PROJECT_DIR"
  git pull
elif [ -d "$PROJECT_DIR" ]; then
  echo "Directory exists without .git — cloning into it"
  cd "$PROJECT_DIR"
  git init
  git remote add origin "$REPO_URL"
  git fetch origin
  git checkout -t origin/main
else
  mkdir -p "$(dirname "$PROJECT_DIR")"
  git clone "$REPO_URL" "$PROJECT_DIR"
  cd "$PROJECT_DIR"
fi

echo "=== Moving logs into project dir ==="
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
cp "$EARLY_LOG_FILE" "$LOG_DIR/"
LOG_FILE="$LOG_DIR/$(basename "$EARLY_LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging now at: $LOG_FILE"

echo "=== Current git commit ==="
git rev-parse --short HEAD || true
git status --short || true

echo "=== Copy .env if deploy script provided one ==="
if [ -f "$REMOTE_ENV_FILE" ]; then
  cp "$REMOTE_ENV_FILE" "$PROJECT_DIR/.env"
  echo ".env copied to $PROJECT_DIR/.env"
else
  echo "No $REMOTE_ENV_FILE found; skipping .env copy."
fi

echo "=== Save machine fingerprint ==="
mkdir -p "$PROJECT_DIR/outputs"

{
  echo "Date:"
  date
  echo
  echo "Hostname:"
  hostname
  echo
  echo "User:"
  whoami
  echo
  echo "Project dir:"
  echo "$PROJECT_DIR"
  echo
  echo "Git commit:"
  git rev-parse HEAD || true
  echo
  echo "Disk:"
  df -h
  echo
  echo "GPU:"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    echo "nvidia-smi not found"
  fi
  echo
  echo "uv:"
  uv --version || true
} > "$PROJECT_DIR/outputs/machine_fingerprint.txt"

echo "Machine fingerprint saved to outputs/machine_fingerprint.txt"

if [ "$RUN_ENV_SETUP" = "1" ]; then
  ENV_SETUP_SCRIPT="$PROJECT_DIR/scripts/setup_envs.sh"

  echo "=== Looking for project environment setup script ==="
  echo "Expected path: $ENV_SETUP_SCRIPT"

  if [ -f "$ENV_SETUP_SCRIPT" ]; then
    echo "=== Running project environment setup from repo ==="
    cd "$PROJECT_DIR"
    bash "$ENV_SETUP_SCRIPT"
  else
    echo "WARNING: $ENV_SETUP_SCRIPT not found."
    echo "Skipping environment setup."
    echo
    echo "After committing scripts/setup_envs.sh, rerun on the GPU:"
    echo "  cd $PROJECT_DIR"
    echo "  bash scripts/setup_envs.sh"
  fi
else
  echo "RUN_ENV_SETUP=$RUN_ENV_SETUP, so skipping scripts/setup_envs.sh"
fi

echo
echo "=== GPU setup complete ==="
echo
echo "Project directory:"
echo "  $PROJECT_DIR"
echo
echo "Useful next commands:"
echo "  cd $PROJECT_DIR"
echo "  bash scripts/setup_envs.sh"
echo "  tmux ls"