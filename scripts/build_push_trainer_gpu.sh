#!/usr/bin/env bash
# Builds and pushes the GPU trainer image (Dockerfile.trainer-gpu) to the
# Nebius container registry.
#
# No GPU hardware needed to run this — docker build only needs to download
# and layer the nvidia/cuda base image and install the CUDA torch wheel; the
# actual GPU is only touched at container runtime on the Nebius endpoint.
# Safe to run on a CPU-only build machine.
#
# Run from anywhere — resolves the repo root from this script's own path.
# Requires: docker CLI installed, and `nebius registry configure-helper` run
# once on this machine so `docker push` authenticates automatically — via a
# VM-attached service account if there is one, otherwise whatever `nebius`
# profile is active. No manual `docker login` needed either way. Also
# requires the `nebius` CLI itself for the registry path lookup below
# (falls back to NEBIUS_REGISTRY if unavailable).
#
# Usage:
#   scripts/build_push_trainer_gpu.sh [tag]
#   tag defaults to the current short git commit hash. Always also pushes
#   :latest alongside the specific tag.

set -euo pipefail

# uv/direnv install to ~/.local/bin (see scripts/setup_gpu.sh); a fresh SSH
# session that hasn't sourced ~/.bashrc since won't have it on PATH yet.
export PATH="$HOME/.local/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="llm-lab-trainer-gpu"
TAG="${1:-$(git -C "$REPO_ROOT" rev-parse --short HEAD)}"

REGISTRY="${NEBIUS_REGISTRY:-$(cd "$REPO_ROOT" && uv run python -c '
from config.settings import settings
print(settings.nebius_gpu_trainer_image.rsplit("/", 1)[0])
')}"

FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"
LATEST_IMAGE="${REGISTRY}/${IMAGE_NAME}:latest"

echo "Building ${FULL_IMAGE}"
docker build -f "${REPO_ROOT}/Dockerfile.trainer-gpu" -t "${FULL_IMAGE}" -t "${LATEST_IMAGE}" "${REPO_ROOT}"

echo "Pushing ${FULL_IMAGE}"
docker push "${FULL_IMAGE}"
echo "Pushing ${LATEST_IMAGE}"
docker push "${LATEST_IMAGE}"

echo "Done: ${FULL_IMAGE}"
