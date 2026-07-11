#!/usr/bin/env bash
# Builds and pushes the CPU trainer image (Dockerfile.trainer-cpu) to the
# Nebius container registry.
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
#   scripts/build_push_trainer_cpu.sh [tag]
#   tag defaults to the current short git commit hash. Always also pushes
#   :latest alongside the specific tag.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="llm-lab-trainer-cpu"
TAG="${1:-$(git -C "$REPO_ROOT" rev-parse --short HEAD)}"

# Single source of truth for the registry path is config/settings.py — avoid
# a third hardcoded copy of it (the setting itself already duplicates it
# once from the original endpoint smoke test).
REGISTRY="${NEBIUS_REGISTRY:-$(cd "$REPO_ROOT" && uv run python -c '
from config.settings import settings
print(settings.nebius_cpu_trainer_image.rsplit("/", 1)[0])
')}"

FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"
LATEST_IMAGE="${REGISTRY}/${IMAGE_NAME}:latest"

echo "Building ${FULL_IMAGE}"
docker build -f "${REPO_ROOT}/Dockerfile.trainer-cpu" -t "${FULL_IMAGE}" -t "${LATEST_IMAGE}" "${REPO_ROOT}"

echo "Pushing ${FULL_IMAGE}"
docker push "${FULL_IMAGE}"
echo "Pushing ${LATEST_IMAGE}"
docker push "${LATEST_IMAGE}"

echo "Done: ${FULL_IMAGE}"
