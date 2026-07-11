#!/usr/bin/env bash
# Builds and pushes both trainer images. Prefer the per-image scripts
# (build_push_trainer_cpu.sh / build_push_trainer_gpu.sh) if you only
# changed one of them — this just runs both back to back with the same tag.
#
# Usage: scripts/build_push_all_trainers.sh [tag]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/build_push_trainer_cpu.sh" "$@"
"$SCRIPT_DIR/build_push_trainer_gpu.sh" "$@"
