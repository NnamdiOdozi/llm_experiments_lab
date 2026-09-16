"""Source fingerprint — detects a remote trainer image running stale code.

Real incident, 2026-09-16: gradient clipping was added to
backend/training/train_worker.py, but that file runs INSIDE the trainer
container on Nebius (Dockerfile.trainer-gpu does `COPY backend ./backend`), not
on the controller. Until the image is rebuilt and pushed, the container has no
code to read the `grad_clip` config key — so the key is silently ignored. The
run trains, the loss falls, the UI looks perfect, and the clipping never
happens. No error anywhere.

The fix is a fingerprint both sides can compute from their OWN copy of the
files. No build argument, so nothing can drift out of sync with what was
actually baked in. The controller compares its fingerprint against the
endpoint's at remote-run start and logs loudly on mismatch.

Deliberately covers all of backend/ and config/ (exactly what the Dockerfiles
COPY), not just the training package: the endpoint also serves the API the
controller proxies to, so any difference there matters too. That makes the
check slightly eager — an unrelated comment change trips it — which is the
right trade against missing a real one.
"""

import hashlib
from pathlib import Path

# backend/provenance.py -> repo root (or /app inside the container)
_ROOT = Path(__file__).resolve().parent.parent
_TRACKED_DIRS = ("backend", "config")
_SKIP_PARTS = {"__pycache__", ".pytest_cache"}


def _tracked_files() -> list[Path]:
    files: list[Path] = []
    for name in _TRACKED_DIRS:
        base = _ROOT / name
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if _SKIP_PARTS & set(path.parts):
                continue
            files.append(path)
    return sorted(files)


def source_fingerprint() -> str:
    """Short, stable hash of the Python sources shipped into the trainer image.

    Path-relative and content-based, so the controller's repo checkout and the
    container's /app produce the same value for the same code.
    """
    digest = hashlib.sha256()
    for path in _tracked_files():
        digest.update(path.relative_to(_ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]
