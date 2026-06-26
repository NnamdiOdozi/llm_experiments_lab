"""Centralized logging setup — one timestamped file per server session.

Log categories (prefixed in messages):
  [REQUEST]   — HTTP request/response
  [ERROR]     — Application errors
  [TRAINING]  — Training lifecycle events (start/pause/resume/stop/complete/fail)
  [PROMPT]    — Pause-and-prompt inference calls
  [AUDIT]     — Config changes, experiment creation
  [SESSION]   — Server startup/shutdown
"""

import logging
import time
from pathlib import Path

from config.settings import settings

# Module-level loggers for each category
request_log = logging.getLogger("lab.request")
error_log = logging.getLogger("lab.error")
training_log = logging.getLogger("lab.training")
prompt_log = logging.getLogger("lab.prompt")
audit_log = logging.getLogger("lab.audit")
session_log = logging.getLogger("lab.session")

_SESSION_START = time.strftime("%Y-%m-%d_%H-%M-%S")


def get_log_path() -> Path:
    log_dir = settings.data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"session_{_SESSION_START}.log"


def setup_logging():
    """Call once at app startup. Creates session log file, wires all loggers."""
    log_path = get_log_path()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — all levels
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — INFO+
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # Wire up root lab logger
    root = logging.getLogger("lab")
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(ch)

    # Quiet down noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    session_log.info("Server session started — log file: %s", log_path)
    return log_path
