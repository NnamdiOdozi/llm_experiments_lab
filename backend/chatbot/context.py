"""Builds the grounding context injected into every chatbot request.

Static-first, volatile-last ordering (see docs/superpowers/specs/
2026-07-10-grounded-chatbot-design.md §3): the system prompt and the
current template's source code are stable across a whole chat session;
the loss/audit/log snapshot changes almost every turn and is stapled to
the latest user message rather than the system prompt.
"""

import json
from collections import deque
from functools import lru_cache
from pathlib import Path

from backend.logging_config import get_log_path
from config.settings import settings

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "training" / "templates"

_SYSTEM_PROMPT = """You are the grounded lab assistant for the LLM Experiments Lab, a browser-based tool where learners train small transformer/RNN/MoE models from scratch and connect what they observe to LLM theory they've studied.

You are not a generic chatbot. Every message you receive includes the user's current experiment: its config, architecture source code, live training state, and recent changes. Ground your answers in that real state, not generic textbook answers, whenever the injected context covers the question.

The UI has two ways to change things: the Config panel (hyperparameters, dataset, device) and the layer stack (architecture components). You cannot edit code or configs yourself — if the user wants to change something, point them to the right UI panel, don't describe a code edit.

If a question is about part of the implementation that isn't included in your context (e.g. the training runner, pause/resume mechanics, the database layer), say plainly that you don't have visibility into that part of the code, rather than guessing. For general ML/LLM theory questions not tied to this specific run, answer from your own knowledge."""


@lru_cache(maxsize=8)
def _read_template_source(template: str) -> str:
    """Reads model.py + data.py for one architecture template. Cached —
    these files are static and never change at runtime."""
    template_dir = _TEMPLATES_DIR / template
    parts = []
    for filename in ("model.py", "data.py"):
        path = template_dir / filename
        if path.exists():
            parts.append(f"# {filename}\n{path.read_text(encoding='utf-8')}")
    if not parts:
        return f"(No source found for template '{template}')"
    return "\n\n".join(parts)


def _format_loss_snapshot(run: dict | None) -> str:
    """Recent loss trend for the current run. Reads train_loss_history only —
    each metric row written by train_worker.py includes both train_loss and
    val_loss together (see backend/training/train_worker.py), so a second
    read of val_loss_history would be redundant."""
    if run is None:
        return "No training run has been started for this experiment yet."
    train_history = json.loads(run.get("train_loss_history") or "[]")
    recent = train_history[-20:]
    return "\n".join([
        f"Run status: {run.get('status')}",
        f"Step: {run.get('current_step', 0)} / {run.get('total_steps', 0)}",
        f"Recent metrics (last {len(recent)} points): {json.dumps(recent)}",
    ])


def _get_last_audit_change(experiment_id: int) -> str | None:
    """Most recent [AUDIT] log line for this experiment, or None.

    Matching is a plain substring check for "id=<N> " — this matches both
    "id=%d" and "experiment_id=%d" audit call sites (see
    backend/api/experiments.py) since both end in "id=<N> " followed by
    more fields. Fragile if audit_log.info() call sites change format —
    documented in docs/DESIGN_DECISIONS.md.
    """
    log_path = get_log_path()
    if not log_path.exists():
        return None
    marker = f"id={experiment_id} "
    match = None
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            if "lab.audit" in line and marker in line:
                match = line.rstrip("\n")
    return match


def _get_log_tail(n: int) -> list[str]:
    """Last n lines of the current session's log file, any category."""
    log_path = get_log_path()
    if not log_path.exists():
        return []
    with open(log_path, encoding="utf-8") as f:
        return list(deque(f, maxlen=n))
