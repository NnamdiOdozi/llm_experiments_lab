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


def _scan_log_lines(category_marker: str, id_marker: str) -> list[str]:
    """Lines from the current session log matching both substrings, in file
    order. Substring matching is coupled to the exact log message formats —
    documented in docs/DESIGN_DECISIONS.md."""
    log_path = get_log_path()
    if not log_path.exists():
        return []
    with open(log_path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if category_marker in line and id_marker in line]


def _get_last_audit_change(experiment_id: int) -> str | None:
    """Most recent [AUDIT] log line for this experiment, or None.

    The "id=<N> " marker matches both "id=%d" and "experiment_id=%d" audit
    call sites (see backend/api/experiments.py) since both end in "id=<N> "
    followed by more fields.
    """
    matches = _scan_log_lines("lab.audit", f"id={experiment_id} ")
    return matches[-1] if matches else None


def _get_prompt_history(run_id: int) -> list[dict]:
    """Pause-and-prompt exchanges for this run, oldest first, capped to the
    most recent 10 (mirrors the [-20:] loss-history cap above). Parses the
    JSON payload written by backend/api/training.py::prompt_model."""
    pairs = []
    for line in _scan_log_lines("lab.prompt", f"run_id={run_id} "):
        if "payload=" not in line:
            continue
        try:
            pairs.append(json.loads(line.split("payload=", 1)[1]))
        except json.JSONDecodeError:
            continue
    return pairs[-10:]


def _get_log_tail(n: int) -> list[str]:
    """Last n lines of the current session's log file, any category."""
    log_path = get_log_path()
    if not log_path.exists():
        return []
    with open(log_path, encoding="utf-8") as f:
        return list(deque(f, maxlen=n))


def _build_session_context(experiment: dict, config: dict, template: str) -> str:
    source = _read_template_source(template)
    return (
        f"Experiment: {experiment['name']}\n"
        f"Architecture template: {template}\n"
        f"Description: {config.get('description', '')}\n"
        f"Current config:\n{json.dumps(config, indent=2)}\n\n"
        f"Source code for this architecture ({template}):\n{source}"
    )


def _build_volatile_snapshot(experiment_id: int, run: dict | None) -> str:
    parts = [_format_loss_snapshot(run)]
    last_change = _get_last_audit_change(experiment_id)
    if last_change:
        parts.append(f"Last change made: {last_change}")
    if run is not None and run.get("id") is not None:
        prompts = _get_prompt_history(run["id"])
        if prompts:
            lines = [
                f"At step {p.get('step')}, user prompted: {json.dumps(p.get('prompt'))} "
                f"→ model output: {json.dumps(p.get('output'))}"
                for p in prompts
            ]
            parts.append(
                "Pause-and-prompt history (the user prompts the paused half-trained model "
                "to see how output quality evolves as training proceeds):\n" + "\n".join(lines)
            )
    log_tail = _get_log_tail(settings.chatbot_log_tail_lines)
    if log_tail:
        parts.append("Recent log lines:\n" + "".join(log_tail))
    return "\n\n".join(parts)


def assemble_messages(
    experiment: dict, run: dict | None, history: list[dict], user_message: str
) -> list[dict]:
    """Builds the full message list for one chatbot turn.

    Ordering is deliberate — static content first, volatile snapshot last,
    stapled to the current user message. See module docstring and
    docs/superpowers/specs/2026-07-10-grounded-chatbot-design.md §3.

    `history` must NOT include the message currently being sent — callers
    fetch history before writing the new user message to avoid duplicating
    it here.
    """
    config = json.loads(experiment["config_json"])
    template = config.get("template", "transformer")

    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.append({"role": "system", "content": _build_session_context(experiment, config, template)})

    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    volatile = _build_volatile_snapshot(experiment["id"], run)
    messages.append({"role": "user", "content": f"{volatile}\n\nUser question: {user_message}"})
    return messages
