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
