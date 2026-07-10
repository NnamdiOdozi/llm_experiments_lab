# Grounded Chatbot — Backend Implementation Plan (Plan A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the backend half of the grounded chatbot — Token Factory client, context assembly, chat persistence, and the SSE chat endpoint — independently testable with no frontend involved.

**Architecture:** Eager context injection (no RAG/tool-calling), static-then-volatile message ordering, sliding-window history, SSE streaming relay. Full rationale in `docs/superpowers/specs/2026-07-10-grounded-chatbot-design.md`.

**Tech Stack:** FastAPI, aiosqlite, `openai` SDK (`AsyncOpenAI`) pointed at Nebius Token Factory, pytest + pytest-asyncio (new — no test suite currently exists in this repo).

**Working directory for all steps:** `.worktrees/grounded-chatbot/` (branch `feature/grounded-chatbot`).

---

### Task 1: Test harness setup

No test suite exists in this repo yet. This task adds `pytest`/`pytest-asyncio` as dev dependencies and proves the harness works before anything else is built on top of it.

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Add test and runtime dependencies**

Edit `pyproject.toml` — add `"openai>=1.50.0"` to the `dependencies` list, and add a new `[dependency-groups]` table:

```toml
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "aiosqlite>=0.20.0",
    "pydantic-settings>=2.5.0",
    "torch>=2.2.0",
    "numpy>=1.24.0",
    "nbformat>=5.10.0",
    "websockets>=13.0",
    "httpx>=0.27.0",
    "openai>=1.50.0",
]

[dependency-groups]
dev = ["pytest>=8.0.0", "pytest-asyncio>=0.24.0"]
```

- [ ] **Step 2: Install and configure**

Run: `uv sync --group dev`
Expected: `openai`, `pytest`, `pytest-asyncio` appear in the install output.

Add to `pyproject.toml` (new section, anywhere after `[project]`):

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 3: Create test package and a smoke test**

Create `tests/__init__.py` (empty file).

Create `tests/test_smoke.py`:

```python
def test_smoke():
    assert 1 + 1 == 2
```

- [ ] **Step 4: Run it**

Run: `uv run pytest -q`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock tests/__init__.py tests/test_smoke.py
git commit -m "test: add pytest harness and openai dependency"
```

---

### Task 2: Settings additions

**Files:**
- Modify: `config/settings.py`

- [ ] **Step 1: Write a failing test for the new fields**

Create `tests/test_settings.py`:

```python
from config.settings import Settings


def test_chatbot_settings_have_expected_defaults():
    s = Settings(nebius_key=None)
    assert s.token_factory_base_url == "https://api.tokenfactory.nebius.com/v1/"
    assert s.token_factory_model == "Qwen/Qwen3-235B-A22B-Thinking-2507"
    assert s.chatbot_log_tail_lines == 50
    assert s.chatbot_history_window_turns == 10
    assert s.nebius_key is None
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/test_settings.py -v`
Expected: FAIL — `AttributeError` or `TypeError: unexpected keyword argument 'nebius_key'`

- [ ] **Step 3: Add the fields**

In `config/settings.py`, add inside the `Settings` class (after the existing `github_url` field, before `model_config`):

```python
    # Grounded chatbot / Nebius Token Factory
    nebius_key: str | None = None
    token_factory_base_url: str = "https://api.tokenfactory.nebius.com/v1/"
    token_factory_model: str = "Qwen/Qwen3-235B-A22B-Thinking-2507"
    chatbot_log_tail_lines: int = 50
    chatbot_history_window_turns: int = 10
```

- [ ] **Step 4: Run it, verify it passes**

Run: `uv run pytest tests/test_settings.py -v`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add config/settings.py tests/test_settings.py
git commit -m "feat: add chatbot/Token Factory settings"
```

---

### Task 3: Logging addition

**Files:**
- Modify: `backend/logging_config.py`

- [ ] **Step 1: Write a failing test**

Create `tests/test_logging_config.py`:

```python
import logging

from backend.logging_config import chatbot_log


def test_chatbot_log_is_a_lab_logger():
    assert chatbot_log.name == "lab.chatbot"
    assert isinstance(chatbot_log, logging.Logger)
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/test_logging_config.py -v`
Expected: FAIL — `ImportError: cannot import name 'chatbot_log'`

- [ ] **Step 3: Add the logger**

In `backend/logging_config.py`, update the module docstring's category list and add the logger. Current top of file:

```python
"""Centralized logging setup — one timestamped file per server session.

Log categories (prefixed in messages):
  [REQUEST]   — HTTP request/response
  [ERROR]     — Application errors
  [TRAINING]  — Training lifecycle events (start/pause/resume/stop/complete/fail)
  [PROMPT]    — Pause-and-prompt inference calls
  [AUDIT]     — Config changes, experiment creation
  [SESSION]   — Server startup/shutdown
"""
```

Change to:

```python
"""Centralized logging setup — one timestamped file per server session.

Log categories (prefixed in messages):
  [REQUEST]   — HTTP request/response
  [ERROR]     — Application errors
  [TRAINING]  — Training lifecycle events (start/pause/resume/stop/complete/fail)
  [PROMPT]    — Pause-and-prompt inference calls
  [AUDIT]     — Config changes, experiment creation
  [SESSION]   — Server startup/shutdown
  [CHATBOT]   — Grounded chatbot requests to Token Factory
"""
```

Then, after the existing `session_log = logging.getLogger("lab.session")` line, add:

```python
chatbot_log = logging.getLogger("lab.chatbot")
```

- [ ] **Step 4: Run it, verify it passes**

Run: `uv run pytest tests/test_logging_config.py -v`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/logging_config.py tests/test_logging_config.py
git commit -m "feat: add chatbot log category"
```

---

### Task 4: DB layer — `chat_messages` table and access functions

**Files:**
- Modify: `backend/db.py`
- Test: `tests/test_db_chat_messages.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_db_chat_messages.py`:

```python
import pytest

from backend import db


@pytest.fixture
async def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_lab.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    await db.init_db()
    exp_id = await db.create_experiment("Test experiment", {"template": "transformer"})
    return exp_id


async def test_add_and_get_chat_messages(temp_db):
    exp_id = temp_db
    await db.add_chat_message(exp_id, "user", "What does this mean?")
    await db.add_chat_message(
        exp_id, "assistant", "It means the loss is decreasing.",
        prompt_tokens=100, completion_tokens=20, total_tokens=120, latency_ms=850,
    )

    messages = await db.get_chat_messages(exp_id)

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What does this mean?"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["prompt_tokens"] == 100
    assert messages[1]["total_tokens"] == 120


async def test_get_chat_messages_respects_limit_and_order(temp_db):
    exp_id = temp_db
    for i in range(5):
        await db.add_chat_message(exp_id, "user", f"message {i}")

    messages = await db.get_chat_messages(exp_id, limit=2)

    assert len(messages) == 2
    # Must be the two most recent, in chronological order
    assert messages[0]["content"] == "message 3"
    assert messages[1]["content"] == "message 4"


async def test_get_chat_messages_empty_for_unknown_experiment(temp_db):
    messages = await db.get_chat_messages(999999)
    assert messages == []
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/test_db_chat_messages.py -v`
Expected: FAIL — `AttributeError: module 'backend.db' has no attribute 'add_chat_message'`

- [ ] **Step 3: Add the schema**

In `backend/db.py`, add to the `SCHEMA` string, after the `training_runs` table definition and before the closing `"""`:

```python

CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    latency_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

- [ ] **Step 4: Add the access functions**

In `backend/db.py`, after `list_runs_for_experiment` (before the `# ── Sync versions ──` comment), add:

```python
async def add_chat_message(
    experiment_id: int,
    role: str,
    content: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    latency_ms: int | None = None,
) -> int:
    db_conn = await get_db()
    cursor = await db_conn.execute(
        "INSERT INTO chat_messages "
        "(experiment_id, role, content, prompt_tokens, completion_tokens, total_tokens, latency_ms) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (experiment_id, role, content, prompt_tokens, completion_tokens, total_tokens, latency_ms),
    )
    await db_conn.commit()
    row_id = cursor.lastrowid
    await db_conn.close()
    return row_id


async def get_chat_messages(experiment_id: int, limit: int | None = None) -> list[dict]:
    """Chat history for an experiment, oldest first.

    limit=None returns full history (for the UI). limit=N returns only the
    most recent N messages, still in chronological order (for the sliding
    window sent to the LLM) — same function serves both callers.
    """
    db_conn = await get_db()
    if limit is None:
        cursor = await db_conn.execute(
            "SELECT * FROM chat_messages WHERE experiment_id = ? ORDER BY id ASC",
            (experiment_id,),
        )
        rows = await cursor.fetchall()
    else:
        cursor = await db_conn.execute(
            "SELECT * FROM chat_messages WHERE experiment_id = ? ORDER BY id DESC LIMIT ?",
            (experiment_id, limit),
        )
        rows = list(reversed(await cursor.fetchall()))
    await db_conn.close()
    return [dict(r) for r in rows]
```

- [ ] **Step 5: Run it, verify it passes**

Run: `uv run pytest tests/test_db_chat_messages.py -v`
Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add backend/db.py tests/test_db_chat_messages.py
git commit -m "feat: add chat_messages table and access functions"
```

---

### Task 5: Token Factory client

**Files:**
- Create: `backend/chatbot/__init__.py`
- Create: `backend/chatbot/client.py`
- Test: `tests/test_chatbot_client.py`

- [ ] **Step 1: Create the package**

Create `backend/chatbot/__init__.py` (empty file).

- [ ] **Step 2: Write failing tests**

Create `tests/test_chatbot_client.py`:

```python
import pytest

from backend.chatbot import client as tf_client
from config.settings import settings


class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.delta = _FakeDelta(content)


class _FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens, total_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _FakeChunk:
    def __init__(self, content=None, usage=None):
        self.choices = [_FakeChoice(content)] if content is not None else []
        self.usage = usage


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self._chunks:
            yield c


class _FakeCompletions:
    def __init__(self, chunks):
        self._chunks = chunks

    async def create(self, **kwargs):
        return _FakeStream(self._chunks)


class _FakeChat:
    def __init__(self, chunks):
        self.completions = _FakeCompletions(chunks)


class _FakeClient:
    def __init__(self, chunks):
        self.chat = _FakeChat(chunks)


def test_is_configured_false_without_key(monkeypatch):
    monkeypatch.setattr(settings, "nebius_key", None)
    assert tf_client.is_configured() is False


def test_is_configured_true_with_key(monkeypatch):
    monkeypatch.setattr(settings, "nebius_key", "fake-key-value")
    assert tf_client.is_configured() is True


async def test_stream_completion_yields_deltas_then_usage(monkeypatch):
    chunks = [
        _FakeChunk(content="Hello"),
        _FakeChunk(content=" world"),
        _FakeChunk(content=None, usage=_FakeUsage(10, 5, 15)),
    ]
    monkeypatch.setattr(tf_client, "_get_client", lambda: _FakeClient(chunks))

    results = [
        (delta, usage)
        async for delta, usage in tf_client.stream_completion([{"role": "user", "content": "hi"}])
    ]

    deltas = [d for d, _ in results if d]
    assert "".join(deltas) == "Hello world"
    usages = [u for _, u in results if u is not None]
    assert usages == [{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}]
```

- [ ] **Step 3: Run it, verify it fails**

Run: `uv run pytest tests/test_chatbot_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.chatbot.client'`

- [ ] **Step 4: Implement the client**

Create `backend/chatbot/client.py`:

```python
"""Thin wrapper around the Nebius Token Factory OpenAI-compatible API."""

import time
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from backend.logging_config import chatbot_log, error_log
from config.settings import settings

_client: AsyncOpenAI | None = None


def is_configured() -> bool:
    return bool(settings.nebius_key)


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=settings.token_factory_base_url,
            api_key=settings.nebius_key,
        )
    return _client


async def stream_completion(messages: list[dict]) -> AsyncIterator[tuple[str, dict | None]]:
    """Stream a chat completion from Token Factory.

    Yields (text_delta, usage) pairs. usage is None on every chunk except
    the final one, per the OpenAI streaming spec with include_usage set.
    """
    client = _get_client()
    start = time.perf_counter()
    chatbot_log.info(
        "Token Factory request: model=%s messages=%d", settings.token_factory_model, len(messages)
    )
    try:
        stream = await client.chat.completions.create(
            model=settings.token_factory_model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            usage = None
            if chunk.usage is not None:
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }
            delta = ""
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
            if delta or usage is not None:
                yield delta, usage
        elapsed_ms = (time.perf_counter() - start) * 1000
        chatbot_log.info("Token Factory request complete: %.0fms", elapsed_ms)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        error_log.error("Token Factory request failed after %.0fms: %s", elapsed_ms, exc)
        raise
```

- [ ] **Step 5: Run it, verify it passes**

Run: `uv run pytest tests/test_chatbot_client.py -v`
Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add backend/chatbot/__init__.py backend/chatbot/client.py tests/test_chatbot_client.py
git commit -m "feat: add Token Factory streaming client"
```

---

### Task 6: Context assembly

**Files:**
- Create: `backend/chatbot/context.py`
- Test: `tests/test_chatbot_context.py`

This is the core grounding logic: static system prompt, per-template source injection, volatile snapshot (loss/audit/logs), and final message assembly. Built up function by function.

- [ ] **Step 1: Write a failing test for template source reading**

Create `tests/test_chatbot_context.py`:

```python
from backend.chatbot import context


def test_read_template_source_includes_real_transformer_code():
    source = context._read_template_source("transformer")
    assert "class RotaryPositionalEncoding" in source
    assert "model.py" in source
    assert "data.py" in source


def test_read_template_source_unknown_template_does_not_crash():
    source = context._read_template_source("nonexistent_template")
    assert "No source found" in source
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/test_chatbot_context.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.chatbot.context'`

- [ ] **Step 3: Implement template source reading**

Create `backend/chatbot/context.py`:

```python
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
```

- [ ] **Step 4: Run it, verify it passes**

Run: `uv run pytest tests/test_chatbot_context.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/chatbot/context.py tests/test_chatbot_context.py
git commit -m "feat: add chatbot context module with template source reading"
```

- [ ] **Step 6: Write a failing test for the loss snapshot formatter**

Add to `tests/test_chatbot_context.py`:

```python
def test_format_loss_snapshot_with_no_run():
    snapshot = context._format_loss_snapshot(None)
    assert "No training run" in snapshot


def test_format_loss_snapshot_with_run():
    run = {
        "status": "running",
        "current_step": 40,
        "total_steps": 1000,
        "train_loss_history": json.dumps([
            {"step": 20, "train_loss": 1.8, "val_loss": 1.9},
            {"step": 40, "train_loss": 1.5, "val_loss": 1.6},
        ]),
    }
    snapshot = context._format_loss_snapshot(run)
    assert "running" in snapshot
    assert "40 / 1000" in snapshot
    assert "1.5" in snapshot
```

Add `import json` to the top of `tests/test_chatbot_context.py`.

- [ ] **Step 7: Run it, verify it fails**

Run: `uv run pytest tests/test_chatbot_context.py -v`
Expected: FAIL — `AttributeError: module 'backend.chatbot.context' has no attribute '_format_loss_snapshot'`

- [ ] **Step 8: Implement the loss snapshot formatter**

Append to `backend/chatbot/context.py`:

```python


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
```

- [ ] **Step 9: Run it, verify it passes**

Run: `uv run pytest tests/test_chatbot_context.py -v`
Expected: `4 passed`

- [ ] **Step 10: Commit**

```bash
git add backend/chatbot/context.py tests/test_chatbot_context.py
git commit -m "feat: add loss snapshot formatter to chatbot context"
```

- [ ] **Step 11: Write failing tests for audit-change and log-tail readers**

These read the current session's log file (`logging_config.get_log_path()`), so tests monkeypatch that function to point at a temp file.

Add to `tests/test_chatbot_context.py`:

```python
def test_get_last_audit_change_finds_matching_experiment(tmp_path, monkeypatch):
    log_file = tmp_path / "session.log"
    log_file.write_text(
        "2026-07-10 10:00:00 | INFO  | lab.audit | Experiment created: id=5 name='Test' preset=None\n"
        "2026-07-10 10:00:05 | INFO  | lab.audit | Config updated: experiment_id=5 changed={\"lr\": [0.001, 0.003]}\n"
        "2026-07-10 10:00:06 | INFO  | lab.audit | Config updated: experiment_id=50 changed={\"lr\": [0.001, 0.003]}\n"
    )
    monkeypatch.setattr(context, "get_log_path", lambda: log_file)

    result = context._get_last_audit_change(5)

    assert result is not None
    assert "experiment_id=5 " in result
    assert "lr" in result


def test_get_last_audit_change_no_match_returns_none(tmp_path, monkeypatch):
    log_file = tmp_path / "session.log"
    log_file.write_text("2026-07-10 10:00:00 | INFO  | lab.audit | Experiment created: id=999 name='Other'\n")
    monkeypatch.setattr(context, "get_log_path", lambda: log_file)

    assert context._get_last_audit_change(5) is None


def test_get_last_audit_change_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(context, "get_log_path", lambda: tmp_path / "does_not_exist.log")
    assert context._get_last_audit_change(5) is None


def test_get_log_tail_returns_last_n_lines(tmp_path, monkeypatch):
    log_file = tmp_path / "session.log"
    log_file.write_text("\n".join(f"line {i}" for i in range(100)) + "\n")
    monkeypatch.setattr(context, "get_log_path", lambda: log_file)

    tail = context._get_log_tail(5)

    assert len(tail) == 5
    assert tail[-1].strip() == "line 99"
```

- [ ] **Step 12: Run it, verify it fails**

Run: `uv run pytest tests/test_chatbot_context.py -v`
Expected: FAIL — `AttributeError: module 'backend.chatbot.context' has no attribute '_get_last_audit_change'`

- [ ] **Step 13: Implement the audit-change and log-tail readers**

Append to `backend/chatbot/context.py`:

```python


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
```

- [ ] **Step 14: Run it, verify it passes**

Run: `uv run pytest tests/test_chatbot_context.py -v`
Expected: `8 passed`

- [ ] **Step 15: Document the fragile audit-matching decision**

Append to `docs/DESIGN_DECISIONS.md` (check the file's existing heading style first and match it):

```markdown
## Chatbot audit-log matching is a substring check, not structured data

`backend/chatbot/context.py::_get_last_audit_change()` finds the most
recent config/experiment change for the grounded chatbot by searching the
current session's log file for the substring `"id=<experiment_id> "`.
This works because every `audit_log.info(...)` call in
`backend/api/experiments.py` formats its message as either `id=%d ...` or
`experiment_id=%d ...`, and both forms end in `id=<N> ` followed by more
text — so one substring check covers both.

**This is coupled to the exact audit log message format.** If a future
`audit_log.info(...)` call site is added or changed such that the
experiment ID is no longer immediately followed by a space (e.g. it's
last on the line, or formatted as `id: 5` instead of `id=5`), the
chatbot will silently stop finding "last change" for that log line — no
error, just an empty/stale result. If you touch audit log message
formats, check `_get_last_audit_change()`'s tests
(`tests/test_chatbot_context.py`) still pass.
```

- [ ] **Step 16: Commit**

```bash
git add backend/chatbot/context.py tests/test_chatbot_context.py docs/DESIGN_DECISIONS.md
git commit -m "feat: add audit-change and log-tail readers to chatbot context"
```

- [ ] **Step 17: Write a failing test for full message assembly**

Add to `tests/test_chatbot_context.py`:

```python
def test_assemble_messages_structure(monkeypatch, tmp_path):
    log_file = tmp_path / "session.log"
    log_file.write_text(
        "2026-07-10 10:00:00 | INFO  | lab.audit | Config updated: experiment_id=1 changed={\"lr\": [0.001, 0.003]}\n"
    )
    monkeypatch.setattr(context, "get_log_path", lambda: log_file)

    experiment = {
        "id": 1,
        "name": "My experiment",
        "config_json": json.dumps({"template": "transformer", "description": "baseline"}),
    }
    run = {
        "status": "running",
        "current_step": 10,
        "total_steps": 100,
        "train_loss_history": json.dumps([{"step": 10, "train_loss": 2.0, "val_loss": 2.1}]),
    }
    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]

    messages = context.assemble_messages(experiment, run, history, "What does this loss mean?")

    assert messages[0]["role"] == "system"
    assert "grounded lab assistant" in messages[0]["content"]
    assert messages[1]["role"] == "system"
    assert "RotaryPositionalEncoding" in messages[1]["content"]
    assert messages[2] == {"role": "user", "content": "earlier question"}
    assert messages[3] == {"role": "assistant", "content": "earlier answer"}
    last = messages[-1]
    assert last["role"] == "user"
    assert "What does this loss mean?" in last["content"]
    assert "running" in last["content"]
    assert "experiment_id=1" in last["content"]
```

- [ ] **Step 18: Run it, verify it fails**

Run: `uv run pytest tests/test_chatbot_context.py -v`
Expected: FAIL — `AttributeError: module 'backend.chatbot.context' has no attribute 'assemble_messages'`

- [ ] **Step 19: Implement message assembly**

Append to `backend/chatbot/context.py`:

```python


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
```

- [ ] **Step 20: Run it, verify it passes**

Run: `uv run pytest tests/test_chatbot_context.py -v`
Expected: `9 passed`

- [ ] **Step 21: Commit**

```bash
git add backend/chatbot/context.py tests/test_chatbot_context.py
git commit -m "feat: add assemble_messages to chatbot context"
```

---

### Task 7: API endpoints

**Files:**
- Create: `backend/api/chatbot.py`
- Modify: `backend/main.py`
- Test: `tests/test_api_chatbot.py`

- [ ] **Step 1: Write failing integration tests**

Create `tests/test_api_chatbot.py`:

```python
import json

import pytest
from httpx import ASGITransport, AsyncClient

from backend import db
from backend.chatbot import client as tf_client
from backend.main import app


@pytest.fixture
async def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_lab.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    await db.init_db()
    exp_id = await db.create_experiment(
        "Test experiment", {"template": "transformer", "description": "baseline"}
    )
    return exp_id


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_get_messages_empty_for_new_experiment(temp_db, client):
    resp = await client.get(f"/api/chatbot/{temp_db}/messages")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_get_messages_404_for_unknown_experiment(client):
    resp = await client.get("/api/chatbot/999999/messages")
    assert resp.status_code == 404


async def test_post_message_returns_503_when_not_configured(temp_db, client, monkeypatch):
    monkeypatch.setattr(tf_client, "is_configured", lambda: False)
    resp = await client.post(f"/api/chatbot/{temp_db}/message", json={"message": "hello"})
    assert resp.status_code == 503


async def test_post_message_streams_and_persists(temp_db, client, monkeypatch):
    monkeypatch.setattr(tf_client, "is_configured", lambda: True)

    async def fake_stream(messages):
        yield "Hello", None
        yield " there", None
        yield "", {"prompt_tokens": 50, "completion_tokens": 2, "total_tokens": 52}

    monkeypatch.setattr(tf_client, "stream_completion", fake_stream)

    resp = await client.post(f"/api/chatbot/{temp_db}/message", json={"message": "hi"})

    assert resp.status_code == 200
    assert "data:" in resp.text
    assert "event: done" in resp.text

    messages = await db.get_chat_messages(temp_db)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "hi"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Hello there"
    assert messages[1]["total_tokens"] == 52


async def test_post_message_history_excludes_message_being_sent(temp_db, client, monkeypatch):
    monkeypatch.setattr(tf_client, "is_configured", lambda: True)
    captured = {}

    async def fake_stream(messages):
        captured["messages"] = messages
        yield "ok", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

    monkeypatch.setattr(tf_client, "stream_completion", fake_stream)

    await client.post(f"/api/chatbot/{temp_db}/message", json={"message": "first"})
    await client.post(f"/api/chatbot/{temp_db}/message", json={"message": "second"})

    last_user_turn = captured["messages"][-1]["content"]
    assert "second" in last_user_turn
    # "first" should appear exactly once total across all messages sent to
    # the model (as prior history), not duplicated into the current turn too
    occurrences = sum(1 for m in captured["messages"] if m["content"] == "first")
    assert occurrences == 1
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/test_api_chatbot.py -v`
Expected: FAIL — `404` for all routes (router not registered) or import errors, since `backend/api/chatbot.py` doesn't exist yet.

- [ ] **Step 3: Implement the router**

Create `backend/api/chatbot.py`:

```python
"""Grounded chatbot endpoints — SSE chat streaming and history."""

import json
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend import db
from backend.chatbot import client as tf_client
from backend.chatbot.context import assemble_messages
from backend.logging_config import chatbot_log
from config.settings import settings

router = APIRouter(prefix="/api/chatbot", tags=["chatbot"])


class ChatMessageRequest(BaseModel):
    message: str


@router.get("/{experiment_id}/messages")
async def get_messages(experiment_id: int):
    exp = await db.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(404, "Experiment not found")
    return await db.get_chat_messages(experiment_id)


@router.post("/{experiment_id}/message")
async def post_message(experiment_id: int, req: ChatMessageRequest):
    exp = await db.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(404, "Experiment not found")

    if not tf_client.is_configured():
        chatbot_log.warning(
            "Chat request for experiment_id=%d but NEBIUS_KEY not configured", experiment_id
        )
        raise HTTPException(503, "Chatbot unavailable — no Token Factory API key configured")

    # Fetch history BEFORE writing the new user message, so assemble_messages
    # doesn't see the current message twice (once in history, once stapled
    # as the current turn).
    history = await db.get_chat_messages(
        experiment_id, limit=settings.chatbot_history_window_turns * 2
    )
    await db.add_chat_message(experiment_id, "user", req.message)

    runs = await db.list_runs_for_experiment(experiment_id)
    latest_run = runs[0] if runs else None
    messages = assemble_messages(exp, latest_run, history, req.message)

    async def event_stream():
        full_text = []
        usage_info = None
        start = time.perf_counter()
        try:
            async for delta, usage in tf_client.stream_completion(messages):
                if delta:
                    full_text.append(delta)
                    yield f"data: {json.dumps({'delta': delta})}\n\n"
                if usage is not None:
                    usage_info = usage
        except Exception as exc:
            chatbot_log.error("Chat stream failed for experiment_id=%d: %s", experiment_id, exc)
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
            return

        latency_ms = int((time.perf_counter() - start) * 1000)
        assistant_text = "".join(full_text)
        await db.add_chat_message(
            experiment_id,
            "assistant",
            assistant_text,
            prompt_tokens=(usage_info or {}).get("prompt_tokens"),
            completion_tokens=(usage_info or {}).get("completion_tokens"),
            total_tokens=(usage_info or {}).get("total_tokens"),
            latency_ms=latency_ms,
        )
        yield f"event: done\ndata: {json.dumps({'usage': usage_info})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

- [ ] **Step 4: Wire the router into the app**

In `backend/main.py`, add the import alongside the other routers:

```python
from backend.api.experiments import router as experiments_router
from backend.api.training import router as training_router
from backend.api.codegen import router as code_router
from backend.api.chatbot import router as chatbot_router
```

And register it alongside the others:

```python
app.include_router(experiments_router)
app.include_router(training_router)
app.include_router(code_router)
app.include_router(chatbot_router)
```

- [ ] **Step 5: Run it, verify it passes**

Run: `uv run pytest tests/test_api_chatbot.py -v`
Expected: `5 passed`

- [ ] **Step 6: Run the full backend test suite**

Run: `uv run pytest -q`
Expected: all tests pass (should be around 26-27 total across every test file added in this plan).

- [ ] **Step 7: Commit**

```bash
git add backend/api/chatbot.py backend/main.py tests/test_api_chatbot.py
git commit -m "feat: add chatbot SSE endpoint and wire into app"
```

---

### Task 8: Manual smoke check against the real app (no mocks)

This is not a new automated test — it's a manual sanity check that the whole stack boots and the 503 path works correctly when `NEBIUS_KEY` genuinely isn't set in the environment (proves the "unavailable" UX described in the spec, §9).

- [ ] **Step 1: Start the backend**

```bash
cd .worktrees/grounded-chatbot
uv run uvicorn backend.main:app --reload
```

- [ ] **Step 2: Create a test experiment and hit the chat endpoint**

In another terminal:

```bash
curl -s -X POST http://localhost:8000/api/experiments/from-preset/baseline_transformer
# note the returned experiment_id, then:
curl -i -X POST http://localhost:8000/api/chatbot/<experiment_id>/message \
  -H "Content-Type: application/json" \
  -d '{"message": "what does this config do?"}'
```

Expected: `503` if `.env`'s `NEBIUS_KEY` isn't loaded in this shell, or a streaming `text/event-stream` response with real model output if it is. Either outcome is correct — this step confirms the endpoint is live and reachable, not that Token Factory itself is reachable (that's outside this plan's automated test coverage, since it would require live network calls and a real key in CI).

- [ ] **Step 3: Stop the server**

`Ctrl-C` in the uvicorn terminal.

No commit for this task — it's verification only, nothing changes.

---

## Definition of done for Plan A

- [ ] All tasks 1–7 committed on `feature/grounded-chatbot`
- [ ] `uv run pytest -q` passes with 0 failures
- [ ] Task 8 manual check performed at least once
- [ ] Ready to hand off to Plan B (frontend), which will call `GET/POST /api/chatbot/{id}/...` against this backend
