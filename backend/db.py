"""SQLite database for experiments and training runs."""

import json
import sqlite3
import aiosqlite

from backend.training.status import RunStatus, ACTIVE_STATUSES, TERMINAL_STATUSES
from backend.training.worker_status import WorkerStatus, TERMINAL_WORKER_STATUSES
from config.settings import settings

DB_PATH = settings.database_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    config_json TEXT NOT NULL,
    notes_md TEXT DEFAULT '',
    preset_key TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    status TEXT NOT NULL DEFAULT 'queued',
    device TEXT NOT NULL DEFAULT 'cpu',
    train_loss_history TEXT DEFAULT '[]',
    val_loss_history TEXT DEFAULT '[]',
    final_train_loss REAL,
    final_val_loss REAL,
    total_steps INTEGER DEFAULT 0,
    current_step INTEGER DEFAULT 0,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    config_snapshot TEXT,
    seed INTEGER,
    template_key TEXT,
    dataset_name TEXT,
    checkpoint_path TEXT,
    metrics_path TEXT,
    error_message TEXT,
    device_name TEXT,
    param_count INTEGER,
    package_versions TEXT,
    git_commit TEXT,
    execution_backend TEXT DEFAULT 'local',
    remote_endpoint_id TEXT,
    remote_run_id INTEGER,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

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

-- Remote worker (Nebius endpoint) lifecycle — separate from training_runs.
-- One worker_session can host multiple training_runs over its lifetime.
CREATE TABLE IF NOT EXISTS worker_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL UNIQUE,
    device_type TEXT NOT NULL,
    backend_type TEXT NOT NULL,
    worker_status TEXT NOT NULL DEFAULT 'none',
    nebius_endpoint_id TEXT,
    endpoint_url TEXT,
    actual_platform TEXT,
    actual_preset TEXT,
    idle_timeout_seconds INTEGER NOT NULL,
    last_activity_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Columns added after initial schema — ALTER TABLE for existing DBs
_MIGRATIONS = [
    ("config_snapshot", "TEXT"),
    ("seed", "INTEGER"),
    ("template_key", "TEXT"),
    ("dataset_name", "TEXT"),
    ("checkpoint_path", "TEXT"),
    ("metrics_path", "TEXT"),
    ("error_message", "TEXT"),
    ("device_name", "TEXT"),
    ("param_count", "INTEGER"),
    ("package_versions", "TEXT"),
    ("git_commit", "TEXT"),
    ("execution_backend", "TEXT DEFAULT 'local'"),
    ("remote_endpoint_id", "TEXT"),
    ("remote_run_id", "INTEGER"),
]

# worker_sessions columns added after initial schema
_WORKER_SESSION_MIGRATIONS = [
    ("actual_platform", "TEXT"),
    ("actual_preset", "TEXT"),
]


async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db():
    db = await get_db()
    await db.executescript(SCHEMA)
    # Apply migrations for existing DBs (ALTER TABLE is idempotent with IF NOT EXISTS check)
    for col_name, col_type in _MIGRATIONS:
        try:
            await db.execute(f"ALTER TABLE training_runs ADD COLUMN {col_name} {col_type}")
        except Exception:
            pass  # Column already exists
    for col_name, col_type in _WORKER_SESSION_MIGRATIONS:
        try:
            await db.execute(f"ALTER TABLE worker_sessions ADD COLUMN {col_name} {col_type}")
        except Exception:
            pass  # Column already exists
    await db.commit()
    await db.close()


async def create_experiment(name: str, config: dict, preset_key: str | None = None) -> int:
    db = await get_db()
    cursor = await db.execute(
        "INSERT INTO experiments (name, config_json, preset_key) VALUES (?, ?, ?)",
        (name, json.dumps(config), preset_key),
    )
    await db.commit()
    row_id = cursor.lastrowid
    await db.close()
    return row_id


async def get_experiment(experiment_id: int) -> dict | None:
    db = await get_db()
    cursor = await db.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
    row = await cursor.fetchone()
    await db.close()
    if row is None:
        return None
    return dict(row)


async def list_experiments() -> list[dict]:
    db = await get_db()
    cursor = await db.execute("SELECT * FROM experiments ORDER BY created_at DESC")
    rows = await cursor.fetchall()
    await db.close()
    return [dict(r) for r in rows]


async def create_training_run(experiment_id: int, device: str = "cpu", **extra) -> int:
    cols = ["experiment_id", "device", "status"] + list(extra.keys())
    placeholders = ", ".join("?" for _ in cols)
    col_names = ", ".join(cols)
    values = [experiment_id, device, RunStatus.QUEUED] + list(extra.values())
    db = await get_db()
    cursor = await db.execute(
        f"INSERT INTO training_runs ({col_names}) VALUES ({placeholders})",
        values,
    )
    await db.commit()
    row_id = cursor.lastrowid
    await db.close()
    return row_id


async def update_training_run(run_id: int, **kwargs):
    db = await get_db()
    set_clause = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [run_id]
    await db.execute(f"UPDATE training_runs SET {set_clause} WHERE id = ?", values)
    await db.commit()
    await db.close()


async def get_training_run(run_id: int) -> dict | None:
    db = await get_db()
    cursor = await db.execute("SELECT * FROM training_runs WHERE id = ?", (run_id,))
    row = await cursor.fetchone()
    await db.close()
    if row is None:
        return None
    return dict(row)


async def list_runs_for_experiment(experiment_id: int) -> list[dict]:
    db = await get_db()
    # ORDER BY id, not started_at — started_at is nullable and only gets set
    # once training actually begins, not at row creation, so a run that
    # hasn't started yet (or whose started_at otherwise doesn't line up with
    # true creation order) can sort ahead of a genuinely newer run. id is an
    # AUTOINCREMENT primary key, always monotonic at creation time, never
    # NULL — the same pattern list_open_runs() already uses correctly.
    # Found live 2026-07-12: the chatbot's "latest run" (runs[0] here) was a
    # stale, already-cancelled earlier run instead of the actual paused run
    # at step 307, misleading the chatbot's whole grounding. See
    # docs/DESIGN_DECISIONS.md.
    cursor = await db.execute(
        "SELECT * FROM training_runs WHERE experiment_id = ? ORDER BY id DESC",
        (experiment_id,),
    )
    rows = await cursor.fetchall()
    await db.close()
    return [dict(r) for r in rows]


async def list_open_runs() -> list[dict]:
    """Every run not in a terminal state, across all experiments — the set a
    user would want to see/stop from an "Experiments" overview page.
    """
    statuses = tuple(TERMINAL_STATUSES)
    placeholders = ",".join("?" for _ in statuses)
    db = await get_db()
    cursor = await db.execute(
        f"SELECT training_runs.*, experiments.name AS experiment_name "
        f"FROM training_runs JOIN experiments ON experiments.id = training_runs.experiment_id "
        f"WHERE training_runs.status NOT IN ({placeholders}) "
        f"ORDER BY training_runs.id DESC",
        statuses,
    )
    rows = await cursor.fetchall()
    await db.close()
    return [dict(r) for r in rows]


async def add_chat_message(
    experiment_id: int,
    role: str,
    content: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    latency_ms: int | None = None,
) -> int:
    db = await get_db()
    cursor = await db.execute(
        "INSERT INTO chat_messages "
        "(experiment_id, role, content, prompt_tokens, completion_tokens, total_tokens, latency_ms) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (experiment_id, role, content, prompt_tokens, completion_tokens, total_tokens, latency_ms),
    )
    await db.commit()
    row_id = cursor.lastrowid
    await db.close()
    return row_id


async def get_chat_messages(experiment_id: int, limit: int | None = None) -> list[dict]:
    """Chat history for an experiment, oldest first.

    limit=None returns full history (for the UI). limit=N returns only the
    most recent N messages, still in chronological order (for the sliding
    window sent to the LLM) — same function serves both callers.
    """
    db = await get_db()
    if limit is None:
        cursor = await db.execute(
            "SELECT * FROM chat_messages WHERE experiment_id = ? ORDER BY id ASC",
            (experiment_id,),
        )
        rows = await cursor.fetchall()
    else:
        cursor = await db.execute(
            "SELECT * FROM chat_messages WHERE experiment_id = ? ORDER BY id DESC LIMIT ?",
            (experiment_id, limit),
        )
        rows = list(reversed(await cursor.fetchall()))
    await db.close()
    return [dict(r) for r in rows]


# ── Worker sessions (Track B remote job workers) ──

async def create_worker_session(
    session_id: str, device_type: str, backend_type: str, idle_timeout_seconds: int,
) -> int:
    db = await get_db()
    cursor = await db.execute(
        "INSERT INTO worker_sessions "
        "(session_id, device_type, backend_type, idle_timeout_seconds) "
        "VALUES (?, ?, ?, ?)",
        (session_id, device_type, backend_type, idle_timeout_seconds),
    )
    await db.commit()
    row_id = cursor.lastrowid
    await db.close()
    return row_id


async def get_worker_session(session_id: str) -> dict | None:
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM worker_sessions WHERE session_id = ?", (session_id,),
    )
    row = await cursor.fetchone()
    await db.close()
    if row is None:
        return None
    return dict(row)


async def update_worker_session(session_id: str, **kwargs):
    set_clause = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [session_id]
    db = await get_db()
    await db.execute(
        f"UPDATE worker_sessions SET {set_clause} WHERE session_id = ?", values,
    )
    await db.commit()
    await db.close()


async def touch_worker_session(session_id: str):
    """Bump last_activity_at — called on any command/heartbeat for idle-timeout tracking."""
    db = await get_db()
    await db.execute(
        "UPDATE worker_sessions SET last_activity_at = CURRENT_TIMESTAMP "
        "WHERE session_id = ?",
        (session_id,),
    )
    await db.commit()
    await db.close()


async def list_active_worker_sessions() -> list[dict]:
    statuses = tuple(TERMINAL_WORKER_STATUSES)
    placeholders = ",".join("?" for _ in statuses)
    db = await get_db()
    cursor = await db.execute(
        f"SELECT * FROM worker_sessions WHERE worker_status NOT IN ({placeholders})",
        statuses,
    )
    rows = await cursor.fetchall()
    await db.close()
    return [dict(r) for r in rows]


# ── Sync versions for use in training threads ──

def sync_update_training_run(run_id: int, **kwargs):
    """Synchronous DB update for use in background training threads."""
    conn = sqlite3.connect(DB_PATH)
    set_clause = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [run_id]
    conn.execute(f"UPDATE training_runs SET {set_clause} WHERE id = ?", values)
    conn.commit()
    conn.close()


async def reconcile_orphaned_runs() -> int:
    """Mark runs with active status as failed — called on startup to clear stale state.
    After a backend restart no worker processes exist, so any 'active' row is orphaned.
    """
    statuses = tuple(ACTIVE_STATUSES)
    placeholders = ",".join("?" for _ in statuses)
    db = await get_db()
    cursor = await db.execute(
        f"UPDATE training_runs SET status = 'failed', error_message = 'Backend restarted — worker lost' "
        f"WHERE status IN ({placeholders})",
        statuses,
    )
    count = cursor.rowcount
    await db.commit()
    await db.close()
    return count


async def count_active_runs_in_db(device_filter: str | None = None) -> int:
    """Count runs with active statuses in DB — survives API restarts."""
    statuses = tuple(ACTIVE_STATUSES)
    placeholders = ",".join("?" for _ in statuses)
    query = f"SELECT COUNT(*) FROM training_runs WHERE status IN ({placeholders})"
    params: list = list(statuses)
    if device_filter:
        query += " AND device LIKE ?"
        params.append(f"{device_filter}%")
    db = await get_db()
    cursor = await db.execute(query, params)
    row = await cursor.fetchone()
    await db.close()
    return row[0] if row else 0


async def get_run_status_from_db(run_id: int) -> dict | None:
    """Read run status from DB — survives backend restarts."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT id, experiment_id, status, current_step, total_steps, "
        "template_key, started_at, completed_at, config_snapshot, execution_backend "
        "FROM training_runs WHERE id = ?",
        (run_id,),
    )
    row = await cursor.fetchone()
    await db.close()
    if row is None:
        return None
    r = dict(row)
    # Compute total_steps from config if not stored
    total = r.get("total_steps") or 0
    if total == 0 and r.get("config_snapshot"):
        try:
            cfg = json.loads(r["config_snapshot"])
            t = cfg.get("training", {})
            total = t.get("max_iters", t.get("epochs", 0) * 100)
        except (json.JSONDecodeError, TypeError):
            pass
    return {
        "run_id": r["id"],
        "status": r["status"],
        "current_step": r["current_step"] or 0,
        "total_steps": total,
        "metrics_count": 0,
        "template": r.get("template_key") or "transformer",
        "elapsed_seconds": 0,
        "from_db": True,
        "execution_backend": r.get("execution_backend") or "local",
    }
