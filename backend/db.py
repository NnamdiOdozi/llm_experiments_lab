"""SQLite database for experiments and training runs."""

import json
import sqlite3
import aiosqlite

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
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
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
    values = [experiment_id, device, "queued"] + list(extra.values())
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
    cursor = await db.execute(
        "SELECT * FROM training_runs WHERE experiment_id = ? ORDER BY started_at DESC",
        (experiment_id,),
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
    active_statuses = ("queued", "starting", "running", "pause_requested", "checkpointing", "resuming")
    placeholders = ",".join("?" for _ in active_statuses)
    db = await get_db()
    cursor = await db.execute(
        f"UPDATE training_runs SET status = 'failed', error_message = 'Backend restarted — worker lost' "
        f"WHERE status IN ({placeholders})",
        active_statuses,
    )
    count = cursor.rowcount
    await db.commit()
    await db.close()
    return count


async def count_active_runs_in_db(device_filter: str | None = None) -> int:
    """Count runs with active statuses in DB — survives API restarts."""
    active_statuses = ("queued", "starting", "running", "pause_requested", "checkpointing", "resuming")
    placeholders = ",".join("?" for _ in active_statuses)
    query = f"SELECT COUNT(*) FROM training_runs WHERE status IN ({placeholders})"
    params: list = list(active_statuses)
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
        "template_key, started_at, completed_at, config_snapshot "
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
    }
