"""SQLite database for experiments and training runs."""

import json
import sqlite3
import aiosqlite
from pathlib import Path

DB_PATH = Path("lab.db")

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
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);
"""


async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db():
    db = await get_db()
    await db.executescript(SCHEMA)
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


async def create_training_run(experiment_id: int, device: str = "cpu") -> int:
    db = await get_db()
    cursor = await db.execute(
        "INSERT INTO training_runs (experiment_id, device, status) VALUES (?, ?, 'queued')",
        (experiment_id, device),
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
