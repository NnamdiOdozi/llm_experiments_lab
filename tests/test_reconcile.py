"""Tests for startup orphaned-run reconciliation (db.reconcile_orphaned_runs).

Fable review, 2026-07-14 — DESIGN_DECISIONS §70: reconciliation used to mark
EVERY active run failed on startup, including remote runs still legitimately
training in the Nebius container (which survives a local API restart). Only
local runs — plus remote runs that died mid-provisioning (remote_run_id
still NULL) — are truly orphaned by a restart.
"""

import pytest

from backend import db
from backend.training.status import RunStatus


@pytest.fixture
async def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_lab.db")
    await db.init_db()
    return await db.create_experiment("Reconcile test", {"template": "transformer"})


async def test_reconcile_marks_local_but_spares_live_remote_runs(temp_db):
    exp_id = temp_db

    local_running = await db.create_training_run(exp_id, device="cpu", execution_backend="local")
    await db.update_training_run(local_running, status=RunStatus.RUNNING)

    # Remote run mid-training — the container outlives a local restart
    remote_running = await db.create_training_run(
        exp_id, device="cuda", execution_backend="nebius_endpoint", remote_run_id=7,
    )
    await db.update_training_run(remote_running, status=RunStatus.RUNNING)

    # Remote run that died mid-provisioning — nothing remote exists yet
    remote_provisioning = await db.create_training_run(
        exp_id, device="cuda", execution_backend="nebius_endpoint",
    )  # status stays QUEUED, remote_run_id stays NULL

    # Paused local run — has a checkpoint, must survive restarts (PAUSED
    # is deliberately not in ACTIVE_STATUSES)
    local_paused = await db.create_training_run(exp_id, device="cpu", execution_backend="local")
    await db.update_training_run(local_paused, status=RunStatus.PAUSED)

    count = await db.reconcile_orphaned_runs()

    assert count == 2  # local_running + remote_provisioning
    assert (await db.get_training_run(local_running))["status"] == RunStatus.FAILED
    assert (await db.get_training_run(remote_provisioning))["status"] == RunStatus.FAILED
    assert (await db.get_training_run(remote_running))["status"] == RunStatus.RUNNING
    assert (await db.get_training_run(local_paused))["status"] == RunStatus.PAUSED
