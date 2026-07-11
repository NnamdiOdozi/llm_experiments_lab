"""Regression tests for the 2026-07-11 incident: the training-panel tag showed
'Serverless' for a run whose execution_backend was actually 'local', because
the tag read the global /api/nebius/workers/{device} status instead of the
specific run's own execution_backend. Fix: every run-status response must
carry execution_backend so the frontend can tell per-run truth from global
config.
"""
import pytest

from backend import db
from backend.training.status import RunStatus


@pytest.fixture
async def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_lab.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    await db.init_db()


async def test_get_run_status_from_db_includes_execution_backend_local(temp_db):
    exp_id = await db.create_experiment("Exp", {"template": "transformer"})
    run_id = await db.create_training_run(exp_id, "cpu")
    await db.update_training_run(run_id, status=RunStatus.FAILED)

    status = await db.get_run_status_from_db(run_id)

    assert status["execution_backend"] == "local"


async def test_get_run_status_from_db_includes_execution_backend_remote(temp_db):
    exp_id = await db.create_experiment("Exp", {"template": "transformer"})
    run_id = await db.create_training_run(
        exp_id, "cpu", execution_backend="nebius_endpoint", remote_endpoint_id="aiendpoint-1", remote_run_id=7,
    )
    await db.update_training_run(run_id, status=RunStatus.RUNNING)

    status = await db.get_run_status_from_db(run_id)

    assert status["execution_backend"] == "nebius_endpoint"
