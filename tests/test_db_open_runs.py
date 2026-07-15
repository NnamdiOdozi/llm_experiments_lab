import pytest

from backend import db
from backend.training.status import RunStatus


@pytest.fixture
async def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_lab.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    await db.init_db()


async def test_list_open_runs_includes_active_and_paused(temp_db):
    exp_id = await db.create_experiment("Exp A", {"template": "transformer"})
    running_id = await db.create_training_run(exp_id, "cpu")
    await db.update_training_run(running_id, status=RunStatus.RUNNING)
    paused_id = await db.create_training_run(exp_id, "cpu")
    await db.update_training_run(paused_id, status=RunStatus.PAUSED)

    open_runs = await db.list_runs()

    run_ids = [r["id"] for r in open_runs]
    assert running_id in run_ids
    assert paused_id in run_ids


async def test_list_open_runs_excludes_terminal_statuses(temp_db):
    exp_id = await db.create_experiment("Exp B", {"template": "transformer"})
    completed_id = await db.create_training_run(exp_id, "cpu")
    await db.update_training_run(completed_id, status=RunStatus.COMPLETED)
    failed_id = await db.create_training_run(exp_id, "cpu")
    await db.update_training_run(failed_id, status=RunStatus.FAILED)
    cancelled_id = await db.create_training_run(exp_id, "cpu")
    await db.update_training_run(cancelled_id, status=RunStatus.CANCELLED)

    open_runs = await db.list_runs()

    run_ids = [r["id"] for r in open_runs]
    assert completed_id not in run_ids
    assert failed_id not in run_ids
    assert cancelled_id not in run_ids


async def test_list_runs_include_terminal_returns_everything(temp_db):
    """Direct user request, 2026-07-15 — no UI path back to a run once it's
    finished/failed/stopped. include_terminal=True is what feeds that.
    See docs/DESIGN_DECISIONS.md §79b."""
    exp_id = await db.create_experiment("Exp C", {"template": "transformer"})
    running_id = await db.create_training_run(exp_id, "cpu")
    await db.update_training_run(running_id, status=RunStatus.RUNNING)
    completed_id = await db.create_training_run(exp_id, "cpu")
    await db.update_training_run(completed_id, status=RunStatus.COMPLETED)
    failed_id = await db.create_training_run(exp_id, "cpu")
    await db.update_training_run(failed_id, status=RunStatus.FAILED)
    cancelled_id = await db.create_training_run(exp_id, "cpu")
    await db.update_training_run(cancelled_id, status=RunStatus.CANCELLED)

    all_runs = await db.list_runs(include_terminal=True)

    run_ids = [r["id"] for r in all_runs]
    assert running_id in run_ids
    assert completed_id in run_ids
    assert failed_id in run_ids
    assert cancelled_id in run_ids


async def test_list_runs_respects_limit(temp_db):
    exp_id = await db.create_experiment("Exp D", {"template": "transformer"})
    for _ in range(5):
        run_id = await db.create_training_run(exp_id, "cpu")
        await db.update_training_run(run_id, status=RunStatus.COMPLETED)

    limited = await db.list_runs(include_terminal=True, limit=3)

    assert len(limited) == 3


async def test_list_open_runs_includes_experiment_name(temp_db):
    exp_id = await db.create_experiment("My Cool Experiment", {"template": "transformer"})
    run_id = await db.create_training_run(exp_id, "cpu")
    await db.update_training_run(run_id, status=RunStatus.QUEUED)

    open_runs = await db.list_runs()

    row = next(r for r in open_runs if r["id"] == run_id)
    assert row["experiment_name"] == "My Cool Experiment"
