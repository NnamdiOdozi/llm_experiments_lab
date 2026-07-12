import pytest

from backend import db
from backend.training.status import RunStatus


@pytest.fixture
async def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_lab.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    await db.init_db()


async def test_list_runs_for_experiment_orders_by_id_not_started_at(temp_db):
    """Regression test (2026-07-12): the chatbot's grounding picks
    list_runs_for_experiment(...)[0] as "the current run". Sorting by
    started_at (nullable, only set once training actually begins) instead
    of id let an older, already-terminal run outrank a genuinely newer,
    still-active one — the chatbot told the user their run was "cancelled
    at step 0" while they were actively prompting a paused run at step 307.
    """
    exp_id = await db.create_experiment("Exp", {"template": "transformer"})

    # Older run: started and finished quickly, has an early started_at.
    older_run_id = await db.create_training_run(exp_id, "cpu")
    await db.update_training_run(
        older_run_id, status=RunStatus.CANCELLED, started_at="2026-07-12 10:00:00",
    )

    # Newer run: created later, but its started_at is NULL (e.g. still
    # provisioning/queued) — must still sort ahead of the older run.
    newer_run_id = await db.create_training_run(exp_id, "cpu")
    await db.update_training_run(newer_run_id, status=RunStatus.PAUSED, current_step=307)

    runs = await db.list_runs_for_experiment(exp_id)

    assert runs[0]["id"] == newer_run_id
    assert runs[0]["status"] == RunStatus.PAUSED
    assert runs[0]["current_step"] == 307
