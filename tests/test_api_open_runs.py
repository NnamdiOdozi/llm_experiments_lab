import pytest
from httpx import ASGITransport, AsyncClient

from backend import db
from backend.main import app
from backend.training.status import RunStatus


@pytest.fixture
async def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_lab.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    await db.init_db()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_open_runs_lists_active_runs_with_experiment_name(temp_db, client):
    exp_id = await db.create_experiment("My Experiment", {"template": "transformer"})
    run_id = await db.create_training_run(exp_id, "cpu")
    await db.update_training_run(run_id, status=RunStatus.RUNNING)

    resp = await client.get("/api/training/open")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["id"] == run_id
    assert body[0]["experiment_name"] == "My Experiment"
    assert body[0]["status"] == "running"


async def test_open_runs_excludes_completed_runs(temp_db, client):
    exp_id = await db.create_experiment("My Experiment", {"template": "transformer"})
    run_id = await db.create_training_run(exp_id, "cpu")
    await db.update_training_run(run_id, status=RunStatus.COMPLETED)

    resp = await client.get("/api/training/open")

    assert resp.json() == []
