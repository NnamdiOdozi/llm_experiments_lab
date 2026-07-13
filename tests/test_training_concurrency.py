"""Per device x execution-backend concurrency limits (local cpu/gpu, serverless
cpu/gpu are independent — see config/settings.py and docs/DESIGN_DECISIONS.md).
"""

import pytest
from httpx import ASGITransport, AsyncClient

from backend import db
from config.settings import settings


@pytest.fixture
async def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_lab.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    await db.init_db()
    exp_id = await db.create_experiment(
        "Test experiment", {"template": "transformer"}, preset_key="tiny-shakespeare",
    )
    return exp_id


@pytest.fixture
async def client():
    from backend.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_count_active_runs_in_db_filters_by_device_and_backend(temp_db):
    exp_id = temp_db
    await db.create_training_run(exp_id, "cpu", execution_backend="local")
    await db.create_training_run(exp_id, "cpu", execution_backend="nebius_endpoint")
    await db.create_training_run(exp_id, "cuda", execution_backend="local")
    await db.create_training_run(exp_id, "cuda", execution_backend="nebius_endpoint")

    assert await db.count_active_runs_in_db("cpu", "local") == 1
    assert await db.count_active_runs_in_db("cpu", "nebius_endpoint") == 1
    assert await db.count_active_runs_in_db("cuda", "local") == 1
    assert await db.count_active_runs_in_db("cuda", "nebius_endpoint") == 1
    assert await db.count_active_runs_in_db("cpu") == 2
    assert await db.count_active_runs_in_db() == 4


async def test_start_training_rejects_local_cpu_over_limit(temp_db, client, monkeypatch):
    exp_id = temp_db
    monkeypatch.setattr(settings, "max_concurrent_local_cpu_runs", 0)

    resp = await client.post(
        "/api/training/start", json={"experiment_id": exp_id, "device": "cpu", "backend": "local"}
    )

    assert resp.status_code == 429
    assert "local CPU" in resp.json()["detail"]


async def test_start_training_local_and_serverless_limits_are_independent(temp_db, client, monkeypatch):
    """Zeroing the local CPU limit must not block a serverless CPU start."""
    exp_id = temp_db
    monkeypatch.setattr(settings, "max_concurrent_local_cpu_runs", 0)
    monkeypatch.setattr(settings, "max_concurrent_serverless_cpu_runs", 5)

    async def fake_start_remote_run(run_id, exp, device):
        return None

    import backend.api.training as training_module
    monkeypatch.setattr(training_module, "_start_remote_run", fake_start_remote_run)

    resp = await client.post(
        "/api/training/start",
        json={"experiment_id": exp_id, "device": "cpu", "backend": "nebius_endpoint"},
    )

    assert resp.status_code == 200


async def test_start_training_rejects_serverless_gpu_over_limit(temp_db, client, monkeypatch):
    exp_id = temp_db
    monkeypatch.setattr(settings, "max_concurrent_serverless_gpu_runs", 1)
    await db.create_training_run(exp_id, "cuda", execution_backend="nebius_endpoint")

    resp = await client.post(
        "/api/training/start",
        json={"experiment_id": exp_id, "device": "cuda", "backend": "nebius_endpoint"},
    )

    assert resp.status_code == 429
    assert "serverless GPU" in resp.json()["detail"]
