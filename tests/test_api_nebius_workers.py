import pytest
from httpx import ASGITransport, AsyncClient

from backend import db
from backend.main import app
from backend.training.worker_status import WorkerStatus


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


async def test_worker_status_returns_none_when_no_worker_exists(temp_db, client):
    resp = await client.get("/api/nebius/workers/cpu")

    assert resp.status_code == 200
    assert resp.json()["worker_status"] == "none"


async def test_worker_status_returns_idle_seconds_and_thresholds(temp_db, client):
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", idle_timeout_seconds=1800)
    await db.update_worker_session("worker-cpu", worker_status=WorkerStatus.READY, nebius_endpoint_id="aiendpoint-1")

    resp = await client.get("/api/nebius/workers/cpu")

    body = resp.json()
    assert body["worker_status"] == "ready"
    assert body["idle_timeout_seconds"] == 1800
    assert body["warning_seconds"] == 600
    assert body["seconds_idle"] < 5


async def test_worker_status_uses_gpu_thresholds_for_cuda_device(temp_db, client):
    await db.create_worker_session("worker-gpu", "gpu", "nebius_endpoint", idle_timeout_seconds=600)
    await db.update_worker_session("worker-gpu", worker_status=WorkerStatus.READY, nebius_endpoint_id="aiendpoint-2")

    resp = await client.get("/api/nebius/workers/cuda")

    body = resp.json()
    assert body["idle_timeout_seconds"] == 600
    assert body["warning_seconds"] == 300


async def test_heartbeat_touches_last_activity(temp_db, client):
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", idle_timeout_seconds=1800)

    resp = await client.post("/api/nebius/workers/cpu/heartbeat")

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


async def test_heartbeat_is_a_noop_when_no_worker_exists(temp_db, client):
    resp = await client.post("/api/nebius/workers/cpu/heartbeat")

    assert resp.status_code == 200
