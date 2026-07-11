import pytest
from httpx import ASGITransport, AsyncClient

from backend import db
from backend.api import nebius as nebius_api
from backend.main import app
from backend.nebius import endpoints_client
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


async def test_worker_status_reports_local_backend_mode_by_default(temp_db, client):
    resp = await client.get("/api/nebius/workers/cpu")

    body = resp.json()
    assert body["backend_mode"] == "local"
    assert body["preset"] is None


async def test_worker_status_reports_backend_mode(temp_db, client, monkeypatch):
    monkeypatch.setattr(nebius_api.settings, "training_backend", "nebius_endpoint")

    resp = await client.get("/api/nebius/workers/cpu")

    assert resp.json()["backend_mode"] == "nebius_endpoint"


async def test_worker_status_reports_actual_preset_not_configured_preset(temp_db, client, monkeypatch):
    """Regression test for the 2026-07-11 incident: config said 8vcpu-32gb but
    the real running endpoint was still 4vcpu-16gb. preset must reflect what's
    actually running (worker_sessions.actual_preset), never the config value."""
    monkeypatch.setattr(nebius_api.settings, "training_backend", "nebius_endpoint")
    monkeypatch.setattr(nebius_api.settings, "nebius_cpu_preset", "8vcpu-32gb")
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", idle_timeout_seconds=1800)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.READY,
        nebius_endpoint_id="aiendpoint-1", actual_platform="cpu-d3", actual_preset="4vcpu-16gb",
    )

    resp = await client.get("/api/nebius/workers/cpu")

    assert resp.json()["preset"] == "4vcpu-16gb"


async def test_worker_status_preset_is_none_before_any_endpoint_exists(temp_db, client, monkeypatch):
    """No endpoint has ever been created yet — must not guess from config."""
    monkeypatch.setattr(nebius_api.settings, "training_backend", "nebius_endpoint")
    monkeypatch.setattr(nebius_api.settings, "nebius_cpu_preset", "8vcpu-32gb")

    resp = await client.get("/api/nebius/workers/cpu")

    assert resp.json()["preset"] is None


async def test_get_worker_logs_returns_raw_text(temp_db, client, monkeypatch):
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", idle_timeout_seconds=1800)
    await db.update_worker_session("worker-cpu", nebius_endpoint_id="aiendpoint-1")

    async def fake_get_logs(endpoint_id, tail=200):
        assert endpoint_id == "aiendpoint-1"
        return "2026-07-11T22:00:00Z INFO Uvicorn running\n"

    monkeypatch.setattr(endpoints_client, "get_logs", fake_get_logs)

    resp = await client.get("/api/nebius/workers/cpu/logs")

    assert resp.status_code == 200
    assert resp.json() == {"logs": "2026-07-11T22:00:00Z INFO Uvicorn running\n"}


async def test_get_worker_logs_returns_empty_when_no_worker_exists(temp_db, client):
    resp = await client.get("/api/nebius/workers/cpu/logs")

    assert resp.status_code == 200
    assert resp.json() == {"logs": ""}


async def test_heartbeat_touches_last_activity(temp_db, client):
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", idle_timeout_seconds=1800)

    resp = await client.post("/api/nebius/workers/cpu/heartbeat")

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


async def test_heartbeat_is_a_noop_when_no_worker_exists(temp_db, client):
    resp = await client.post("/api/nebius/workers/cpu/heartbeat")

    assert resp.status_code == 200
