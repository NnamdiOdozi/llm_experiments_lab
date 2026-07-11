import pytest

from backend import db
from backend.nebius import endpoints_client, worker_manager
from backend.training.worker_status import WorkerStatus


@pytest.fixture
async def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_lab.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    await db.init_db()


async def test_ensure_worker_creates_new_endpoint_when_none_exists(temp_db, monkeypatch):
    async def fake_create_endpoint(**kwargs):
        return "aiendpoint-new"

    async def fake_get_endpoint(endpoint_id):
        return {"status": {"state": "RUNNING", "public_endpoints": ["https://new.tunnel.nebius.cloud"]}}

    monkeypatch.setattr(endpoints_client, "create_endpoint", fake_create_endpoint)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)

    worker = await worker_manager.ensure_worker("cpu")

    assert worker["worker_status"] == WorkerStatus.READY
    assert worker["nebius_endpoint_id"] == "aiendpoint-new"
    assert worker["endpoint_url"] == "https://new.tunnel.nebius.cloud"


async def test_ensure_worker_reuses_ready_worker_without_recreating(temp_db, monkeypatch):
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.READY,
        nebius_endpoint_id="aiendpoint-existing", endpoint_url="https://existing.tunnel.nebius.cloud",
    )

    async def fail_if_called(**kwargs):
        raise AssertionError("should not create a new endpoint when one is already READY")

    monkeypatch.setattr(endpoints_client, "create_endpoint", fail_if_called)
    monkeypatch.setattr(endpoints_client, "start_endpoint", fail_if_called)

    worker = await worker_manager.ensure_worker("cpu")

    assert worker["nebius_endpoint_id"] == "aiendpoint-existing"
    assert worker["endpoint_url"] == "https://existing.tunnel.nebius.cloud"


async def test_ensure_worker_starts_stopped_endpoint(temp_db, monkeypatch):
    await db.create_worker_session("worker-gpu", "gpu", "nebius_endpoint", 600)
    await db.update_worker_session(
        "worker-gpu", worker_status=WorkerStatus.STOPPED, nebius_endpoint_id="aiendpoint-existing",
    )

    started = {}

    async def fake_start_endpoint(endpoint_id):
        started["endpoint_id"] = endpoint_id

    async def fake_get_endpoint(endpoint_id):
        return {"status": {"state": "RUNNING", "public_endpoints": ["https://restarted.tunnel.nebius.cloud"]}}

    async def fail_if_called(**kwargs):
        raise AssertionError("should not create a new endpoint when one already exists")

    monkeypatch.setattr(endpoints_client, "create_endpoint", fail_if_called)
    monkeypatch.setattr(endpoints_client, "start_endpoint", fake_start_endpoint)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)

    worker = await worker_manager.ensure_worker("cuda")

    assert started["endpoint_id"] == "aiendpoint-existing"
    assert worker["worker_status"] == WorkerStatus.READY
    assert worker["endpoint_url"] == "https://restarted.tunnel.nebius.cloud"


async def test_ensure_worker_raises_on_timeout(temp_db, monkeypatch):
    monkeypatch.setattr(worker_manager.settings, "nebius_endpoint_ready_timeout_seconds", 2)
    monkeypatch.setattr(worker_manager.settings, "nebius_endpoint_poll_interval_seconds", 1)

    async def fake_create_endpoint(**kwargs):
        return "aiendpoint-slow"

    async def fake_get_endpoint(endpoint_id):
        return {"status": {"state": "PROVISIONING", "public_endpoints": []}}

    async def fake_sleep(seconds):
        pass

    monkeypatch.setattr(endpoints_client, "create_endpoint", fake_create_endpoint)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)
    monkeypatch.setattr(worker_manager.asyncio, "sleep", fake_sleep)

    with pytest.raises(worker_manager.WorkerProvisionError):
        await worker_manager.ensure_worker("cpu")

    session = await db.get_worker_session("worker-cpu")
    assert session["worker_status"] == WorkerStatus.FAILED
