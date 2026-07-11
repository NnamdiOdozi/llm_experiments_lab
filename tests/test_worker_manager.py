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


async def test_ensure_worker_creates_new_endpoint_when_existing_one_was_deleted(temp_db, monkeypatch):
    """Endpoint was manually deleted outside the app — start_endpoint fails
    with 'not found'. Must self-heal by creating a fresh one, not 500."""
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.STOPPED, nebius_endpoint_id="aiendpoint-deleted",
    )

    async def fake_start_endpoint(endpoint_id):
        if endpoint_id == "aiendpoint-deleted":
            raise endpoints_client.NebiusEndpointError("nebius ai endpoint start failed (exit 1): not found")
        raise AssertionError(f"unexpected start_endpoint call for {endpoint_id}")

    async def fake_create_endpoint(**kwargs):
        return "aiendpoint-fresh"

    async def fake_get_endpoint(endpoint_id):
        assert endpoint_id == "aiendpoint-fresh"
        return {"status": {"state": "RUNNING", "public_endpoints": ["https://fresh.tunnel.nebius.cloud"]}}

    monkeypatch.setattr(endpoints_client, "start_endpoint", fake_start_endpoint)
    monkeypatch.setattr(endpoints_client, "create_endpoint", fake_create_endpoint)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)

    worker = await worker_manager.ensure_worker("cpu")

    assert worker["worker_status"] == WorkerStatus.READY
    assert worker["nebius_endpoint_id"] == "aiendpoint-fresh"
    assert worker["endpoint_url"] == "https://fresh.tunnel.nebius.cloud"


async def test_ensure_worker_creates_new_endpoint_when_it_disappears_mid_poll(temp_db, monkeypatch):
    """Endpoint existed at start_endpoint time, was polled once, then got deleted
    before reaching RUNNING — must create a second endpoint and succeed on it."""
    created_ids = ["aiendpoint-first", "aiendpoint-second"]
    get_calls = {"count": 0}

    async def fake_create_endpoint(**kwargs):
        return created_ids.pop(0)

    async def fake_get_endpoint(endpoint_id):
        get_calls["count"] += 1
        if get_calls["count"] == 1:
            assert endpoint_id == "aiendpoint-first"
            return {"status": {"state": "PROVISIONING", "public_endpoints": []}}
        if get_calls["count"] == 2:
            assert endpoint_id == "aiendpoint-first"
            raise endpoints_client.NebiusEndpointError("nebius ai endpoint get failed (exit 1): not found")
        assert endpoint_id == "aiendpoint-second"
        return {"status": {"state": "RUNNING", "public_endpoints": ["https://second.tunnel.nebius.cloud"]}}

    async def fake_sleep(seconds):
        pass

    monkeypatch.setattr(endpoints_client, "create_endpoint", fake_create_endpoint)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)
    monkeypatch.setattr(worker_manager.asyncio, "sleep", fake_sleep)

    worker = await worker_manager.ensure_worker("cpu")

    assert worker["worker_status"] == WorkerStatus.READY
    assert worker["nebius_endpoint_id"] == "aiendpoint-second"
    assert worker["endpoint_url"] == "https://second.tunnel.nebius.cloud"
    assert get_calls["count"] == 3


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
