import pytest

from backend import db
from backend.nebius import endpoints_client, worker_manager
from backend.training.worker_status import WorkerStatus


@pytest.fixture
async def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_lab.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    await db.init_db()


@pytest.fixture(autouse=True)
def no_live_endpoint_found_by_default(monkeypatch):
    """create_new_worker() now checks for a live, already-RUNNING endpoint
    before creating one — default every test to "none found" (matching
    behavior before that check existed) so each test doesn't need its own
    boilerplate mock for it. The one test that actually exercises adoption
    overrides this itself."""
    async def fake_find_running_endpoint(name):
        return None
    monkeypatch.setattr(endpoints_client, "find_running_endpoint", fake_find_running_endpoint)


async def test_create_new_worker_uses_the_correct_image_per_device_type(temp_db, monkeypatch):
    """Regression guard for the CPU/GPU image split (2026-07-12) — a CPU
    worker must never accidentally get the CUDA-bearing GPU image, and
    vice versa."""
    monkeypatch.setattr(worker_manager.settings, "nebius_cpu_trainer_image", "registry/llm-lab-trainer-cpu:latest")
    monkeypatch.setattr(worker_manager.settings, "nebius_gpu_trainer_image", "registry/llm-lab-trainer-gpu:latest")
    captured = {}

    async def fake_create_endpoint(**kwargs):
        captured["image"] = kwargs["image"]
        return "aiendpoint-new"

    async def fake_get_endpoint(endpoint_id):
        return {
            "spec": {"platform": "cpu-d3", "preset": "4vcpu-16gb"},
            "status": {"state": "RUNNING", "public_endpoints": ["https://new.tunnel.nebius.cloud"]},
        }

    monkeypatch.setattr(endpoints_client, "create_endpoint", fake_create_endpoint)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)

    await worker_manager.ensure_worker("cpu")
    assert captured["image"] == "registry/llm-lab-trainer-cpu:latest"

    await worker_manager.ensure_worker("cuda")
    assert captured["image"] == "registry/llm-lab-trainer-gpu:latest"


async def test_ensure_worker_creates_new_endpoint_when_none_exists(temp_db, monkeypatch):
    async def fake_create_endpoint(**kwargs):
        return "aiendpoint-new"

    async def fake_get_endpoint(endpoint_id):
        return {
            "spec": {"platform": "cpu-d3", "preset": "4vcpu-16gb"},
            "status": {"state": "RUNNING", "public_endpoints": ["https://new.tunnel.nebius.cloud"]},
        }

    monkeypatch.setattr(endpoints_client, "create_endpoint", fake_create_endpoint)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)

    worker = await worker_manager.ensure_worker("cpu")

    assert worker["worker_status"] == WorkerStatus.READY
    assert worker["nebius_endpoint_id"] == "aiendpoint-new"
    assert worker["endpoint_url"] == "https://new.tunnel.nebius.cloud"


async def test_ensure_worker_records_the_actual_platform_and_preset(temp_db, monkeypatch):
    """Regression test for the 2026-07-11 incident: the frontend showed the
    *configured* nebius_cpu_preset (8vcpu-32gb) while the real endpoint was
    still 4vcpu-16gb, because nothing captured the endpoint's real spec."""
    async def fake_create_endpoint(**kwargs):
        return "aiendpoint-new"

    async def fake_get_endpoint(endpoint_id):
        return {
            "spec": {"platform": "cpu-d3", "preset": "4vcpu-16gb"},
            "status": {"state": "RUNNING", "public_endpoints": ["https://new.tunnel.nebius.cloud"]},
        }

    monkeypatch.setattr(endpoints_client, "create_endpoint", fake_create_endpoint)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)
    # Config says 8vcpu-32gb, but the fake endpoint's real spec is 4vcpu-16gb —
    # the stored value must be the real one, not the configured one.
    monkeypatch.setattr(worker_manager.settings, "nebius_cpu_preset", "8vcpu-32gb")

    worker = await worker_manager.ensure_worker("cpu")

    assert worker["actual_platform"] == "cpu-d3"
    assert worker["actual_preset"] == "4vcpu-16gb"


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


async def test_ensure_worker_skips_straight_to_create_when_shutting_down(temp_db, monkeypatch):
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.SHUTTING_DOWN, nebius_endpoint_id="aiendpoint-dying",
    )

    async def fail_if_called(endpoint_id):
        raise AssertionError("should not attempt to start a worker that's shutting down")

    async def fake_create_endpoint(**kwargs):
        return "aiendpoint-fresh"

    async def fake_get_endpoint(endpoint_id):
        return {
            "spec": {"platform": "cpu-d3", "preset": "4vcpu-16gb"},
            "status": {"state": "RUNNING", "public_endpoints": ["https://fresh.tunnel.nebius.cloud"]},
        }

    monkeypatch.setattr(endpoints_client, "start_endpoint", fail_if_called)
    monkeypatch.setattr(endpoints_client, "create_endpoint", fake_create_endpoint)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)

    worker = await worker_manager.ensure_worker("cpu")

    assert worker["nebius_endpoint_id"] == "aiendpoint-fresh"
    assert worker["worker_status"] == WorkerStatus.READY


async def test_ensure_worker_adopts_existing_live_endpoint_instead_of_duplicating(temp_db, monkeypatch):
    """Regression test for the 2026-07-12 incident: a CPU endpoint created
    out-of-band (e.g. via scripts/create_nebius_endpoint.py before it wrote
    to the DB) was already RUNNING, but the app's DB had no row for it —
    ensure_worker created a second, duplicate CPU endpoint. Must adopt the
    live one by name instead."""
    async def fake_find_running_endpoint(name):
        assert name == "llm-lab-cpu-trainer"
        return {"metadata": {"id": "aiendpoint-adopted"}}

    async def fail_if_called(**kwargs):
        raise AssertionError("should not create a duplicate when a live RUNNING endpoint already exists")

    async def fake_get_endpoint(endpoint_id):
        assert endpoint_id == "aiendpoint-adopted"
        return {
            "spec": {"platform": "cpu-d3", "preset": "8vcpu-32gb"},
            "status": {"state": "RUNNING", "public_endpoints": ["https://adopted.tunnel.nebius.cloud"]},
        }

    monkeypatch.setattr(endpoints_client, "find_running_endpoint", fake_find_running_endpoint)
    monkeypatch.setattr(endpoints_client, "create_endpoint", fail_if_called)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)

    worker = await worker_manager.ensure_worker("cpu")

    assert worker["nebius_endpoint_id"] == "aiendpoint-adopted"
    assert worker["worker_status"] == WorkerStatus.READY
    assert worker["endpoint_url"] == "https://adopted.tunnel.nebius.cloud"


async def test_ensure_worker_debounces_when_already_provisioning(temp_db, monkeypatch):
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.PROVISIONING, nebius_endpoint_id="aiendpoint-existing",
    )

    async def fail_if_called(**kwargs):
        raise AssertionError("should not create or start a second worker while one is already mid-provision")

    monkeypatch.setattr(endpoints_client, "create_endpoint", fail_if_called)
    monkeypatch.setattr(endpoints_client, "start_endpoint", fail_if_called)

    with pytest.raises(worker_manager.WorkerBusyError):
        await worker_manager.ensure_worker("cpu")


async def test_ensure_worker_debounces_when_already_starting(temp_db, monkeypatch):
    await db.create_worker_session("worker-gpu", "gpu", "nebius_endpoint", 600)
    await db.update_worker_session(
        "worker-gpu", worker_status=WorkerStatus.STARTING, nebius_endpoint_id="aiendpoint-existing",
    )

    async def fail_if_called(**kwargs):
        raise AssertionError("should not create or start a second worker while one is already mid-start")

    monkeypatch.setattr(endpoints_client, "create_endpoint", fail_if_called)
    monkeypatch.setattr(endpoints_client, "start_endpoint", fail_if_called)

    with pytest.raises(worker_manager.WorkerBusyError):
        await worker_manager.ensure_worker("cuda")


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
