from datetime import datetime, timedelta, timezone

import pytest

from backend import db
from backend.nebius import endpoints_client, idle_monitor
from backend.nebius.endpoints_client import NebiusEndpointError
from backend.training.worker_status import WorkerStatus


@pytest.fixture
async def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_lab.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    await db.init_db()


def test_seconds_since_computes_elapsed_time():
    ten_seconds_ago = (datetime.now(timezone.utc) - timedelta(seconds=10)).strftime("%Y-%m-%d %H:%M:%S")

    elapsed = idle_monitor.seconds_since(ten_seconds_ago)

    assert 9 <= elapsed <= 12


async def test_stop_idle_workers_stops_endpoint_past_timeout(temp_db, monkeypatch):
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", idle_timeout_seconds=0)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.READY, nebius_endpoint_id="aiendpoint-abc123",
    )

    stopped_ids = []

    async def fake_stop_endpoint(endpoint_id):
        stopped_ids.append(endpoint_id)

    monkeypatch.setattr(endpoints_client, "stop_endpoint", fake_stop_endpoint)

    count = await idle_monitor.stop_idle_workers()

    assert count == 1
    assert stopped_ids == ["aiendpoint-abc123"]
    session = await db.get_worker_session("worker-cpu")
    assert session["worker_status"] == WorkerStatus.STOPPED


async def test_stop_idle_workers_ignores_sessions_not_idle_enough(temp_db, monkeypatch):
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", idle_timeout_seconds=999999)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.READY, nebius_endpoint_id="aiendpoint-abc123",
    )

    async def fail_if_called(endpoint_id):
        raise AssertionError("should not stop a worker that isn't idle past its timeout")

    monkeypatch.setattr(endpoints_client, "stop_endpoint", fail_if_called)

    count = await idle_monitor.stop_idle_workers()

    assert count == 0


async def test_stop_idle_workers_treats_already_deleted_endpoint_as_stopped(temp_db, monkeypatch):
    """Real incident, 2026-07-14: user deleted the Nebius endpoint manually
    (outside the app) after finishing with it. Every idle-scan afterward
    hit "rpc error: code = NotFound" trying to stop something already
    gone — previously that exception propagated straight past the DB
    update, so worker_status stayed READY forever (never actually
    reflecting reality), which is what made stop_training() believe a
    long-gone endpoint was still there to proxy a stop request to,
    permanently stranding any run using it. An endpoint that's already
    gone should be treated as already stopped. See
    docs/DESIGN_DECISIONS.md."""
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", idle_timeout_seconds=0)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.READY, nebius_endpoint_id="aiendpoint-gone",
    )

    async def fake_stop_endpoint(endpoint_id):
        raise NebiusEndpointError(
            "nebius ai endpoint stop --id aiendpoint-gone failed (exit 13): "
            "Error: rpc error: code = NotFound desc = not found request = abc123"
        )

    monkeypatch.setattr(endpoints_client, "stop_endpoint", fake_stop_endpoint)

    count = await idle_monitor.stop_idle_workers()

    assert count == 1
    session = await db.get_worker_session("worker-cpu")
    assert session["worker_status"] == WorkerStatus.STOPPED


async def test_stop_idle_workers_still_raises_on_a_real_failure(temp_db, monkeypatch):
    """Only NotFound is swallowed — a genuine failure (network/auth/etc.)
    must still surface, not be silently treated as success. See
    docs/DESIGN_DECISIONS.md."""
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", idle_timeout_seconds=0)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.READY, nebius_endpoint_id="aiendpoint-abc123",
    )

    async def fake_stop_endpoint(endpoint_id):
        raise NebiusEndpointError("nebius ai endpoint stop --id aiendpoint-abc123 timed out")

    monkeypatch.setattr(endpoints_client, "stop_endpoint", fake_stop_endpoint)

    with pytest.raises(NebiusEndpointError):
        await idle_monitor.stop_idle_workers()

    session = await db.get_worker_session("worker-cpu")
    assert session["worker_status"] == WorkerStatus.READY


async def test_stop_idle_workers_skips_non_ready_sessions(temp_db, monkeypatch):
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", idle_timeout_seconds=0)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.PROVISIONING, nebius_endpoint_id="aiendpoint-abc123",
    )

    async def fail_if_called(endpoint_id):
        raise AssertionError("should not stop a worker that's still provisioning")

    monkeypatch.setattr(endpoints_client, "stop_endpoint", fail_if_called)

    count = await idle_monitor.stop_idle_workers()

    assert count == 0
