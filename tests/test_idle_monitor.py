from datetime import datetime, timedelta, timezone

import pytest

from backend import db
from backend.nebius import endpoints_client, idle_monitor
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
