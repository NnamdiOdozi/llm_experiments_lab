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


# --- T7: Reconciler updates error to failed (Part 5 / D6) ---
async def test_reconciler_updates_error_to_failed(temp_db, monkeypatch):
    """Part 5 / D6: Reconciler converges ERROR endpoint status to FAILED."""
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.READY,
        nebius_endpoint_id="aiendpoint-error",
    )

    async def fake_get_endpoint(endpoint_id):
        return {"status": {"state": "ERROR"}}

    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)

    await idle_monitor.reconcile_worker_sessions()

    session = await db.get_worker_session("worker-cpu")
    assert session["worker_status"] == WorkerStatus.FAILED


# --- T8: Reconciler updates stopping to shutting_down (Part 5 / D6) ---
async def test_reconciler_updates_stopping_to_shutting_down(temp_db, monkeypatch):
    """Part 5 / D6: Reconciler maps STOPPING → SHUTTING_DOWN (gives SHUTTING_DOWN its writer)."""
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.READY,
        nebius_endpoint_id="aiendpoint-stopping",
    )

    async def fake_get_endpoint(endpoint_id):
        return {"status": {"state": "STOPPING"}}

    async def fake_probe(url):
        return True

    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)
    monkeypatch.setattr(endpoints_client, "probe_endpoint_url", fake_probe)

    await idle_monitor.reconcile_worker_sessions()

    session = await db.get_worker_session("worker-cpu")
    assert session["worker_status"] == WorkerStatus.SHUTTING_DOWN


async def test_reconciler_skips_sessions_locked_by_a_provisioning_task(temp_db, monkeypatch):
    """A live provisioning task owns its session (holds the per-device lock);
    the reconciler must not fight it — no live-state fetch, no DB write."""
    from backend.nebius import worker_manager

    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.STARTING, nebius_endpoint_id="aiendpoint-midflight",
    )

    async def fail_if_called(endpoint_id):
        raise AssertionError("reconciler must not query a locked session's endpoint")

    monkeypatch.setattr(endpoints_client, "get_endpoint", fail_if_called)

    lock = worker_manager._lock_for("worker-cpu")
    async with lock:
        await idle_monitor.reconcile_worker_sessions()

    session = await db.get_worker_session("worker-cpu")
    assert session["worker_status"] == WorkerStatus.STARTING  # untouched


async def test_reconciler_one_session_failure_does_not_stop_the_scan(temp_db, monkeypatch):
    """One session's CLI failure must not prevent the others from converging."""
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.READY, nebius_endpoint_id="aiendpoint-cli-broken",
    )
    await db.create_worker_session("worker-gpu", "gpu", "nebius_endpoint", 600)
    await db.update_worker_session(
        "worker-gpu", worker_status=WorkerStatus.STARTING, nebius_endpoint_id="aiendpoint-actually-stopped",
    )

    async def fake_get_endpoint(endpoint_id):
        if endpoint_id == "aiendpoint-cli-broken":
            # Not a NebiusEndpointError (that maps to exists=False) — an
            # unexpected crash, e.g. malformed CLI output.
            raise RuntimeError("unexpected CLI output")
        return {"status": {"state": "STOPPED"}}

    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)

    await idle_monitor.reconcile_worker_sessions()

    gpu = await db.get_worker_session("worker-gpu")
    assert gpu["worker_status"] == WorkerStatus.STOPPED  # still converged


async def test_stop_idle_workers_recheck_skips_worker_touched_after_scan_read(temp_db, monkeypatch):
    """TOCTOU guard: a worker claimed (touched) between the scan's session
    read and the stop call must not be stopped under its new run."""
    # Nonzero timeout: overdue per the ancient last_activity_at below, but a
    # fresh touch puts it comfortably back under the limit (timeout=0 would
    # make even a just-touched worker count as idle).
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", idle_timeout_seconds=3600)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.READY, nebius_endpoint_id="aiendpoint-claimed",
        last_activity_at="2020-01-01 00:00:00",
    )

    real_get = db.get_worker_session
    touched = {"done": False}

    async def get_and_touch(session_id):
        # Simulate a run claiming the worker between the scan's list read
        # and the pre-stop re-check: the FIRST re-fetch inside
        # stop_idle_workers sees a freshly touched clock.
        if not touched["done"]:
            touched["done"] = True
            await db.touch_worker_session(session_id)
        return await real_get(session_id)

    async def fail_if_called(endpoint_id):
        raise AssertionError("must not stop a worker that was touched after the scan read")

    monkeypatch.setattr(db, "get_worker_session", get_and_touch)
    monkeypatch.setattr(endpoints_client, "stop_endpoint", fail_if_called)

    count = await idle_monitor.stop_idle_workers()

    assert count == 0
    session = await real_get("worker-cpu")
    assert session["worker_status"] == WorkerStatus.READY
