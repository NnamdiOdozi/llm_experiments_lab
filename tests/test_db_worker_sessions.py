import pytest

from backend import db
from backend.training.worker_status import WorkerStatus


@pytest.fixture
async def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_lab.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    await db.init_db()


async def test_create_and_get_worker_session(temp_db):
    await db.create_worker_session(
        session_id="sess-1", device_type="cpu", backend_type="nebius_endpoint",
        idle_timeout_seconds=1800,
    )

    session = await db.get_worker_session("sess-1")

    assert session["session_id"] == "sess-1"
    assert session["device_type"] == "cpu"
    assert session["backend_type"] == "nebius_endpoint"
    assert session["idle_timeout_seconds"] == 1800
    assert session["worker_status"] == WorkerStatus.NONE
    assert session["nebius_endpoint_id"] is None
    assert session["endpoint_url"] is None


async def test_get_worker_session_returns_none_for_unknown(temp_db):
    session = await db.get_worker_session("no-such-session")
    assert session is None


async def test_update_worker_session_status(temp_db):
    await db.create_worker_session(
        session_id="sess-2", device_type="gpu", backend_type="nebius_endpoint",
        idle_timeout_seconds=600,
    )

    await db.update_worker_session(
        "sess-2", worker_status=WorkerStatus.READY,
        nebius_endpoint_id="aiendpoint-abc123", endpoint_url="https://example.tunnel.nebius.cloud",
    )

    session = await db.get_worker_session("sess-2")
    assert session["worker_status"] == WorkerStatus.READY
    assert session["nebius_endpoint_id"] == "aiendpoint-abc123"
    assert session["endpoint_url"] == "https://example.tunnel.nebius.cloud"


async def test_touch_worker_session_sets_last_activity(temp_db):
    await db.create_worker_session(
        session_id="sess-3", device_type="cpu", backend_type="nebius_endpoint",
        idle_timeout_seconds=1800,
    )

    await db.touch_worker_session("sess-3")

    session = await db.get_worker_session("sess-3")
    assert session["last_activity_at"] is not None


async def test_list_active_worker_sessions_excludes_terminal(temp_db):
    await db.create_worker_session(
        session_id="sess-active", device_type="cpu", backend_type="nebius_endpoint",
        idle_timeout_seconds=1800,
    )
    await db.create_worker_session(
        session_id="sess-stopped", device_type="cpu", backend_type="nebius_endpoint",
        idle_timeout_seconds=1800,
    )
    await db.update_worker_session("sess-stopped", worker_status=WorkerStatus.STOPPED)

    active = await db.list_active_worker_sessions()

    session_ids = [s["session_id"] for s in active]
    assert "sess-active" in session_ids
    assert "sess-stopped" not in session_ids
