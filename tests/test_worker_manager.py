import asyncio

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
    """create_new_worker() now checks for a live RUNNING endpoint and a
    STOPPED one before creating fresh — default every test to "neither
    found" (matching behavior before those checks existed) so each test
    doesn't need its own boilerplate mock for them. Tests that actually
    exercise adoption/restart override these themselves."""
    async def fake_find_running_endpoint(name):
        return None
    async def fake_find_endpoint(name, state):
        return None
    monkeypatch.setattr(endpoints_client, "find_running_endpoint", fake_find_running_endpoint)
    monkeypatch.setattr(endpoints_client, "find_endpoint", fake_find_endpoint)


@pytest.fixture(autouse=True)
def reachable_by_default(monkeypatch):
    """ensure_worker()'s READY liveness check now also probes the tunnel
    URL itself (§79 — Nebius can report RUNNING while the tunnel routes
    nowhere), only reached when get_endpoint() already returned RUNNING.
    Default every test to "reachable" so existing/new tests exercising the
    READY-reuse path don't each need their own boilerplate mock; the one
    test that specifically covers a dead tunnel overrides this itself."""
    async def fake_probe(url):
        return True
    monkeypatch.setattr(endpoints_client, "probe_endpoint_url", fake_probe)


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

    async def fake_get_endpoint(endpoint_id):
        assert endpoint_id == "aiendpoint-existing"
        return {"status": {"state": "RUNNING"}}

    monkeypatch.setattr(endpoints_client, "create_endpoint", fail_if_called)
    monkeypatch.setattr(endpoints_client, "start_endpoint", fail_if_called)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)

    worker = await worker_manager.ensure_worker("cpu")

    assert worker["nebius_endpoint_id"] == "aiendpoint-existing"
    assert worker["endpoint_url"] == "https://existing.tunnel.nebius.cloud"


async def test_ensure_worker_reprovisions_when_ready_row_points_at_deleted_endpoint(temp_db, monkeypatch):
    """Regression test for the 2026-07-12 incident: the user deleted every
    endpoint via the Nebius console. Nothing tells our DB that happened, so
    the READY row was stale — the app kept reusing a dead endpoint's URL,
    404ing on every single run until the DB was fixed by hand. Must verify
    liveness before trusting READY, and re-provision if it's actually gone."""
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.READY,
        nebius_endpoint_id="aiendpoint-deleted-manually", endpoint_url="https://gone.tunnel.nebius.cloud",
    )

    async def fake_get_endpoint(endpoint_id):
        if endpoint_id == "aiendpoint-deleted-manually":
            raise endpoints_client.NebiusEndpointError("nebius ai endpoint get failed (exit 1): not found")
        assert endpoint_id == "aiendpoint-fresh"
        return {"status": {"state": "RUNNING", "public_endpoints": ["https://fresh.tunnel.nebius.cloud"]}}

    async def fake_start_endpoint(endpoint_id):
        raise endpoints_client.NebiusEndpointError("nebius ai endpoint start failed (exit 1): not found")

    async def fake_create_endpoint(**kwargs):
        return "aiendpoint-fresh"

    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)
    monkeypatch.setattr(endpoints_client, "start_endpoint", fake_start_endpoint)
    monkeypatch.setattr(endpoints_client, "create_endpoint", fake_create_endpoint)

    worker = await worker_manager.ensure_worker("cpu")

    assert worker["nebius_endpoint_id"] == "aiendpoint-fresh"
    assert worker["endpoint_url"] == "https://fresh.tunnel.nebius.cloud"


async def test_ensure_worker_reprovisions_when_ready_endpoint_tunnel_is_dead(temp_db, monkeypatch):
    """Real incident, 2026-07-15: a CPU endpoint reported State: RUNNING
    (and its own container logs showed a clean startup, no crash) while its
    public tunnel URL returned a bare 404 for every path — Nebius's gateway
    responding, not this app. State alone isn't enough to trust READY; the
    URL itself must actually answer. See docs/DESIGN_DECISIONS.md §79."""
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.READY,
        nebius_endpoint_id="aiendpoint-dead-tunnel", endpoint_url="https://dead.tunnel.nebius.cloud",
    )

    async def fake_get_endpoint(endpoint_id):
        if endpoint_id == "aiendpoint-dead-tunnel":
            return {"status": {"state": "RUNNING"}}  # Nebius says fine
        assert endpoint_id == "aiendpoint-fresh"
        return {"status": {"state": "RUNNING", "public_endpoints": ["https://fresh.tunnel.nebius.cloud"]}}

    async def fake_probe(url):
        assert url == "https://dead.tunnel.nebius.cloud"
        return False  # tunnel doesn't actually answer, despite state RUNNING

    async def fake_start_endpoint(endpoint_id):
        raise endpoints_client.NebiusEndpointError("should not attempt to start the dead-tunnel endpoint")

    async def fake_create_endpoint(**kwargs):
        return "aiendpoint-fresh"

    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)
    monkeypatch.setattr(endpoints_client, "probe_endpoint_url", fake_probe)
    monkeypatch.setattr(endpoints_client, "start_endpoint", fake_start_endpoint)
    monkeypatch.setattr(endpoints_client, "create_endpoint", fake_create_endpoint)

    worker = await worker_manager.ensure_worker("cpu")

    assert worker["nebius_endpoint_id"] == "aiendpoint-fresh"
    assert worker["endpoint_url"] == "https://fresh.tunnel.nebius.cloud"


async def test_ensure_worker_serializes_concurrent_calls_for_same_device(temp_db, monkeypatch):
    """Real incident, 2026-07-15: Nebius support traced a CPU/GPU endpoint
    ERROR state to two overlapping Start commands from this app's own
    service account. ensure_worker() used to read worker_status, then only
    commit STARTING/PROVISIONING several lines later, after the actual
    network call — nothing locked that gap, so two /start requests landing
    close together could both read the pre-commit status and both fire a
    Start/create command. Two concurrent calls must never both reach
    create_endpoint; the second must get a fast WorkerBusyError instead.
    See docs/DESIGN_DECISIONS.md §79a."""
    call_count = 0
    in_flight = False

    async def fake_create_endpoint(**kwargs):
        nonlocal call_count, in_flight
        assert not in_flight, "two concurrent calls both entered create_endpoint — lock did not serialize them"
        in_flight = True
        call_count += 1
        await asyncio.sleep(0.05)  # simulate real network latency — gives an unlocked race a chance to manifest
        in_flight = False
        return "aiendpoint-fresh"

    async def fake_get_endpoint(endpoint_id):
        return {"status": {"state": "RUNNING", "public_endpoints": ["https://fresh.tunnel.nebius.cloud"]}}

    monkeypatch.setattr(endpoints_client, "create_endpoint", fake_create_endpoint)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)

    results = await asyncio.gather(
        worker_manager.ensure_worker("cpu"),
        worker_manager.ensure_worker("cpu"),
        return_exceptions=True,
    )

    assert call_count == 1
    errors = [r for r in results if isinstance(r, Exception)]
    successes = [r for r in results if not isinstance(r, Exception)]
    assert len(errors) == 1 and isinstance(errors[0], worker_manager.WorkerBusyError)
    assert len(successes) == 1
    assert successes[0]["nebius_endpoint_id"] == "aiendpoint-fresh"


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


async def test_create_new_worker_restarts_stopped_endpoint_instead_of_creating(temp_db, monkeypatch):
    """A stopped endpoint the app's DB doesn't know about (e.g. the local
    worker_sessions row was lost, or it was created out-of-band) should be
    restarted, not abandoned in favor of a brand new one — this was the
    2026-07-12 GPU incident: a stopped endpoint sat idle while a second one
    got created for the same device type."""
    async def fake_find_endpoint(name, state):
        assert name == "llm-lab-gpu-trainer"
        assert state == "STOPPED"
        return {"metadata": {"id": "aiendpoint-restarted"}}

    async def fail_if_called(**kwargs):
        raise AssertionError("should not create a new endpoint when a stopped one already exists")

    started = {}

    async def fake_start_endpoint(endpoint_id):
        started["id"] = endpoint_id

    async def fake_get_endpoint(endpoint_id):
        assert endpoint_id == "aiendpoint-restarted"
        return {
            "spec": {"platform": "gpu-h100-1", "preset": "16vcpu-200gb-1gpu"},
            "status": {"state": "RUNNING", "public_endpoints": ["https://restarted.tunnel.nebius.cloud"]},
        }

    monkeypatch.setattr(endpoints_client, "find_endpoint", fake_find_endpoint)
    monkeypatch.setattr(endpoints_client, "create_endpoint", fail_if_called)
    monkeypatch.setattr(endpoints_client, "start_endpoint", fake_start_endpoint)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)

    worker = await worker_manager.ensure_worker("cuda")

    assert started["id"] == "aiendpoint-restarted"
    assert worker["nebius_endpoint_id"] == "aiendpoint-restarted"
    assert worker["worker_status"] == WorkerStatus.READY
    assert worker["endpoint_url"] == "https://restarted.tunnel.nebius.cloud"


async def test_create_new_worker_resets_stale_idle_clock(temp_db, monkeypatch):
    """Real bug, 2026-07-15: create_new_worker() (and the STARTING
    transitions in ensure_worker()) only ever updated worker_status, never
    last_activity_at — so a freshly (re)provisioned worker inherited
    whatever idle-clock value it had from before, sometimes a much older,
    already-near-expiry timestamp. Live symptom: the idle-timeout warning
    banner showed "stopping in a few minutes" for a worker that had just
    started provisioning for a brand new run. See
    docs/DESIGN_DECISIONS.md."""
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    # Simulate a worker whose last real activity was ages ago — e.g. it sat
    # idle-stopped for a long time before this new run came in to reuse it.
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.STOPPED, nebius_endpoint_id="aiendpoint-old",
        last_activity_at="2020-01-01 00:00:00",
    )

    async def fake_start_endpoint(endpoint_id):
        assert endpoint_id == "aiendpoint-old"

    async def fake_get_endpoint(endpoint_id):
        return {
            "spec": {"platform": "cpu-d3", "preset": "4vcpu-16gb"},
            "status": {"state": "RUNNING", "public_endpoints": ["https://fresh.tunnel.nebius.cloud"]},
        }

    monkeypatch.setattr(endpoints_client, "start_endpoint", fake_start_endpoint)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)

    await worker_manager.ensure_worker("cpu")

    session = await db.get_worker_session("worker-cpu")
    from backend.nebius.idle_monitor import seconds_since
    assert seconds_since(session["last_activity_at"]) < 10


async def test_ensure_worker_debounces_when_already_provisioning(temp_db, monkeypatch):
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.PROVISIONING, nebius_endpoint_id="aiendpoint-existing",
    )

    async def fail_if_called(**kwargs):
        raise AssertionError("should not create or start a second worker while one is already mid-provision")

    async def fake_get_endpoint(endpoint_id):
        # Nebius agrees it's genuinely still provisioning — the busy claim
        # is real, must not be treated as stale. See docs/DESIGN_DECISIONS.md.
        return {"status": {"state": "PROVISIONING"}}

    monkeypatch.setattr(endpoints_client, "create_endpoint", fail_if_called)
    monkeypatch.setattr(endpoints_client, "start_endpoint", fail_if_called)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)

    with pytest.raises(worker_manager.WorkerBusyError):
        await worker_manager.ensure_worker("cpu")


async def test_ensure_worker_debounces_when_already_starting(temp_db, monkeypatch):
    await db.create_worker_session("worker-gpu", "gpu", "nebius_endpoint", 600)
    await db.update_worker_session(
        "worker-gpu", worker_status=WorkerStatus.STARTING, nebius_endpoint_id="aiendpoint-existing",
    )

    async def fail_if_called(**kwargs):
        raise AssertionError("should not create or start a second worker while one is already mid-start")

    async def fake_get_endpoint(endpoint_id):
        return {"status": {"state": "STARTING"}}

    monkeypatch.setattr(endpoints_client, "create_endpoint", fail_if_called)
    monkeypatch.setattr(endpoints_client, "start_endpoint", fail_if_called)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)

    with pytest.raises(worker_manager.WorkerBusyError):
        await worker_manager.ensure_worker("cuda")


async def test_ensure_worker_ignores_stale_busy_status_when_endpoint_was_stopped_out_of_band(temp_db, monkeypatch):
    """Real incident, 2026-07-15: manually stopped the CPU endpoint a
    second time (to test the earlier fix), then started a new run — got
    rejected with "already being provisioned" even though nothing was
    provisioning anymore; worker_status was a stale STARTING nothing had
    ever re-verified or cleared. The busy check now verifies against
    Nebius before trusting a PROVISIONING/STARTING claim, same as the
    READY path already does. See docs/DESIGN_DECISIONS.md."""
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.STARTING, nebius_endpoint_id="aiendpoint-was-starting",
    )
    monkeypatch.setattr(worker_manager.settings, "nebius_endpoint_ready_timeout_seconds", 2)
    monkeypatch.setattr(worker_manager.settings, "nebius_endpoint_poll_interval_seconds", 1)

    async def fake_get_endpoint(endpoint_id):
        assert endpoint_id == "aiendpoint-was-starting"
        return {"status": {"state": "STOPPED"}}  # manually stopped out-of-band, stays stopped

    async def fake_start_endpoint(endpoint_id):
        assert endpoint_id == "aiendpoint-was-starting"

    async def fake_sleep(seconds):
        pass

    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)
    monkeypatch.setattr(endpoints_client, "start_endpoint", fake_start_endpoint)
    monkeypatch.setattr(worker_manager.asyncio, "sleep", fake_sleep)

    # Must not raise WorkerBusyError — falls through to attempt starting
    # the (now-stopped) existing endpoint instead of being permanently
    # blocked by the stale claim. The endpoint staying STOPPED in this
    # mock means it never actually reaches RUNNING either, so the only
    # thing under test here is which exception (if any) comes out first.
    with pytest.raises(worker_manager.WorkerProvisionError):
        await worker_manager.ensure_worker("cpu")


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
        if endpoint_id == "aiendpoint-deleted":
            # Confirms genuinely gone (not just a slow start) before the
            # self-heal-to-create path is allowed to fire.
            raise endpoints_client.NebiusEndpointError("nebius ai endpoint get failed (exit 1): not found")
        assert endpoint_id == "aiendpoint-fresh"
        return {"status": {"state": "RUNNING", "public_endpoints": ["https://fresh.tunnel.nebius.cloud"]}}

    monkeypatch.setattr(endpoints_client, "start_endpoint", fake_start_endpoint)
    monkeypatch.setattr(endpoints_client, "create_endpoint", fake_create_endpoint)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)

    worker = await worker_manager.ensure_worker("cpu")

    assert worker["worker_status"] == WorkerStatus.READY
    assert worker["nebius_endpoint_id"] == "aiendpoint-fresh"
    assert worker["endpoint_url"] == "https://fresh.tunnel.nebius.cloud"


async def test_ensure_worker_keeps_waiting_when_start_times_out_but_endpoint_still_exists(temp_db, monkeypatch):
    """Regression test for the 2026-07-12 GPU incident: a start command
    timing out does NOT mean the endpoint was deleted — a real GPU cold
    start can outlast the client-side wait. Must check live status before
    assuming deletion, and keep waiting on the same endpoint instead of
    abandoning it and creating a wasteful duplicate."""
    await db.create_worker_session("worker-gpu", "gpu", "nebius_endpoint", 600)
    await db.update_worker_session(
        "worker-gpu", worker_status=WorkerStatus.STOPPED, nebius_endpoint_id="aiendpoint-slow",
    )

    async def fake_start_endpoint(endpoint_id):
        raise endpoints_client.NebiusEndpointError("nebius ai endpoint start --id aiendpoint-slow timed out")

    async def fail_if_called(**kwargs):
        raise AssertionError("should not create a duplicate when the endpoint is just slow, not deleted")

    async def fake_get_endpoint(endpoint_id):
        assert endpoint_id == "aiendpoint-slow"
        return {"status": {"state": "RUNNING", "public_endpoints": ["https://slow.tunnel.nebius.cloud"]}}

    monkeypatch.setattr(endpoints_client, "start_endpoint", fake_start_endpoint)
    monkeypatch.setattr(endpoints_client, "create_endpoint", fail_if_called)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)

    worker = await worker_manager.ensure_worker("cuda")

    assert worker["nebius_endpoint_id"] == "aiendpoint-slow"
    assert worker["worker_status"] == WorkerStatus.READY
    assert worker["endpoint_url"] == "https://slow.tunnel.nebius.cloud"


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


async def test_ensure_worker_retries_start_once_endpoint_settles_to_stopped(temp_db, monkeypatch):
    """Real incident, 2026-07-15: user manually stopped the CPU endpoint,
    then started a new run seconds later. The initial start_endpoint()
    call landed while it was still mid-STOPPING and got explicitly
    rejected by Nebius (not a timeout) — the old code just kept passively
    polling get_endpoint() for RUNNING, which could never happen since no
    start was ever actually accepted, until the full budget expired. Must
    retry the start command once the poll loop observes the endpoint has
    settled into STOPPED, not only ever attempt it the one time before
    entering the wait loop. See docs/DESIGN_DECISIONS.md."""
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.STOPPED, nebius_endpoint_id="aiendpoint-was-stopping",
    )
    monkeypatch.setattr(worker_manager.settings, "nebius_endpoint_ready_timeout_seconds", 10)
    monkeypatch.setattr(worker_manager.settings, "nebius_endpoint_poll_interval_seconds", 1)

    start_calls = {"count": 0}

    async def fake_start_endpoint(endpoint_id):
        start_calls["count"] += 1
        if start_calls["count"] == 1:
            raise endpoints_client.NebiusEndpointError(
                "nebius ai endpoint start failed (exit 5): rpc error: code = Internal desc = internal error"
            )
        # Second call (the retry) succeeds — mid-STOPPING has since settled.

    get_calls = {"count": 0}

    async def fake_get_endpoint(endpoint_id):
        get_calls["count"] += 1
        if get_calls["count"] == 1:
            # The fallback check right after the first start_endpoint()
            # failure — endpoint still exists, just not up yet.
            return {"status": {"state": "STOPPING"}}
        if get_calls["count"] == 2:
            # First poll-loop iteration — settled to STOPPED, should
            # trigger exactly one retry of start_endpoint().
            return {"status": {"state": "STOPPED"}}
        return {"status": {"state": "RUNNING", "public_endpoints": ["https://recovered.tunnel.nebius.cloud"]}}

    async def fake_sleep(seconds):
        pass

    monkeypatch.setattr(endpoints_client, "start_endpoint", fake_start_endpoint)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)
    monkeypatch.setattr(worker_manager.asyncio, "sleep", fake_sleep)

    worker = await worker_manager.ensure_worker("cpu")

    assert start_calls["count"] == 2  # initial (rejected) + exactly one retry
    assert worker["worker_status"] == WorkerStatus.READY
    assert worker["endpoint_url"] == "https://recovered.tunnel.nebius.cloud"
