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
    """Part 3 / D3: create_new_worker() now checks for an endpoint in any state
    using find_endpoint_any_state before creating fresh — default every test
    to "none found" (matching behavior before adoption existed) so each test
    doesn't need its own boilerplate mock for them. Tests that actually
    exercise adoption override this themselves."""
    async def fake_find_endpoint_any_state(name):
        return None
    monkeypatch.setattr(endpoints_client, "find_endpoint_any_state", fake_find_endpoint_any_state)


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


async def test_ensure_worker_settles_then_starts_when_endpoint_shutting_down(temp_db, monkeypatch):
    """Contract updated with the decision-table rework: DB SHUTTING_DOWN is
    now just the reconciler's cache of Nebius STOPPING — live state decides.
    The old behavior ("skip straight to create") paid for a duplicate
    endpoint; now a STOPPING endpoint is adopted, the poll loop waits for it
    to settle to STOPPED, and only then issues the start. No create, and no
    blind start into a still-stopping endpoint (the §79a overlap hazard)."""
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.SHUTTING_DOWN, nebius_endpoint_id="aiendpoint-dying",
    )
    monkeypatch.setattr(worker_manager.settings, "nebius_endpoint_poll_interval_seconds", 0.01)
    monkeypatch.setattr(worker_manager.settings, "nebius_endpoint_ready_timeout_seconds", 1)

    calls = {"gets": 0, "starts": 0}

    async def fake_start_endpoint(endpoint_id):
        assert endpoint_id == "aiendpoint-dying"
        calls["starts"] += 1

    async def fail_if_created(**kwargs):
        raise AssertionError("should adopt and restart the settling endpoint, not create a duplicate")

    async def fake_get_endpoint(endpoint_id):
        calls["gets"] += 1
        base = {"spec": {"platform": "cpu-d3", "preset": "4vcpu-16gb"}}
        if calls["gets"] <= 2:
            # Still settling — no start may be issued while STOPPING.
            assert calls["starts"] == 0, "start_endpoint fired while endpoint was still STOPPING"
            return {**base, "status": {"state": "STOPPING"}}
        if calls["starts"] == 0:
            return {**base, "status": {"state": "STOPPED"}}
        return {**base, "status": {"state": "RUNNING", "public_endpoints": ["https://revived.tunnel.nebius.cloud"]}}

    monkeypatch.setattr(endpoints_client, "start_endpoint", fake_start_endpoint)
    monkeypatch.setattr(endpoints_client, "create_endpoint", fail_if_created)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)

    worker = await worker_manager.ensure_worker("cpu")

    assert calls["starts"] == 1
    assert worker["nebius_endpoint_id"] == "aiendpoint-dying"
    assert worker["worker_status"] == WorkerStatus.READY
    assert worker["endpoint_url"] == "https://revived.tunnel.nebius.cloud"


async def test_ensure_worker_adopts_existing_live_endpoint_instead_of_duplicating(temp_db, monkeypatch):
    """Regression test for the 2026-07-12 incident: a CPU endpoint created
    out-of-band (e.g. via scripts/create_nebius_endpoint.py before it wrote
    to the DB) was already RUNNING, but the app's DB had no row for it —
    ensure_worker created a second, duplicate CPU endpoint. Must adopt the
    live one by name instead. Part 3 / D3 now searches any state, not just RUNNING."""
    async def fake_find_endpoint_any_state(name):
        assert name == "llm-lab-cpu-trainer"
        return {"metadata": {"id": "aiendpoint-adopted"}, "status": {"state": "RUNNING"}}

    async def fail_if_called(**kwargs):
        raise AssertionError("should not create a duplicate when a live RUNNING endpoint already exists")

    async def fake_get_endpoint(endpoint_id):
        assert endpoint_id == "aiendpoint-adopted"
        return {
            "spec": {"platform": "cpu-d3", "preset": "8vcpu-32gb"},
            "status": {"state": "RUNNING", "public_endpoints": ["https://adopted.tunnel.nebius.cloud"]},
        }

    monkeypatch.setattr(endpoints_client, "find_endpoint_any_state", fake_find_endpoint_any_state)
    monkeypatch.setattr(endpoints_client, "create_endpoint", fail_if_called)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)

    worker = await worker_manager.ensure_worker("cpu")

    assert worker["nebius_endpoint_id"] == "aiendpoint-adopted"
    assert worker["worker_status"] == WorkerStatus.READY
    assert worker["endpoint_url"] == "https://adopted.tunnel.nebius.cloud"


async def test_create_new_worker_restarts_stopped_endpoint_instead_of_creating(temp_db, monkeypatch):
    """A stopped endpoint the app's DB doesn't know about (e.g. the local
    worker_sessions row was lost, or it was created out-of-band) should be
    adopted and restarted, not abandoned — this was the 2026-07-12 GPU incident:
    a stopped endpoint sat idle while a second one got created. Part 3 / D3 now
    uses find_endpoint_any_state which finds it in any state."""
    async def fake_find_endpoint_any_state(name):
        assert name == "llm-lab-gpu-trainer"
        return {"metadata": {"id": "aiendpoint-restarted"}, "status": {"state": "STOPPED"}}

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

    monkeypatch.setattr(endpoints_client, "find_endpoint_any_state", fake_find_endpoint_any_state)
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

    # Stateful fake: STOPPED until start_endpoint is called, RUNNING after —
    # keeps this exercising the restart path under the decision-table flow
    # (a fake that's RUNNING from the outset would take the adopt-in-place
    # branch and never restart anything).
    started = {"called": False}

    async def fake_start_endpoint(endpoint_id):
        assert endpoint_id == "aiendpoint-old"
        started["called"] = True

    async def fake_get_endpoint(endpoint_id):
        if not started["called"]:
            return {"spec": {"platform": "cpu-d3", "preset": "4vcpu-16gb"}, "status": {"state": "STOPPED"}}
        return {
            "spec": {"platform": "cpu-d3", "preset": "4vcpu-16gb"},
            "status": {"state": "RUNNING", "public_endpoints": ["https://fresh.tunnel.nebius.cloud"]},
        }

    async def fail_if_created(**kwargs):
        raise AssertionError("must restart the stopped endpoint, not create a new one")

    monkeypatch.setattr(endpoints_client, "start_endpoint", fake_start_endpoint)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)
    monkeypatch.setattr(endpoints_client, "create_endpoint", fail_if_created)
    monkeypatch.setattr(worker_manager.settings, "nebius_endpoint_poll_interval_seconds", 0.01)
    monkeypatch.setattr(worker_manager.settings, "nebius_endpoint_ready_timeout_seconds", 1)

    await worker_manager.ensure_worker("cpu")

    assert started["called"]

    session = await db.get_worker_session("worker-cpu")
    from backend.nebius.idle_monitor import seconds_since
    assert seconds_since(session["last_activity_at"]) < 10


async def test_ensure_worker_debounces_when_already_provisioning(temp_db, monkeypatch):
    """Real incident, 2026-07-15: ensure_worker checked worker_status blindly.
    If DB says PROVISIONING but Nebius endpoint is actually gone (manual stop) or
    in a contradicting state, we'd raise WorkerBusyError forever. Must verify
    against live state — only treat as busy if endpoint is genuinely mid-flight
    (STARTING or RUNNING). See docs/DESIGN_DECISIONS.md §79a."""
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.PROVISIONING, nebius_endpoint_id="aiendpoint-existing",
    )

    async def fail_if_called(**kwargs):
        raise AssertionError("should not create or start a second worker while one is already mid-provision")

    async def fake_get_endpoint(endpoint_id):
        # Nebius agrees it's genuinely still mid-flight — the busy claim is real.
        # STARTING is the actual Nebius state for a provisioning endpoint.
        return {"status": {"state": "STARTING"}}

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
    """Regression test: a stopped endpoint should be restarted, not abandoned.
    If DB says STOPPED but Nebius still has the endpoint (even if dead for a moment),
    start_endpoint should be called to restart it."""
    await db.create_worker_session("worker-gpu", "gpu", "nebius_endpoint", 600)
    await db.update_worker_session(
        "worker-gpu", worker_status=WorkerStatus.STOPPED, nebius_endpoint_id="aiendpoint-existing",
        endpoint_url="https://existing.tunnel.nebius.cloud",
    )

    # Stateful fake: the endpoint stays STOPPED until start_endpoint is
    # actually issued, then reports RUNNING. (The old fake reported RUNNING
    # from the outset — under the decision-table flow that's the adopt-in-
    # place case, where NOT issuing a start command is the whole point.)
    started = {}

    async def fake_start_endpoint(endpoint_id):
        started["endpoint_id"] = endpoint_id

    async def fake_get_endpoint(endpoint_id):
        if "endpoint_id" not in started:
            return {"spec": {"platform": "gpu-l40s-a", "preset": "1gpu-8vcpu-32gb"}, "status": {"state": "STOPPED"}}
        return {
            "spec": {"platform": "gpu-l40s-a", "preset": "1gpu-8vcpu-32gb"},
            "status": {"state": "RUNNING", "public_endpoints": ["https://restarted.tunnel.nebius.cloud"]}
        }

    async def fake_probe(url):
        return True  # Tunnel is reachable

    async def fail_if_called(**kwargs):
        raise AssertionError("should not create a new endpoint when one already exists")

    monkeypatch.setattr(endpoints_client, "create_endpoint", fail_if_called)
    monkeypatch.setattr(endpoints_client, "start_endpoint", fake_start_endpoint)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)
    monkeypatch.setattr(endpoints_client, "probe_endpoint_url", fake_probe)
    monkeypatch.setattr(worker_manager.settings, "nebius_endpoint_poll_interval_seconds", 0.01)
    monkeypatch.setattr(worker_manager.settings, "nebius_endpoint_ready_timeout_seconds", 1)

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
        endpoint_url="https://slow.tunnel.nebius.cloud",
    )

    async def fake_start_endpoint(endpoint_id):
        raise endpoints_client.NebiusEndpointError("nebius ai endpoint start --id aiendpoint-slow timed out")

    async def fail_if_called(**kwargs):
        raise AssertionError("should not create a duplicate when the endpoint is just slow, not deleted")

    async def fake_get_endpoint(endpoint_id):
        assert endpoint_id == "aiendpoint-slow"
        return {
            "spec": {"platform": "gpu-l40s-a", "preset": "1gpu-8vcpu-32gb"},
            "status": {"state": "RUNNING", "public_endpoints": ["https://slow.tunnel.nebius.cloud"]}
        }

    async def fake_probe(url):
        return True

    monkeypatch.setattr(endpoints_client, "start_endpoint", fake_start_endpoint)
    monkeypatch.setattr(endpoints_client, "create_endpoint", fail_if_called)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)
    monkeypatch.setattr(endpoints_client, "probe_endpoint_url", fake_probe)

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

    # New-contract sequence: while the endpoint is mid-STOPPING the decision
    # table deliberately issues NO start at all (the old code fired blindly
    # into STOPPING and got rejected — one fewer overlapping op now). The
    # poll loop then observes settled STOPPED and issues the start; if that
    # is rejected (still-settling race), the NEXT STOPPED observation
    # retries. So: STOPPING → STOPPED (start #1 rejected) → STOPPED
    # (start #2 accepted) → RUNNING.
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
        base = {"metadata": {"id": endpoint_id}, "spec": {"platform": "cpu-d3", "preset": "4vcpu-16gb"}}
        if get_calls["count"] == 1:
            # Decision-table live check — still mid-STOPPING, no start issued.
            return {**base, "status": {"state": "STOPPING"}}
        if start_calls["count"] < 2:
            # Settled to STOPPED; stays STOPPED until a start is ACCEPTED.
            return {**base, "status": {"state": "STOPPED"}}
        return {
            **base,
            "status": {"state": "RUNNING", "public_endpoints": ["https://recovered.tunnel.nebius.cloud"]}
        }

    async def fake_sleep(seconds):
        pass

    async def fake_probe(url):
        return True

    monkeypatch.setattr(endpoints_client, "start_endpoint", fake_start_endpoint)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)
    monkeypatch.setattr(endpoints_client, "probe_endpoint_url", fake_probe)
    monkeypatch.setattr(worker_manager.asyncio, "sleep", fake_sleep)

    worker = await worker_manager.ensure_worker("cpu")

    assert start_calls["count"] == 2  # first settle-start (rejected) + exactly one retry
    assert worker["worker_status"] == WorkerStatus.READY
    assert worker["endpoint_url"] == "https://recovered.tunnel.nebius.cloud"


# --- T1: ERROR endpoint deleted not restarted (Part 2 / D2) ---
async def test_error_endpoint_deleted_not_restarted(temp_db, monkeypatch):
    """Part 2 / D2: ERROR endpoints can only be recovered by deletion, not restart.
    When ensure_worker encounters an ERROR state endpoint, it must delete and recreate,
    never attempt start_endpoint on it."""
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.READY,
        nebius_endpoint_id="aiendpoint-error", endpoint_url="https://error.tunnel.nebius.cloud",
    )

    deleted_ids = []
    created_count = 0

    async def fake_get_endpoint(endpoint_id):
        if endpoint_id == "aiendpoint-error":
            return {"status": {"state": "ERROR"}}
        elif endpoint_id == "aiendpoint-fresh":
            return {
                "spec": {"platform": "cpu-d3", "preset": "4vcpu-16gb"},
                "status": {"state": "RUNNING", "public_endpoints": ["https://fresh.tunnel.nebius.cloud"]},
            }
        raise endpoints_client.NebiusEndpointError(f"unknown endpoint {endpoint_id}")

    async def fake_delete_endpoint(endpoint_id):
        deleted_ids.append(endpoint_id)

    async def fake_start_endpoint(endpoint_id):
        raise endpoints_client.NebiusEndpointError("should not start ERROR endpoint")

    async def fake_create_endpoint(**kwargs):
        nonlocal created_count
        created_count += 1
        return "aiendpoint-fresh"

    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)
    monkeypatch.setattr(endpoints_client, "delete_endpoint", fake_delete_endpoint)
    monkeypatch.setattr(endpoints_client, "start_endpoint", fake_start_endpoint)
    monkeypatch.setattr(endpoints_client, "create_endpoint", fake_create_endpoint)

    worker = await worker_manager.ensure_worker("cpu")

    assert "aiendpoint-error" in deleted_ids
    assert created_count == 1
    assert worker["nebius_endpoint_id"] == "aiendpoint-fresh"


# --- T2: Adoption endpoint in any state (Part 3 / D3) ---
async def test_adoption_endpoint_in_any_state(temp_db, monkeypatch):
    """Part 3 / D3: create_new_worker must adopt endpoints in any transient state,
    not just RUNNING or STOPPED. A console-started or console-stopped endpoint should
    never be duplicated."""
    async def fake_find_endpoint_any_state(name):
        # Return endpoint in STARTING state (mid-provision)
        return {"metadata": {"id": "aiendpoint-starting"}, "status": {"state": "STARTING"}}

    async def fail_if_create_called(**kwargs):
        raise AssertionError("should not create when endpoint exists in STARTING state")

    async def fake_get_endpoint(endpoint_id):
        if endpoint_id == "aiendpoint-starting":
            return {
                "spec": {"platform": "cpu-d3", "preset": "4vcpu-16gb"},
                "status": {"state": "RUNNING", "public_endpoints": ["https://adopted.tunnel.nebius.cloud"]},
            }
        raise endpoints_client.NebiusEndpointError(f"unknown {endpoint_id}")

    monkeypatch.setattr(endpoints_client, "find_endpoint_any_state", fake_find_endpoint_any_state)
    monkeypatch.setattr(endpoints_client, "create_endpoint", fail_if_create_called)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)

    worker = await worker_manager.ensure_worker("cpu")

    assert worker["nebius_endpoint_id"] == "aiendpoint-starting"
    assert worker["worker_status"] == WorkerStatus.READY


# --- T3: Dead tunnel stops before recreate (Part 2 / D4) ---
async def test_dead_tunnel_stops_before_recreate(temp_db, monkeypatch):
    """Part 2 / D4: When an endpoint's tunnel is dead (RUNNING but unreachable),
    stop it first before creating a new one. This prevents orphaned endpoints burning money."""
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.READY,
        nebius_endpoint_id="aiendpoint-dead", endpoint_url="https://dead.tunnel.nebius.cloud",
    )

    stopped_ids = []
    created_count = 0

    async def fake_get_endpoint(endpoint_id):
        if endpoint_id == "aiendpoint-dead":
            return {"status": {"state": "RUNNING"}}
        elif endpoint_id == "aiendpoint-fresh":
            return {
                "spec": {"platform": "cpu-d3", "preset": "4vcpu-16gb"},
                "status": {"state": "RUNNING", "public_endpoints": ["https://fresh.tunnel.nebius.cloud"]},
            }
        raise endpoints_client.NebiusEndpointError(f"unknown {endpoint_id}")

    async def fake_probe(url):
        if url == "https://dead.tunnel.nebius.cloud":
            return False  # Dead tunnel
        return True

    async def fake_stop_endpoint(endpoint_id):
        stopped_ids.append(endpoint_id)

    async def fake_create_endpoint(**kwargs):
        nonlocal created_count
        created_count += 1
        return "aiendpoint-fresh"

    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)
    monkeypatch.setattr(endpoints_client, "probe_endpoint_url", fake_probe)
    monkeypatch.setattr(endpoints_client, "stop_endpoint", fake_stop_endpoint)
    monkeypatch.setattr(endpoints_client, "create_endpoint", fake_create_endpoint)

    worker = await worker_manager.ensure_worker("cpu")

    assert "aiendpoint-dead" in stopped_ids
    assert created_count == 1
    assert worker["nebius_endpoint_id"] == "aiendpoint-fresh"


# --- T4: Start retries bounded (Part 7 / D7) ---
async def test_start_retries_bounded(temp_db, monkeypatch):
    """Part 7 / D7: Poll loop retries start_endpoint up to max_retries attempts
    (default 3) when STOPPED is observed, not just once forever."""
    monkeypatch.setattr(worker_manager.settings, "nebius_endpoint_start_max_retries", 2)
    monkeypatch.setattr(worker_manager.settings, "nebius_endpoint_poll_interval_seconds", 0.01)
    # Without this the loop runs ready_timeout/0.01 = tens of thousands of
    # iterations — this single missing patch once cost the suite 370s.
    monkeypatch.setattr(worker_manager.settings, "nebius_endpoint_ready_timeout_seconds", 1)

    start_attempts = 0

    async def fake_create_endpoint(**kwargs):
        return "aiendpoint-test"

    async def fake_get_endpoint(endpoint_id):
        nonlocal start_attempts
        # Always return STOPPED so retry logic keeps triggering
        return {"status": {"state": "STOPPED"}}

    async def fake_start_endpoint(endpoint_id):
        nonlocal start_attempts
        start_attempts += 1
        # Fail each time to force retries
        raise endpoints_client.NebiusEndpointError("start rejected")

    monkeypatch.setattr(endpoints_client, "create_endpoint", fake_create_endpoint)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)
    monkeypatch.setattr(endpoints_client, "start_endpoint", fake_start_endpoint)

    with pytest.raises(worker_manager.WorkerProvisionError):
        await worker_manager.ensure_worker("cpu")

    # Should be max_retries attempts (2) within poll loop, not unlimited retries
    assert start_attempts <= 2


async def test_ensure_worker_adopts_running_endpoint_with_unknown_reachability_in_place(temp_db, monkeypatch):
    """Regression pin for a review-caught bug: a session with no stored
    endpoint_url has reachable=None (unknown — the probe never ran), and the
    dead-tunnel branch used `not reachable`, so None was treated like a
    CONFIRMED-dead tunnel: the healthy RUNNING endpoint was stopped and a
    duplicate created. Unknown reachability must adopt in place — no stop,
    no create, no start — and let the poll loop record the URL."""
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session(
        "worker-cpu", worker_status=WorkerStatus.STOPPED, nebius_endpoint_id="aiendpoint-healthy",
        # deliberately NO endpoint_url — reachability cannot be probed
    )
    monkeypatch.setattr(worker_manager.settings, "nebius_endpoint_poll_interval_seconds", 0.01)
    monkeypatch.setattr(worker_manager.settings, "nebius_endpoint_ready_timeout_seconds", 1)

    async def fake_get_endpoint(endpoint_id):
        return {
            "spec": {"platform": "cpu-d3", "preset": "4vcpu-16gb"},
            "status": {"state": "RUNNING", "public_endpoints": ["https://healthy.tunnel.nebius.cloud"]},
        }

    async def fail_stop(endpoint_id):
        raise AssertionError("must not stop a RUNNING endpoint whose reachability is merely unknown")

    async def fail_create(**kwargs):
        raise AssertionError("must not create a duplicate for a healthy RUNNING endpoint")

    async def fail_start(endpoint_id):
        raise AssertionError("must not issue start on an already-RUNNING endpoint (§79a overlap)")

    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)
    monkeypatch.setattr(endpoints_client, "stop_endpoint", fail_stop)
    monkeypatch.setattr(endpoints_client, "create_endpoint", fail_create)
    monkeypatch.setattr(endpoints_client, "start_endpoint", fail_start)

    worker = await worker_manager.ensure_worker("cpu")

    assert worker["worker_status"] == WorkerStatus.READY
    assert worker["nebius_endpoint_id"] == "aiendpoint-healthy"
    assert worker["endpoint_url"] == "https://healthy.tunnel.nebius.cloud"
