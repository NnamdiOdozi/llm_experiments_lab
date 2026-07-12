"""Remote (nebius_endpoint) proxying in backend/api/training.py.

The backend to use is chosen per-request (StartRunRequest.backend), not by
the app-wide training_backend setting — users pick local vs serverless per
run from the frontend, the same way they already pick device (2026-07-11
session). Local-mode behavior is covered implicitly by not passing
backend="nebius_endpoint" — these tests only cover the remote branch.
"""

import asyncio
import json

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from backend import db
from backend.api import training as training_module
from backend.main import app
from backend.nebius import worker_manager


class FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://fake.example")
            raise httpx.HTTPStatusError("error", request=request, response=self)

    def json(self):
        return self._json_data


class FakeAsyncClient:
    """Replaces httpx.AsyncClient — returns queued responses in call order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def request(self, method, url, json=None):
        self.calls.append((method, url, json))
        return self.responses.pop(0)


@pytest.fixture
async def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_lab.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    await db.init_db()
    exp_id = await db.create_experiment(
        "Test experiment", {"template": "transformer"}, preset_key="tiny-shakespeare",
    )
    return exp_id


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _fake_worker():
    return {
        "session_id": "worker-cpu",
        "nebius_endpoint_id": "aiendpoint-abc123",
        "endpoint_url": "https://cpu.tunnel.nebius.cloud",
    }


async def test_start_training_proxies_to_remote_endpoint(temp_db, client, monkeypatch):
    exp_id = temp_db

    async def fake_ensure_worker(device):
        return _fake_worker()

    monkeypatch.setattr(worker_manager, "ensure_worker", fake_ensure_worker)

    fake_client = FakeAsyncClient([
        FakeResponse({"id": 99}),  # POST /api/experiments (mirror)
        FakeResponse({"run_id": 7, "status": "queued"}),  # POST /api/training/start
    ])
    monkeypatch.setattr(training_module.httpx, "AsyncClient", lambda timeout=30: fake_client)

    # _start_remote_run is backgrounded (see training.py), not awaited
    # inside the request — capture the task at creation time (not via
    # _provisioning_tasks, which the task's own done-callback empties as
    # soon as it finishes, racy to read after the fact) so the test can
    # deterministically wait for it before asserting the DB's final state.
    created_tasks = []
    orig_create_task = training_module.asyncio.create_task

    def capturing_create_task(coro):
        task = orig_create_task(coro)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(training_module.asyncio, "create_task", capturing_create_task)

    resp = await client.post(
        "/api/training/start",
        json={"experiment_id": exp_id, "device": "cpu", "backend": "nebius_endpoint"},
    )

    assert resp.status_code == 200
    run_id = resp.json()["run_id"]
    await created_tasks[0]

    db_run = await db.get_training_run(run_id)
    assert db_run["execution_backend"] == "nebius_endpoint"
    assert db_run["remote_endpoint_id"] == "aiendpoint-abc123"
    assert db_run["remote_run_id"] == 7
    # Regression guard (2026-07-12): _start_remote_run used to leave the
    # local status column frozen at its QUEUED creation default forever
    # after a successful handoff — see docs/DESIGN_DECISIONS.md.
    assert db_run["status"] == "running"

    # Second call (training/start) must target the remote experiment id, not the local one
    assert fake_client.calls[1][1] == "https://cpu.tunnel.nebius.cloud/api/training/start"
    assert fake_client.calls[1][2]["experiment_id"] == 99


async def test_start_training_marks_run_failed_when_worker_unavailable(temp_db, client, monkeypatch):
    exp_id = temp_db

    async def fake_ensure_worker(device):
        raise worker_manager.WorkerProvisionError("endpoint stuck in PROVISIONING")

    monkeypatch.setattr(worker_manager, "ensure_worker", fake_ensure_worker)

    # Provisioning is backgrounded (see training.py) — /start itself always
    # returns 200 with the run_id immediately now; a provisioning failure
    # only shows up in the run's own DB status once the background task
    # finishes, not as this request's HTTP status.
    created_tasks = []
    orig_create_task = training_module.asyncio.create_task

    def capturing_create_task(coro):
        task = orig_create_task(coro)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(training_module.asyncio, "create_task", capturing_create_task)

    resp = await client.post(
        "/api/training/start",
        json={"experiment_id": exp_id, "device": "cpu", "backend": "nebius_endpoint"},
    )

    assert resp.status_code == 200
    run_id = resp.json()["run_id"]
    await created_tasks[0]

    db_run = await db.get_training_run(run_id)
    assert db_run["status"] == "failed"


async def test_stop_training_cancels_in_flight_provisioning(temp_db, client, monkeypatch):
    """Part F: Stop must be able to interrupt provisioning itself, not just
    an already-running remote run — before this, execution_backend wasn't
    set until provisioning finished, so stop_training's _is_remote() check
    was False the whole time and stop_run() (local-only) reported "not
    found" for a still-provisioning remote run."""
    exp_id = temp_db
    worker_call_started = asyncio.Event()

    async def fake_ensure_worker(device):
        worker_call_started.set()
        await asyncio.Event().wait()  # never resolves on its own — only via cancellation

    monkeypatch.setattr(worker_manager, "ensure_worker", fake_ensure_worker)

    created_tasks = []
    orig_create_task = training_module.asyncio.create_task

    def capturing_create_task(coro):
        task = orig_create_task(coro)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(training_module.asyncio, "create_task", capturing_create_task)

    resp = await client.post(
        "/api/training/start",
        json={"experiment_id": exp_id, "device": "cpu", "backend": "nebius_endpoint"},
    )
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    await asyncio.wait_for(worker_call_started.wait(), timeout=2)

    stop_resp = await client.post(f"/api/training/{run_id}/stop")
    assert stop_resp.status_code == 200

    await asyncio.gather(created_tasks[0], return_exceptions=True)

    db_run = await db.get_training_run(run_id)
    assert db_run["status"] == "cancelled"


async def test_execution_backend_is_correct_while_still_provisioning(temp_db, client, monkeypatch):
    """Regression test (2026-07-12): execution_backend used to only get set
    to 'nebius_endpoint' near the END of _start_remote_run (after mirroring
    the experiment remotely) — invisible before Part F's backgrounding
    change, since /start blocked until that write happened. Once /start
    started returning immediately, the frontend could poll status during
    the ~6min provisioning window and see the DB's 'local' schema default
    instead — a serverless run showing "local" on the Experiments page.
    execution_backend must be correct from the moment the row is created."""
    exp_id = temp_db
    worker_call_started = asyncio.Event()

    async def fake_ensure_worker(device):
        worker_call_started.set()
        await asyncio.Event().wait()  # still "provisioning" — never resolves on its own

    monkeypatch.setattr(worker_manager, "ensure_worker", fake_ensure_worker)

    resp = await client.post(
        "/api/training/start",
        json={"experiment_id": exp_id, "device": "cpu", "backend": "nebius_endpoint"},
    )
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    await asyncio.wait_for(worker_call_started.wait(), timeout=2)

    # Still mid-provisioning at this point — the whole point of the test.
    db_run = await db.get_training_run(run_id)
    assert db_run["execution_backend"] == "nebius_endpoint"

    # Cleanup: cancel the still-running task so it doesn't leak into other tests.
    task = training_module._provisioning_tasks.get(run_id)
    if task is not None:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_start_training_defaults_to_local_when_backend_omitted(temp_db, client, monkeypatch):
    async def fail_if_called(device):
        raise AssertionError("should not touch the remote worker when backend is omitted")

    monkeypatch.setattr(worker_manager, "ensure_worker", fail_if_called)

    resp = await client.post("/api/training/start", json={"experiment_id": temp_db, "device": "cpu"})

    assert resp.status_code == 200
    run_id = resp.json()["run_id"]
    db_run = await db.get_training_run(run_id)
    assert db_run["execution_backend"] == "local"


async def test_start_training_stays_local_even_if_global_setting_is_remote(temp_db, client, monkeypatch):
    """Per-request choice must win over whatever the global setting is —
    that setting is only the frontend's initial suggestion. See
    docs/DESIGN_DECISIONS.md §10/§11."""
    monkeypatch.setattr(training_module.settings, "training_backend", "nebius_endpoint")

    async def fail_if_called(device):
        raise AssertionError("should not touch the remote worker when backend=local is explicit")

    monkeypatch.setattr(worker_manager, "ensure_worker", fail_if_called)

    resp = await client.post(
        "/api/training/start",
        json={"experiment_id": temp_db, "device": "cpu", "backend": "local"},
    )

    assert resp.status_code == 200
    run_id = resp.json()["run_id"]
    db_run = await db.get_training_run(run_id)
    assert db_run["execution_backend"] == "local"


async def test_pause_training_proxies_to_remote_run(temp_db, client, monkeypatch):
    run_id = await db.create_training_run(
        temp_db, "cpu", execution_backend="nebius_endpoint",
        remote_endpoint_id="aiendpoint-abc123", remote_run_id=7,
    )
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session("worker-cpu", endpoint_url="https://cpu.tunnel.nebius.cloud")

    fake_client = FakeAsyncClient([FakeResponse({"run_id": 7, "status": "pausing"})])
    monkeypatch.setattr(training_module.httpx, "AsyncClient", lambda timeout=30: fake_client)

    resp = await client.post(f"/api/training/{run_id}/pause")

    assert resp.status_code == 200
    assert resp.json()["run_id"] == run_id  # local id, not the remote 7
    assert fake_client.calls[0] == ("POST", "https://cpu.tunnel.nebius.cloud/api/training/7/pause", None)


async def test_pause_resume_and_prompt_touch_the_worker_idle_clock(temp_db, client, monkeypatch):
    """Regression test (2026-07-12): a user prompting a paused remote model
    for ~10 minutes (genuinely active engagement) saw an idle-timeout
    warning banner, because pause/resume/prompt never touched
    last_activity_at at all — only worker acquisition and the manual
    "Continue session" heartbeat did."""
    run_id = await db.create_training_run(
        temp_db, "cpu", execution_backend="nebius_endpoint",
        remote_endpoint_id="aiendpoint-abc123", remote_run_id=7,
    )
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session(
        "worker-cpu", endpoint_url="https://cpu.tunnel.nebius.cloud",
        last_activity_at="2020-01-01 00:00:00",
    )

    for path, response in [
        ("pause", FakeResponse({"run_id": 7, "status": "pausing"})),
        ("resume", FakeResponse({"run_id": 7, "status": "resuming"})),
        ("prompt", FakeResponse({"run_id": 7, "output": "hello"})),
    ]:
        await db.update_worker_session("worker-cpu", last_activity_at="2020-01-01 00:00:00")
        fake_client = FakeAsyncClient([response])
        monkeypatch.setattr(training_module.httpx, "AsyncClient", lambda timeout=30: fake_client)

        body = {"prompt": "hi", "max_new_tokens": 10} if path == "prompt" else None
        resp = await client.post(f"/api/training/{run_id}/{path}", json=body)

        assert resp.status_code == 200, f"{path} failed: {resp.text}"
        session = await db.get_worker_session("worker-cpu")
        assert session["last_activity_at"] != "2020-01-01 00:00:00", f"{path} did not touch the idle clock"


async def test_get_status_translates_remote_run_id_back_to_local(temp_db, client, monkeypatch):
    run_id = await db.create_training_run(
        temp_db, "cpu", execution_backend="nebius_endpoint",
        remote_endpoint_id="aiendpoint-abc123", remote_run_id=7,
    )
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session("worker-cpu", endpoint_url="https://cpu.tunnel.nebius.cloud")

    fake_client = FakeAsyncClient([
        # The endpoint's own status.json reports "local" from its own
        # perspective (it's a local subprocess to that container) — the
        # controller must override this, not trust it. Regression test for
        # the 2026-07-11 incident (see docs/DESIGN_DECISIONS.md §10).
        FakeResponse({"run_id": 7, "status": "running", "current_step": 20, "total_steps": 100, "execution_backend": "local"}),
    ])
    monkeypatch.setattr(training_module.httpx, "AsyncClient", lambda timeout=30: fake_client)

    resp = await client.get(f"/api/training/{run_id}/status")

    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == run_id
    assert body["status"] == "running"
    assert body["current_step"] == 20
    assert body["execution_backend"] == "nebius_endpoint"


async def test_get_status_syncs_local_row_and_touches_worker_on_terminal_transition(temp_db, client, monkeypatch):
    """Regression test (2026-07-12): local status only ever advanced via
    explicit local pause/resume/prompt actions — a run that finishes
    (completed/failed/cancelled) on the remote side on its own, without any
    of those, left the local row stale and never touched the idle clock.
    Not special-cased to "completed" — paused/cancelled/failed all matter."""
    run_id = await db.create_training_run(
        temp_db, "cpu", status="running", execution_backend="nebius_endpoint",
        remote_endpoint_id="aiendpoint-abc123", remote_run_id=7,
    )
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session(
        "worker-cpu", endpoint_url="https://cpu.tunnel.nebius.cloud",
        last_activity_at="2020-01-01 00:00:00",
    )

    fake_client = FakeAsyncClient([
        FakeResponse({"run_id": 7, "status": "completed", "current_step": 1000, "total_steps": 1000}),
    ])
    monkeypatch.setattr(training_module.httpx, "AsyncClient", lambda timeout=30: fake_client)

    resp = await client.get(f"/api/training/{run_id}/status")

    assert resp.status_code == 200
    db_run = await db.get_training_run(run_id)
    assert db_run["status"] == "completed"
    assert db_run["current_step"] == 1000
    session = await db.get_worker_session("worker-cpu")
    assert session["last_activity_at"] != "2020-01-01 00:00:00"


async def test_get_status_serves_local_queued_status_while_still_provisioning(temp_db, client, monkeypatch):
    """Regression test (2026-07-12): while _start_remote_run is still
    mirroring the run to the endpoint (can take several minutes),
    remote_run_id is None — proxying anyway builds a URL with "None" in it
    and fails every single poll, which the frontend can't distinguish from
    a genuine outage ("backend disconnected" for a run that's simply still
    starting up). Must serve the local QUEUED status instead, and must not
    touch the proxy client at all."""
    run_id = await db.create_training_run(
        temp_db, "cuda", execution_backend="nebius_endpoint",
        remote_endpoint_id="aiendpoint-abc123",
    )

    def fail_if_called(timeout=30):
        raise AssertionError("should not proxy while remote_run_id is still None")

    monkeypatch.setattr(training_module.httpx, "AsyncClient", fail_if_called)

    resp = await client.get(f"/api/training/{run_id}/status")

    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == run_id
    assert body["status"] == "queued"
    assert body["execution_backend"] == "nebius_endpoint"


async def test_get_metrics_returns_empty_list_while_still_provisioning(temp_db, client, monkeypatch):
    """Regression test (2026-07-12): fetchMetrics is called right after
    fetchRunStatus on every frontend poll — a 502 here alone was enough to
    trip the "backend disconnected" banner even after run_status() was
    fixed to report "queued" correctly, since the frontend treats either
    call failing as a disconnect. Must not proxy while remote_run_id is
    still None, and must not touch the network at all."""
    run_id = await db.create_training_run(
        temp_db, "cuda", execution_backend="nebius_endpoint",
        remote_endpoint_id="aiendpoint-abc123",
    )

    def fail_if_called(timeout=30):
        raise AssertionError("should not proxy while remote_run_id is still None")

    monkeypatch.setattr(training_module.httpx, "AsyncClient", fail_if_called)

    resp = await client.get(f"/api/training/{run_id}/metrics")

    assert resp.status_code == 200
    assert resp.json() == []


async def test_get_metrics_syncs_loss_history_to_local_row_for_remote_run(temp_db, client, monkeypatch):
    """Regression test (2026-07-12): train_loss_history on the local row
    never got written for a remote run — nothing else writes it, so the
    chatbot's loss-trend snapshot always saw '[]' no matter how far a
    Nebius run had actually progressed. get_metrics() must mirror the
    proxied metrics into the local row, same shape write_metric() uses for
    local runs."""
    run_id = await db.create_training_run(
        temp_db, "cuda", execution_backend="nebius_endpoint",
        remote_endpoint_id="aiendpoint-abc123", remote_run_id=7,
    )
    await db.create_worker_session("worker-gpu", "gpu", "nebius_endpoint", 600)
    await db.update_worker_session("worker-gpu", endpoint_url="https://gpu.tunnel.nebius.cloud")

    fake_client = FakeAsyncClient([
        FakeResponse([
            {"step": 20, "train_loss": 1.8, "val_loss": 1.9},
            {"step": 40, "train_loss": 1.5, "val_loss": 1.6},
        ]),
    ])
    monkeypatch.setattr(training_module.httpx, "AsyncClient", lambda timeout=30: fake_client)

    resp = await client.get(f"/api/training/{run_id}/metrics")

    assert resp.status_code == 200
    assert len(resp.json()) == 2
    db_run = await db.get_training_run(run_id)
    train_history = json.loads(db_run["train_loss_history"])
    assert len(train_history) == 2
    assert train_history[-1]["train_loss"] == 1.5


async def test_list_open_runs_overlays_live_status_for_remote_run(temp_db, client, monkeypatch):
    """Regression test (2026-07-12): _start_remote_run never updates local
    status past its QUEUED creation default after handoff — a genuinely
    running remote run showed QUEUED/step 0 forever in Open Runs. Must
    overlay live status/step from the remote endpoint for display."""
    run_id = await db.create_training_run(
        temp_db, "cpu", execution_backend="nebius_endpoint",
        remote_endpoint_id="aiendpoint-abc123", remote_run_id=7,
    )
    # Local row frozen at its creation default, as _start_remote_run leaves
    # it pre-fix (this test targets list_open_runs' own overlay, not the
    # separate _start_remote_run status=RUNNING fix).
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session("worker-cpu", endpoint_url="https://cpu.tunnel.nebius.cloud")

    fake_client = FakeAsyncClient([
        FakeResponse({"run_id": 7, "status": "running", "current_step": 370, "total_steps": 1000}),
    ])
    monkeypatch.setattr(training_module.httpx, "AsyncClient", lambda timeout=30: fake_client)

    resp = await client.get("/api/training/open")

    assert resp.status_code == 200
    runs = {r["id"]: r for r in resp.json()}
    assert runs[run_id]["status"] == "running"
    assert runs[run_id]["current_step"] == 370
    assert runs[run_id]["total_steps"] == 1000


async def test_list_open_runs_falls_back_to_local_status_when_proxy_fails(temp_db, client, monkeypatch):
    run_id = await db.create_training_run(
        temp_db, "cpu", execution_backend="nebius_endpoint",
        remote_endpoint_id="aiendpoint-abc123", remote_run_id=7,
    )
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session("worker-cpu", endpoint_url="https://cpu.tunnel.nebius.cloud")

    fake_client = FakeAsyncClient([FakeResponse({}, status_code=502)])
    monkeypatch.setattr(training_module.httpx, "AsyncClient", lambda timeout=30: fake_client)

    resp = await client.get("/api/training/open")

    assert resp.status_code == 200
    runs = {r["id"]: r for r in resp.json()}
    assert runs[run_id]["status"] == "queued"  # stale local value, not a crash


async def test_list_open_runs_excludes_remote_run_that_is_actually_terminal_live(temp_db, client, monkeypatch):
    """The local status column being non-terminal doesn't mean the run
    actually still is — db.list_open_runs()'s own terminal-status filter
    only sees that stale local value, so a remote run that's genuinely
    completed must be filtered out here too, after the live overlay."""
    run_id = await db.create_training_run(
        temp_db, "cpu", execution_backend="nebius_endpoint",
        remote_endpoint_id="aiendpoint-abc123", remote_run_id=7,
    )
    await db.create_worker_session("worker-cpu", "cpu", "nebius_endpoint", 1800)
    await db.update_worker_session("worker-cpu", endpoint_url="https://cpu.tunnel.nebius.cloud")

    fake_client = FakeAsyncClient([
        FakeResponse({"run_id": 7, "status": "completed", "current_step": 1000, "total_steps": 1000}),
    ])
    monkeypatch.setattr(training_module.httpx, "AsyncClient", lambda timeout=30: fake_client)

    resp = await client.get("/api/training/open")

    assert resp.status_code == 200
    assert run_id not in {r["id"] for r in resp.json()}
