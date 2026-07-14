"""Stopping stuck/paused runs — direct user report, 2026-07-14: four
paused runs in Open Runs, none stoppable from the browser. Two distinct
real bugs found and fixed:

1. Local runs whose status.json is missing entirely (legacy run
   directories predating status.json tracking) always fell through to
   "Run not found" (400), even though the DB itself said paused.
2. Remote runs whose Nebius endpoint had been deleted outside the app
   always 502'd on stop (dead endpoint URL), leaving them permanently
   stuck in Open Runs with no way to clear them.

See docs/DESIGN_DECISIONS.md.
"""

import json

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from backend import db
from backend.api import training as training_module
from backend.main import app
from backend.training.status import RunStatus
from config.settings import settings


class FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://fake.example")
            raise httpx.HTTPStatusError("error", request=request, response=self)

    def json(self):
        return self._json_data


class FakeAsyncClient:
    def __init__(self, responses=None, exc=None):
        self.responses = list(responses or [])
        self.exc = exc
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def request(self, method, url, json=None):
        self.calls.append((method, url, json))
        if self.exc:
            raise self.exc
        return self.responses.pop(0)


@pytest.fixture
async def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_lab.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    await db.init_db()
    exp_id = await db.create_experiment("Test experiment", {"template": "transformer"}, preset_key="tiny-shakespeare")
    return exp_id


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_stop_local_paused_run_with_no_status_json_falls_back_to_db_status(temp_db, client, monkeypatch, tmp_path):
    """Legacy run directory — checkpoint.pt/metrics.jsonl/run_meta.json but
    no status.json, exactly as found live for runs 26/27. DB says paused;
    that must be enough to stop it."""
    from backend.training import artifacts

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    exp_id = temp_db
    run_id = await db.create_training_run(exp_id, device="cpu", execution_backend="local")
    await db.update_training_run(run_id, status=RunStatus.PAUSED)

    rd = artifacts.run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "checkpoint.pt").write_bytes(b"fake")
    (rd / "run_meta.json").write_text(json.dumps({"device": "cpu"}))
    assert not (rd / "status.json").exists()

    resp = await client.post(f"/api/training/{run_id}/stop")

    assert resp.status_code == 200
    run = await db.get_training_run(run_id)
    assert run["status"] == RunStatus.CANCELLED


async def test_stop_local_run_still_404s_when_db_also_says_not_paused(temp_db, client, monkeypatch, tmp_path):
    """The fallback is scoped to DB status == paused — a run that's
    genuinely not found/not stoppable still 400s, not silently accepted."""
    from backend.training import artifacts

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    exp_id = temp_db
    run_id = await db.create_training_run(exp_id, device="cpu", execution_backend="local")
    await db.update_training_run(run_id, status=RunStatus.COMPLETED)
    rd = artifacts.run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)

    resp = await client.post(f"/api/training/{run_id}/stop")

    assert resp.status_code == 400


async def test_stop_remote_run_falls_back_to_local_cancel_when_endpoint_is_gone(temp_db, client, monkeypatch):
    """Endpoint deleted outside the app — proxy call fails, but the run
    must still be clearable from Open Runs instead of stuck forever."""
    exp_id = temp_db
    run_id = await db.create_training_run(
        exp_id, "cuda", execution_backend="nebius_endpoint",
        remote_endpoint_id="aiendpoint-gone", remote_run_id=2,
    )
    await db.update_training_run(run_id, status=RunStatus.PAUSED)
    await db.create_worker_session("worker-gpu", "gpu", "nebius_endpoint", 600)
    await db.update_worker_session("worker-gpu", endpoint_url="https://gpu.tunnel.nebius.cloud")

    fake_client = FakeAsyncClient(exc=httpx.ConnectError("connection refused"))
    monkeypatch.setattr(training_module.httpx, "AsyncClient", lambda timeout=30: fake_client)

    resp = await client.post(f"/api/training/{run_id}/stop")

    assert resp.status_code == 200
    run = await db.get_training_run(run_id)
    assert run["status"] == RunStatus.CANCELLED


async def test_stop_remote_run_succeeds_normally_when_endpoint_responds(temp_db, client, monkeypatch):
    """Sanity check: the fallback doesn't mask a working remote stop —
    the normal proxied path is unaffected."""
    exp_id = temp_db
    run_id = await db.create_training_run(
        exp_id, "cuda", execution_backend="nebius_endpoint",
        remote_endpoint_id="aiendpoint-abc", remote_run_id=1,
    )
    await db.update_training_run(run_id, status=RunStatus.PAUSED)
    await db.create_worker_session("worker-gpu", "gpu", "nebius_endpoint", 600)
    await db.update_worker_session("worker-gpu", endpoint_url="https://gpu.tunnel.nebius.cloud")

    fake_client = FakeAsyncClient(responses=[FakeResponse({"run_id": 1, "status": "stopping"})])
    monkeypatch.setattr(training_module.httpx, "AsyncClient", lambda timeout=30: fake_client)

    resp = await client.post(f"/api/training/{run_id}/stop")

    assert resp.status_code == 200
    assert fake_client.calls[0] == ("POST", "https://gpu.tunnel.nebius.cloud/api/training/1/stop", None)
