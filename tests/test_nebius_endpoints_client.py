import asyncio
import json

import pytest

from backend.nebius import endpoints_client


class FakeProc:
    def __init__(self, stdout: bytes, stderr: bytes, returncode: int):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.killed = False

    async def communicate(self):
        return self._stdout, self._stderr

    def kill(self):
        self.killed = True

    async def wait(self):
        return self.returncode


class HangingProc(FakeProc):
    """Simulates the real 2026-07-11 incident: `nebius` blocked on stdin under uvicorn."""

    async def communicate(self):
        await asyncio.sleep(999)
        return self._stdout, self._stderr


async def test_create_endpoint_parses_endpoint_id(monkeypatch):
    async def fake_run_cli(*args, **kwargs):
        return json.dumps({"metadata": {"id": "aiendpoint-abc123"}})

    monkeypatch.setattr(endpoints_client, "_run_cli", fake_run_cli)

    endpoint_id = await endpoints_client.create_endpoint(
        name="llm-lab-cpu-trainer", image="cr.example/llm-lab-backend:phase2",
        platform="cpu-d3", preset="4vcpu-16gb", container_port=8000,
        subnet_id="vpcsubnet-your-subnet-id-here",
    )

    assert endpoint_id == "aiendpoint-abc123"


async def test_create_endpoint_parses_id_from_human_readable_fallback(monkeypatch):
    """Regression test for the 2026-07-12 GPU incident: --format json was
    silently not honored, CLI printed its normal human-readable table
    instead. Endpoint ID must still be recovered from the first line."""
    human_readable_output = (
        "Endpoint ID: aiendpoint-xyz789\n"
        "Endpoint created successfully.\n"
        "Endpoint:\n"
        "  ID:       aiendpoint-xyz789\n"
        "  Name:     llm-lab-gpu-trainer\n"
        "  State:    RUNNING\n"
    )

    async def fake_run_cli(*args, **kwargs):
        return human_readable_output

    monkeypatch.setattr(endpoints_client, "_run_cli", fake_run_cli)

    endpoint_id = await endpoints_client.create_endpoint(
        name="llm-lab-gpu-trainer", image="cr.example/llm-lab-trainer-gpu:latest",
        platform="gpu-l40s-a", preset="1gpu-8vcpu-32gb", container_port=8000,
        subnet_id="vpcsubnet-your-subnet-id-here",
    )

    assert endpoint_id == "aiendpoint-xyz789"


async def test_create_endpoint_raises_with_raw_output_when_truly_unparseable(monkeypatch):
    async def fake_run_cli(*args, **kwargs):
        return "some unexpected output with no endpoint ID anywhere"

    monkeypatch.setattr(endpoints_client, "_run_cli", fake_run_cli)

    with pytest.raises(endpoints_client.NebiusEndpointError, match="unexpected output"):
        await endpoints_client.create_endpoint(
            name="llm-lab-cpu-trainer", image="cr.example/llm-lab-trainer-cpu:latest",
            platform="cpu-d3", preset="8vcpu-32gb", container_port=8000,
            subnet_id="vpcsubnet-your-subnet-id-here",
        )


async def test_find_running_endpoint_matches_by_name_and_state(monkeypatch):
    async def fake_run_cli(*args, **kwargs):
        return json.dumps({"items": [
            {"metadata": {"id": "aiendpoint-stopped", "name": "llm-lab-cpu-trainer"}, "status": {"state": "STOPPED"}},
            {"metadata": {"id": "aiendpoint-running", "name": "llm-lab-cpu-trainer"}, "status": {"state": "RUNNING"}},
            {"metadata": {"id": "aiendpoint-other", "name": "llm-lab-gpu-trainer"}, "status": {"state": "RUNNING"}},
        ]})

    monkeypatch.setattr(endpoints_client, "_run_cli", fake_run_cli)

    found = await endpoints_client.find_running_endpoint("llm-lab-cpu-trainer")

    assert found["metadata"]["id"] == "aiendpoint-running"


async def test_find_running_endpoint_returns_none_when_no_match(monkeypatch):
    async def fake_run_cli(*args, **kwargs):
        return json.dumps({"items": []})

    monkeypatch.setattr(endpoints_client, "_run_cli", fake_run_cli)

    assert await endpoints_client.find_running_endpoint("llm-lab-cpu-trainer") is None


async def test_find_endpoint_matches_by_name_and_arbitrary_state(monkeypatch):
    async def fake_run_cli(*args, **kwargs):
        return json.dumps({"items": [
            {"metadata": {"id": "aiendpoint-stopped", "name": "llm-lab-cpu-trainer"}, "status": {"state": "STOPPED"}},
            {"metadata": {"id": "aiendpoint-running", "name": "llm-lab-cpu-trainer"}, "status": {"state": "RUNNING"}},
        ]})

    monkeypatch.setattr(endpoints_client, "_run_cli", fake_run_cli)

    found = await endpoints_client.find_endpoint("llm-lab-cpu-trainer", "STOPPED")

    assert found["metadata"]["id"] == "aiendpoint-stopped"


async def test_get_endpoint_parses_json(monkeypatch):
    async def fake_run_cli(*args):
        return json.dumps({"status": {"state": "RUNNING"}})

    monkeypatch.setattr(endpoints_client, "_run_cli", fake_run_cli)

    endpoint = await endpoints_client.get_endpoint("aiendpoint-abc123")

    assert endpoint["status"]["state"] == "RUNNING"


def test_extract_public_url_returns_https_url():
    endpoint = {
        "status": {
            "public_endpoints": ["https://port8000-abc.tunnel.applications.eu-north1.nebius.cloud"],
        }
    }

    url = endpoints_client.extract_public_url(endpoint)

    assert url == "https://port8000-abc.tunnel.applications.eu-north1.nebius.cloud"


def test_extract_public_url_returns_none_when_absent():
    endpoint = {"status": {"public_endpoints": []}}

    url = endpoints_client.extract_public_url(endpoint)

    assert url is None


async def test_start_endpoint_calls_cli_with_id(monkeypatch):
    captured = {}

    async def fake_run_cli(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return ""

    monkeypatch.setattr(endpoints_client, "_run_cli", fake_run_cli)

    await endpoints_client.start_endpoint("aiendpoint-abc123")

    assert captured["args"] == ("ai", "endpoint", "start", "--id", "aiendpoint-abc123")


async def test_get_logs_calls_cli_with_tail_and_timestamps(monkeypatch):
    captured = {}

    async def fake_run_cli(*args):
        captured["args"] = args
        return "2026-07-11T22:00:00Z INFO Uvicorn running\n"

    monkeypatch.setattr(endpoints_client, "_run_cli", fake_run_cli)

    logs = await endpoints_client.get_logs("aiendpoint-abc123", tail=50)

    assert captured["args"] == (
        "ai", "endpoint", "logs", "aiendpoint-abc123", "--tail", "50", "--timestamps",
    )
    assert logs == "2026-07-11T22:00:00Z INFO Uvicorn running\n"


async def test_stop_endpoint_calls_cli_with_id(monkeypatch):
    captured = {}

    async def fake_run_cli(*args):
        captured["args"] = args
        return ""

    monkeypatch.setattr(endpoints_client, "_run_cli", fake_run_cli)

    await endpoints_client.stop_endpoint("aiendpoint-abc123")

    assert captured["args"] == ("ai", "endpoint", "stop", "--id", "aiendpoint-abc123")


async def test_run_cli_raises_on_nonzero_exit(monkeypatch):
    async def fake_exec(*args, **kwargs):
        return FakeProc(b"", b"boom", 1)

    monkeypatch.setattr(endpoints_client.asyncio, "create_subprocess_exec", fake_exec)

    with pytest.raises(endpoints_client.NebiusEndpointError, match="boom"):
        await endpoints_client._run_cli("ai", "endpoint", "get", "aiendpoint-1")


async def test_run_cli_closes_stdin_so_the_cli_cannot_block_on_input(monkeypatch):
    captured_kwargs = {}

    async def fake_exec(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return FakeProc(b"{}", b"", 0)

    monkeypatch.setattr(endpoints_client.asyncio, "create_subprocess_exec", fake_exec)

    await endpoints_client._run_cli("ai", "endpoint", "get", "aiendpoint-1")

    assert captured_kwargs["stdin"] == asyncio.subprocess.DEVNULL


async def test_run_cli_raises_and_kills_process_on_timeout(monkeypatch):
    hanging = HangingProc(b"", b"", 0)

    async def fake_exec(*args, **kwargs):
        return hanging

    monkeypatch.setattr(endpoints_client.asyncio, "create_subprocess_exec", fake_exec)

    with pytest.raises(endpoints_client.NebiusEndpointError, match="timed out"):
        await endpoints_client._run_cli("ai", "endpoint", "start", "--id", "aiendpoint-1", timeout=0.05)

    assert hanging.killed is True


async def test_run_cli_kills_process_when_cancelled(monkeypatch):
    """Part F regression guard: cancelling a Stop-during-provisioning
    request must not leave the local `nebius` CLI subprocess orphaned."""
    hanging = HangingProc(b"", b"", 0)

    async def fake_exec(*args, **kwargs):
        return hanging

    monkeypatch.setattr(endpoints_client.asyncio, "create_subprocess_exec", fake_exec)

    task = asyncio.ensure_future(
        endpoints_client._run_cli("ai", "endpoint", "start", "--id", "aiendpoint-1")
    )
    await asyncio.sleep(0)  # let it reach the await inside _run_cli
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert hanging.killed is True
