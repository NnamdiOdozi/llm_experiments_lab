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
    async def fake_run_cli(*args):
        return json.dumps({"metadata": {"id": "aiendpoint-abc123"}})

    monkeypatch.setattr(endpoints_client, "_run_cli", fake_run_cli)

    endpoint_id = await endpoints_client.create_endpoint(
        name="llm-lab-cpu-trainer", image="cr.example/llm-lab-backend:phase2",
        platform="cpu-d3", preset="4vcpu-16gb", container_port=8000,
        subnet_id="vpcsubnet-e00yp4qcbmpde8x2nc",
    )

    assert endpoint_id == "aiendpoint-abc123"


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

    async def fake_run_cli(*args):
        captured["args"] = args
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
