"""Shared test fixtures and test doubles."""

import httpx


class FakeResponse:
    """Test double for httpx.Response."""
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
    """Test double for httpx.AsyncClient — returns queued responses in call order."""

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


import pytest


@pytest.fixture(autouse=True)
def block_real_nebius_cli(request, monkeypatch):
    """Hard guard: no test may ever spawn the real `nebius` binary.

    Real-cost incident, 2026-07-15: two tests with an unmocked
    create_endpoint invoked the real CLI on every suite run and created
    NINE real (billing) CPU endpoints before anyone noticed. Every
    endpoints_client call funnels through _run_cli, so failing it here
    makes that class of leak impossible regardless of per-test mock
    discipline. Tests that exercise _run_cli's own parsing logic
    monkeypatch it themselves, which overrides this fixture.
    """
    from backend.nebius import endpoints_client

    # _run_cli's own unit tests run the real function against a FAKE
    # subprocess (monkeypatched create_subprocess_exec) — they opt out.
    if request.node.get_closest_marker("run_cli_unit"):
        return

    async def refuse(*args, **kwargs):
        raise AssertionError(
            f"Test attempted to run the REAL nebius CLI: nebius {' '.join(map(str, args))} "
            "— mock the endpoints_client function this code path calls."
        )

    monkeypatch.setattr(endpoints_client, "_run_cli", refuse)
