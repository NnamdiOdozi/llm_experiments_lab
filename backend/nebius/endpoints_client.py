"""Thin wrapper around the `nebius ai endpoint` CLI (not the SDK).

CLI over SDK for the same reason as the (now-removed) job client: it's what
was already proven to work in the manual smoke test — see
evidence/nebius-endpoint/endpoint.json. Endpoints are the only remote
training track after the 2026-07-11 pivot away from jobs, since only
endpoints expose a public HTTPS URL the controller can proxy requests to.
"""

import asyncio
import json

from config.settings import settings


class NebiusEndpointError(RuntimeError):
    """Raised when the `nebius` CLI exits non-zero, or doesn't exit at all."""


async def _run_cli(*args: str, timeout: float | None = None) -> str:
    """Run a `nebius` CLI command.

    stdin is explicitly closed (DEVNULL) and the call is timeout-bounded.
    Both guard the same real incident (2026-07-11): `nebius ai endpoint
    start` hung indefinitely when run as a subprocess of the uvicorn
    server — it inherited uvicorn's stdin, which never produced EOF, so
    the CLI sat blocked waiting for input a human would normally provide
    interactively. A direct terminal test with `< /dev/null` returned in
    seconds, confirming it. Without the timeout, any future stuck CLI
    call (not just this one) would hang the request forever.
    """
    proc = await asyncio.create_subprocess_exec(
        "nebius", *args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout or settings.nebius_cli_timeout_seconds,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise NebiusEndpointError(f"nebius {' '.join(args)} timed out")
    if proc.returncode != 0:
        raise NebiusEndpointError(
            f"nebius {' '.join(args)} failed (exit {proc.returncode}): {stderr.decode().strip()}"
        )
    return stdout.decode()


async def create_endpoint(
    name: str, image: str, platform: str, preset: str, container_port: int, subnet_id: str,
) -> str:
    """Create an endpoint hosting the backend image. Returns the endpoint_id."""
    # `endpoint create` itself can take several minutes to return (observed
    # live 2026-07-12: still short of returning past 60s while the endpoint
    # was already visible and "starting" in the Nebius console) — the
    # default nebius_cli_timeout_seconds (60s, sized for quick calls like
    # get/start/stop) killed the subprocess before it returned, even though
    # the resource had already been created server-side. Reuses
    # nebius_endpoint_ready_timeout_seconds (6min) rather than inventing a
    # separate setting, since that's already this project's "how long CPU
    # endpoint startup can take" constant.
    output = await _run_cli(
        "ai", "endpoint", "create",
        "--name", name,
        "--image", image,
        "--container-port", str(container_port),
        "--platform", platform,
        "--preset", preset,
        "--subnet-id", subnet_id,
        "--format", "json",
        timeout=settings.nebius_endpoint_ready_timeout_seconds,
    )
    return json.loads(output)["metadata"]["id"]


async def get_endpoint(endpoint_id: str) -> dict:
    """Current lifecycle status + public URL for an endpoint."""
    output = await _run_cli("ai", "endpoint", "get", endpoint_id, "--format", "json")
    return json.loads(output)


def extract_public_url(endpoint: dict) -> str | None:
    """First https:// URL from an endpoint's status, or None if not yet assigned."""
    for url in endpoint.get("status", {}).get("public_endpoints", []):
        if url.startswith("https://"):
            return url
    return None


async def get_logs(endpoint_id: str, tail: int = 200) -> str:
    """Recent raw container logs — evidence/debugging, not the structured-events feed."""
    return await _run_cli(
        "ai", "endpoint", "logs", endpoint_id, "--tail", str(tail), "--timestamps",
    )


async def start_endpoint(endpoint_id: str) -> None:
    await _run_cli("ai", "endpoint", "start", "--id", endpoint_id)


async def stop_endpoint(endpoint_id: str) -> None:
    await _run_cli("ai", "endpoint", "stop", "--id", endpoint_id)
