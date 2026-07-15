"""Thin wrapper around the `nebius ai endpoint` CLI (not the SDK).

CLI over SDK for the same reason as the (now-removed) job client: it's what
was already proven to work in the manual smoke test — see
evidence/nebius-endpoint/endpoint.json. Endpoints are the only remote
training track after the 2026-07-11 pivot away from jobs, since only
endpoints expose a public HTTPS URL the controller can proxy requests to.
"""

import asyncio
import json
import re

import httpx

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
    except asyncio.CancelledError:
        # A caller (e.g. a cancelled in-flight provisioning task, see
        # backend/api/training.py's Cancel handling) cancelled the await —
        # without this, the local `nebius` CLI subprocess would keep running
        # orphaned on this machine even though nothing is waiting on it
        # anymore. Doesn't stop whatever the CLI already submitted to
        # Nebius's API server-side (not controllable from here, and not
        # needed — see endpoint_create_kwargs' docstring on shared workers).
        proc.kill()
        await proc.wait()
        raise
    if proc.returncode != 0:
        raise NebiusEndpointError(
            f"nebius {' '.join(args)} failed (exit {proc.returncode}): {stderr.decode().strip()}"
        )
    if not stdout.strip():
        # Seen live 2026-07-12 on a GPU `endpoint create`: exit code 0, empty
        # stdout, endpoint created successfully server-side anyway (visible
        # in the console) — whatever nebius printed about it went to stderr
        # instead, and calling code (e.g. json.loads on empty stdout) would
        # otherwise crash with an opaque JSONDecodeError that discards it.
        raise NebiusEndpointError(
            f"nebius {' '.join(args)} exited 0 but produced no stdout. "
            f"stderr: {stderr.decode().strip() or '(empty)'}. "
            "The command may have still succeeded server-side — check "
            "`nebius ai endpoint list --format json`."
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
    try:
        return json.loads(output)["metadata"]["id"]
    except json.JSONDecodeError as exc:
        # Seen live 2026-07-12 on a GPU create: exit 0, non-empty stdout,
        # but not valid JSON (--format json silently not honored that run) —
        # the CLI printed its normal human-readable table instead. The ID is
        # right there as the first line ("Endpoint ID: <id>") — parse it out
        # rather than fail on a resource that was actually created
        # successfully. Falls back to raising (with the raw output included,
        # not a blind JSONDecodeError) only if that line isn't found either.
        match = re.search(r"^Endpoint ID:\s*(\S+)", output, re.MULTILINE)
        if match:
            return match.group(1)
        raise NebiusEndpointError(
            f"nebius ai endpoint create exited 0 but returned unparseable output: {output!r}. "
            "The endpoint may still have been created server-side — check "
            "`nebius ai endpoint list --format json`."
        ) from exc


async def list_endpoints() -> list[dict]:
    output = await _run_cli("ai", "endpoint", "list", "--format", "json")
    return json.loads(output).get("items", [])


async def find_endpoint(name: str, state: str) -> dict | None:
    """Live Nebius endpoint matching name and the given lifecycle state, or
    None. Only exact state matches — Nebius's transient starting/
    provisioning state strings aren't confirmed, so anything not asked for
    explicitly falls through to normal create/restart behavior rather than
    risk adopting something not actually usable yet."""
    for ep in await list_endpoints():
        if ep.get("metadata", {}).get("name") == name and ep.get("status", {}).get("state") == state:
            return ep
    return None


async def find_running_endpoint(name: str) -> dict | None:
    """Live Nebius endpoint matching name and already RUNNING, or None.

    Used to adopt an endpoint that exists server-side but isn't tracked in
    our own DB (e.g. created via scripts/create_nebius_endpoint.py before it
    wrote to worker_sessions, or any other out-of-band creation) instead of
    blindly creating a duplicate — confirmed live 2026-07-12: a manually
    created CPU endpoint the app didn't know about got duplicated this way.
    """
    return await find_endpoint(name, "RUNNING")


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


async def probe_endpoint_url(url: str) -> bool:
    """Is this endpoint's public tunnel actually routing to a live container?

    Real incident, 2026-07-15: a CPU endpoint reported State: RUNNING (and
    its container's own logs showed a clean, uninterrupted startup, no
    crash) while its public tunnel URL returned a bare, non-JSON 404 for
    every path — Nebius's own gateway responding, not this app. This app's
    GET /api/experiments always returns 200 with a JSON list when it's
    genuinely alive; anything else (wrong status, non-JSON body, timeout,
    connection error) means the tunnel isn't actually reaching a live
    container, even though Nebius's reported endpoint *state* — the only
    thing ensure_worker()'s READY liveness check inspected before this —
    said RUNNING the whole time. Deliberately checking for the specific
    success shape rather than "not a 5xx", since the failure mode here
    was a 404, not a 500. See docs/DESIGN_DECISIONS.md.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{url}/api/experiments")
        if resp.status_code != 200:
            return False
        return isinstance(resp.json(), list)
    except (httpx.HTTPError, ValueError):
        return False


async def get_logs(endpoint_id: str, tail: int = 200) -> str:
    """Recent raw container logs — evidence/debugging, not the structured-events feed."""
    return await _run_cli(
        "ai", "endpoint", "logs", endpoint_id, "--tail", str(tail), "--timestamps",
    )


async def start_endpoint(endpoint_id: str) -> None:
    # Distinct, longer timeout than the general nebius_cli_timeout_seconds —
    # see nebius_endpoint_start_timeout_seconds in config/settings.py.
    await _run_cli(
        "ai", "endpoint", "start", "--id", endpoint_id,
        timeout=settings.nebius_endpoint_start_timeout_seconds,
    )


async def stop_endpoint(endpoint_id: str) -> None:
    await _run_cli("ai", "endpoint", "stop", "--id", endpoint_id)
