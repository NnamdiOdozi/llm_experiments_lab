"""Ensures a CPU or GPU trainer endpoint is running and ready to take a request.

One shared endpoint per device type (session_id = "worker-cpu" / "worker-gpu"),
not one per user session — Track B's per-user job model was dropped in the
2026-07-11 pivot. See docs/NEBIUS_SERVERLESS_IMPLEMENTATION_PLAN.md.
"""

import asyncio

from backend import db
from backend.logging_config import nebius_log
from backend.nebius import endpoints_client
from backend.training.worker_status import WorkerStatus, device_type_for, session_id_for
from config.settings import settings


class WorkerProvisionError(RuntimeError):
    """Raised when an endpoint doesn't reach RUNNING within the poll timeout."""


async def _create_new_worker(session_id: str, device_type: str) -> str:
    """Create a fresh endpoint and point session_id at it.

    Idempotent w.r.t. the session_id row: called both for a brand-new
    session (no row yet) and for recovery when an existing session's
    endpoint was deleted outside the app (row exists, just needs a new
    endpoint_id) — so it must not assume the row is absent.
    """
    idle_timeout = (
        settings.gpu_idle_timeout_seconds if device_type == "gpu" else settings.cpu_idle_timeout_seconds
    )
    if await db.get_worker_session(session_id) is None:
        await db.create_worker_session(session_id, device_type, "nebius_endpoint", idle_timeout)
    endpoint_id = await endpoints_client.create_endpoint(
        name=settings.nebius_gpu_endpoint_name if device_type == "gpu" else settings.nebius_cpu_endpoint_name,
        image=settings.nebius_backend_image,
        platform=settings.nebius_gpu_platform if device_type == "gpu" else settings.nebius_cpu_platform,
        preset=settings.nebius_gpu_preset if device_type == "gpu" else settings.nebius_cpu_preset,
        container_port=settings.nebius_endpoint_container_port,
        subnet_id=settings.nebius_subnet_id,
    )
    await db.update_worker_session(session_id, worker_status=WorkerStatus.PROVISIONING, nebius_endpoint_id=endpoint_id)
    nebius_log.info(
        "Worker created — session_id=%s endpoint_id=%s device_type=%s", session_id, endpoint_id, device_type,
    )
    return endpoint_id


async def ensure_worker(device: str) -> dict:
    """Return a READY worker_session dict (with endpoint_url) for the given device."""
    device_type = device_type_for(device)
    session_id = session_id_for(device_type)
    session = await db.get_worker_session(session_id)

    if session is not None and session["worker_status"] == WorkerStatus.READY:
        await db.touch_worker_session(session_id)
        nebius_log.info(
            "Worker reused — session_id=%s endpoint_id=%s", session_id, session["nebius_endpoint_id"],
        )
        return await db.get_worker_session(session_id)

    if session is None:
        endpoint_id = await _create_new_worker(session_id, device_type)
    else:
        endpoint_id = session["nebius_endpoint_id"]
        try:
            await endpoints_client.start_endpoint(endpoint_id)
            await db.update_worker_session(session_id, worker_status=WorkerStatus.STARTING)
            nebius_log.info("Worker starting — session_id=%s endpoint_id=%s", session_id, endpoint_id)
        except endpoints_client.NebiusEndpointError as exc:
            # Endpoint was likely deleted outside the app (console, another
            # process, etc.) — our DB row is stale. Self-heal by creating a
            # fresh one rather than failing the request. See 2026-07-11 session.
            nebius_log.warning(
                "Existing endpoint could not be started, assuming it was deleted "
                "outside the app — session_id=%s endpoint_id=%s error=%s. Creating a new one.",
                session_id, endpoint_id, exc,
            )
            endpoint_id = await _create_new_worker(session_id, device_type)

    max_attempts = max(
        1, settings.nebius_endpoint_ready_timeout_seconds // settings.nebius_endpoint_poll_interval_seconds,
    )
    for _ in range(max_attempts):
        try:
            endpoint = await endpoints_client.get_endpoint(endpoint_id)
        except endpoints_client.NebiusEndpointError as exc:
            # Same self-heal, but for disappearing between start and poll.
            nebius_log.warning(
                "Endpoint disappeared while waiting for it to become ready — "
                "session_id=%s endpoint_id=%s error=%s. Creating a new one.",
                session_id, endpoint_id, exc,
            )
            endpoint_id = await _create_new_worker(session_id, device_type)
            continue
        if endpoint.get("status", {}).get("state") == "RUNNING":
            url = endpoints_client.extract_public_url(endpoint)
            if url:
                spec = endpoint.get("spec", {})
                await db.update_worker_session(
                    session_id, worker_status=WorkerStatus.READY, endpoint_url=url,
                    actual_platform=spec.get("platform"), actual_preset=spec.get("preset"),
                )
                await db.touch_worker_session(session_id)
                nebius_log.info(
                    "Worker ready — session_id=%s endpoint_id=%s url=%s", session_id, endpoint_id, url,
                )
                return await db.get_worker_session(session_id)
        await asyncio.sleep(settings.nebius_endpoint_poll_interval_seconds)

    await db.update_worker_session(session_id, worker_status=WorkerStatus.FAILED)
    nebius_log.error("Worker provisioning timed out — session_id=%s endpoint_id=%s", session_id, endpoint_id)
    raise WorkerProvisionError(f"Endpoint {endpoint_id} did not reach RUNNING within timeout")
