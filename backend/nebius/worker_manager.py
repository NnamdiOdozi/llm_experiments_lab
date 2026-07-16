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


class WorkerBusyError(RuntimeError):
    """Raised when a worker for this device is already mid-provision from a
    different request — debounce against spawning multiple endpoints for
    the same device type. See ensure_worker()."""


# One lock per session_id (only "worker-cpu"/"worker-gpu" ever exist).
# Real incident, 2026-07-15 — Nebius support traced an endpoint ERROR state
# to two overlapping Start commands from this app's own service account.
# ensure_worker() reads worker_status, then only writes STARTING back
# several lines later (after the actual network call to Nebius) — nothing
# locked that gap, so two /start requests landing close together (e.g. a
# retry after a client gave up waiting on a still-running server-side
# coroutine — FastAPI does not cancel a request's coroutine just because
# the client disconnected) could both read the pre-STARTING status and
# both fire a Start command. This lock wraps the entire check-then-act
# sequence so that can't happen. See docs/DESIGN_DECISIONS.md §79a.
_session_locks: dict[str, asyncio.Lock] = {}


def _lock_for(session_id: str) -> asyncio.Lock:
    if session_id not in _session_locks:
        _session_locks[session_id] = asyncio.Lock()
    return _session_locks[session_id]


async def _live_state(session: dict) -> dict:
    """Fetch current Nebius state for a session's endpoint.

    Returns a dict with keys:
    - exists: bool — endpoint found on Nebius (False if CLI error)
    - state: str|None — lifecycle state (RUNNING, STOPPED, STOPPING, ERROR, etc.) or None
    - reachable: bool|None — tunnel answering (only checked if state==RUNNING, else None)
    - endpoint: dict|None — full endpoint dict from get_endpoint or None

    All decision points in ensure_worker use THIS, never raw DB status alone.
    See docs/DESIGN_DECISIONS.md Part 1.
    """
    if not session or not session.get("nebius_endpoint_id"):
        return {"exists": False, "state": None, "reachable": None, "endpoint": None}

    try:
        endpoint = await endpoints_client.get_endpoint(session["nebius_endpoint_id"])
    except endpoints_client.NebiusEndpointError:
        return {"exists": False, "state": None, "reachable": None, "endpoint": None}

    state = endpoint.get("status", {}).get("state")
    reachable = None
    # Only probe the URL if Nebius reports RUNNING — probing a STOPPED/STOPPING/ERROR
    # endpoint's URL would be pointless (no container to answer).
    if state == "RUNNING" and session.get("endpoint_url"):
        reachable = await endpoints_client.probe_endpoint_url(session["endpoint_url"])

    return {
        "exists": True,
        "state": state,
        "reachable": reachable,
        "endpoint": endpoint,
    }


def endpoint_create_kwargs(device_type: str) -> dict:
    """Single source of truth for "which settings back a cpu vs gpu
    endpoint" — reused by create_new_worker() (the running app's
    automatic path) and scripts/create_nebius_endpoint.py (manual/standalone
    creation), so the two can never drift out of sync with each other."""
    return dict(
        name=settings.nebius_gpu_endpoint_name if device_type == "gpu" else settings.nebius_cpu_endpoint_name,
        image=settings.nebius_gpu_trainer_image if device_type == "gpu" else settings.nebius_cpu_trainer_image,
        platform=settings.nebius_gpu_platform if device_type == "gpu" else settings.nebius_cpu_platform,
        preset=settings.nebius_gpu_preset if device_type == "gpu" else settings.nebius_cpu_preset,
        container_port=settings.nebius_endpoint_container_port,
        subnet_id=settings.nebius_subnet_id,
    )


async def create_new_worker(session_id: str, device_type: str) -> str:
    """Get session_id a usable endpoint — adopts a live one, restarts a
    stopped one, or creates fresh, in that preference order.

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

    kwargs = endpoint_create_kwargs(device_type)
    # Part 3 / D3: Adoption fix — find endpoint in ANY state (not just
    # RUNNING or STOPPED), so a console-started or console-stopped endpoint
    # mid-STARTING or mid-STOPPING (e.g. user started/stopped via Nebius console
    # seconds earlier) is found and adopted, never duplicated.
    existing = await endpoints_client.find_endpoint_any_state(kwargs["name"])
    # A DELETING endpoint is a ghost — it will never come back, so adopting
    # it just wastes the whole poll budget watching a corpse until it 404s.
    # Live incident 2026-07-16 (run 227): the just-console-deleted GPU
    # endpoint was adopted "in DELETING state" and the run burned its wait
    # on it. Treat DELETING exactly like "no endpoint found": create fresh.
    if existing is not None and existing.get("status", {}).get("state") == "DELETING":
        nebius_log.info(
            "Ignoring endpoint mid-deletion — will create fresh — "
            "session_id=%s endpoint_id=%s name=%s",
            session_id, existing["metadata"]["id"], kwargs["name"],
        )
        existing = None
    if existing is not None:
        endpoint_id = existing["metadata"]["id"]
        state = existing.get("status", {}).get("state")
        if state == "ERROR":
            # ERROR is unrecoverable by restart — only deletion works
            # (confirmed by Nebius support). Don't adopt; delete and create fresh.
            await endpoints_client.delete_endpoint(endpoint_id)
            endpoint_id = await endpoints_client.create_endpoint(**kwargs)
            nebius_log.info(
                "ERROR endpoint deleted and fresh one created — "
                "session_id=%s old_endpoint_id=%s new_endpoint_id=%s name=%s",
                session_id, existing["metadata"]["id"], endpoint_id, kwargs["name"],
            )
        elif state == "RUNNING":
            # Already live and running — adopt immediately
            nebius_log.info(
                "Adopted existing RUNNING endpoint — session_id=%s endpoint_id=%s name=%s",
                session_id, endpoint_id, kwargs["name"],
            )
        elif state == "STOPPED":
            # A stopped endpoint with this name is just as good as a running one
            # — restart it rather than abandoning it and paying for a fresh
            # create. Only issues the start command here; ensure_worker's own
            # polling loop already waits for RUNNING and populates endpoint_url
            # regardless of how we got to PROVISIONING, so no separate wait logic
            # is needed here.
            await endpoints_client.start_endpoint(endpoint_id)
            nebius_log.info(
                "Restarted stopped endpoint instead of creating a new one — "
                "session_id=%s endpoint_id=%s name=%s", session_id, endpoint_id, kwargs["name"],
            )
        else:
            # STARTING, STOPPING, or any other transient state —
            # adopt the endpoint but don't issue start commands. ensure_worker's
            # own polling loop will watch for settle and handle as needed.
            nebius_log.info(
                "Adopted existing endpoint in %s state — session_id=%s endpoint_id=%s name=%s",
                state, session_id, endpoint_id, kwargs["name"],
            )
    else:
        endpoint_id = await endpoints_client.create_endpoint(**kwargs)
        nebius_log.info(
            "Worker created — session_id=%s endpoint_id=%s device_type=%s", session_id, endpoint_id, device_type,
        )
    await db.update_worker_session(session_id, worker_status=WorkerStatus.PROVISIONING, nebius_endpoint_id=endpoint_id)
    # Real bug, 2026-07-15: this is the only place a worker transitions
    # into existence or comes back from STOPPED — without resetting the
    # idle clock here, a freshly (re)provisioned worker inherited whatever
    # last_activity_at it had from BEFORE (sometimes a much older,
    # already-near-expiry value), so the idle-timeout warning banner could
    # show "stopping in a few minutes" for a worker that had just started
    # provisioning for a brand new run — confusing and, if bad enough
    # timing, meant idle_monitor's next scan could see it as overdue the
    # moment it reaches READY. See docs/DESIGN_DECISIONS.md.
    await db.touch_worker_session(session_id)
    return endpoint_id


async def ensure_worker(device: str) -> dict:
    """Return a READY worker_session dict (with endpoint_url) for the given device.

    Decision table (under lock) — determines whether to reuse, create, or restart:
    1. DB READY + live RUNNING+reachable → reuse (touch, return)
    2. DB PROVISIONING/STARTING + live STARTING|RUNNING → WorkerBusyError
    3. live ERROR (any DB state) → delete → create fresh (Part 2 / D2)
    4. live RUNNING + NOT reachable (dead tunnel) → stop → create (Part 2 / D4)
    5. live STOPPING → do NOT start; poll loop handles settle-then-start
    6. live STOPPED / exists-False / DB NONE/STOPPED/FAILED → start-or-create
    7. DB SHUTTING_DOWN → decision from live state (Part 5 / D6)

    Poll loop (outside lock) — waits for RUNNING with bounded retries on STOPPED:
    - Up to settings.nebius_endpoint_start_max_retries attempts (Part 7 / D7)
    - Retry when poll observes STOPPED and no successful start yet
    - Re-create if endpoint disappears mid-wait
    """
    device_type = device_type_for(device)
    session_id = session_id_for(device_type)

    async with _lock_for(session_id):
        session = await db.get_worker_session(session_id)

        # --- Decision 1: DB READY + live RUNNING+reachable → reuse
        if session is not None and session["worker_status"] == WorkerStatus.READY:
            live = await _live_state(session)
            if live["exists"] and live["state"] == "RUNNING" and live["reachable"]:
                await db.touch_worker_session(session_id)
                nebius_log.info(
                    "Worker reused — session_id=%s endpoint_id=%s", session_id, session["nebius_endpoint_id"],
                )
                return await db.get_worker_session(session_id)
            # READY liveness check failed — log and continue to re-provision
            nebius_log.warning(
                "Worker marked READY in DB but endpoint is gone, not running, or its "
                "tunnel isn't answering — session_id=%s endpoint_id=%s. Re-provisioning.",
                session_id, session.get("nebius_endpoint_id"),
            )

        # --- Decision 2: DB PROVISIONING/STARTING + live STARTING|RUNNING → WorkerBusyError
        if session is not None and session["worker_status"] in (WorkerStatus.PROVISIONING, WorkerStatus.STARTING):
            live = await _live_state(session)
            # Only treat as busy if live state contradicts — stale DB states (endpoint gone,
            # ERROR, STOPPED, STOPPING) mean the endpoint actually died and this is recoverable.
            if live["exists"] and live["state"] in ("STARTING", "RUNNING"):
                nebius_log.info(
                    "Worker busy, debounced — session_id=%s status=%s", session_id, session["worker_status"],
                )
                raise WorkerBusyError(
                    f"{device_type} worker is already being provisioned (status={session['worker_status']}) "
                    "— please wait for it to finish before starting another run."
                )
            if live["exists"] and live["state"]:
                # Endpoint exists but in a recoverable state (STOPPED/STOPPING/ERROR/etc.)
                nebius_log.warning(
                    "Worker marked %s in DB but its endpoint is actually %s — treating as stale, not busy.",
                    session["worker_status"], live["state"]
                )

        # --- For remaining decisions, fetch fresh live state
        live = await _live_state(session)
        endpoint_id = session["nebius_endpoint_id"] if session else None

        # --- Decision 3 (extended): live RUNNING+reachable (any DB state) → reuse
        # Handles case where DB status is STOPPED/FAILED but Nebius says it's RUNNING.
        # This can happen if user manually started the endpoint via console.
        if live["exists"] and live["state"] == "RUNNING" and live["reachable"]:
            await db.touch_worker_session(session_id)
            if session and session["worker_status"] != WorkerStatus.READY:
                await db.update_worker_session(session_id, worker_status=WorkerStatus.READY)
                spec = live["endpoint"].get("spec", {})
                url = endpoints_client.extract_public_url(live["endpoint"])
                if url:
                    await db.update_worker_session(session_id, endpoint_url=url,
                        actual_platform=spec.get("platform"), actual_preset=spec.get("preset"))
            nebius_log.info(
                "Worker reused (was %s in DB) — session_id=%s endpoint_id=%s",
                session.get("worker_status") if session else "unknown", session_id, endpoint_id,
            )
            return await db.get_worker_session(session_id)

        # --- Decision 4: live ERROR (any DB state) → delete → create fresh (Part 2 / D2)
        if live["exists"] and live["state"] == "ERROR":
            nebius_log.error(
                "Endpoint in ERROR state (only recovery is deletion) — session_id=%s endpoint_id=%s",
                session_id, endpoint_id,
            )
            await endpoints_client.delete_endpoint(endpoint_id)
            endpoint_id = await create_new_worker(session_id, device_type)

        # --- Decision 5: live RUNNING + tunnel CONFIRMED dead → stop → create (Part 2 / D4)
        # `reachable is False` (probe ran and failed), NOT `not reachable`:
        # a session with no stored endpoint_url yet (e.g. adopted mid-
        # provision) has reachable=None — unknown, not dead. Treating None
        # as dead stopped and recreated healthy RUNNING endpoints.
        elif live["exists"] and live["state"] == "RUNNING" and live["reachable"] is False:
            nebius_log.warning(
                "Endpoint RUNNING but tunnel unreachable (dead gateway) — stopping and recreating — "
                "session_id=%s endpoint_id=%s", session_id, endpoint_id,
            )
            try:
                await endpoints_client.stop_endpoint(endpoint_id)
            except endpoints_client.NebiusEndpointError as exc:
                nebius_log.warning(
                    "Failed to stop dead-tunnel endpoint (will be orphaned) — "
                    "session_id=%s endpoint_id=%s: %s", session_id, endpoint_id, exc,
                )
            endpoint_id = await create_new_worker(session_id, device_type)

        # --- Decision 5b: live RUNNING (reachability unknown) or STARTING —
        # e.g. console-started out-of-band, or a session with no URL recorded
        # yet. Nothing to command; record the transition and let the poll
        # loop confirm RUNNING and populate endpoint_url. Without this
        # branch these fell into the final else → create_new_worker — a
        # wasted name-lookup round-trip at best.
        elif live["exists"] and live["state"] in ("RUNNING", "STARTING"):
            if not session or session["worker_status"] not in (WorkerStatus.PROVISIONING, WorkerStatus.STARTING):
                await db.update_worker_session(session_id, worker_status=WorkerStatus.STARTING)
                await db.touch_worker_session(session_id)
                nebius_log.info(
                    "Endpoint already %s on Nebius — adopting in place, waiting for it to answer — "
                    "session_id=%s endpoint_id=%s", live["state"], session_id, endpoint_id,
                )

        # --- Decision 6: live STOPPING → do NOT start; fall through to poll loop
        elif live["exists"] and live["state"] == "STOPPING":
            # Don't attempt start while settling. Poll loop below waits for settle then retries.
            if not session or session["worker_status"] not in (WorkerStatus.PROVISIONING, WorkerStatus.STARTING):
                await db.update_worker_session(session_id, worker_status=WorkerStatus.STARTING)
                await db.touch_worker_session(session_id)
                # endpoint_id already set at line 237; just log and continue
                nebius_log.info(
                    "Endpoint settling (STOPPING) — polling for settle-then-start — "
                    "session_id=%s endpoint_id=%s", session_id, endpoint_id,
                )

        # --- Decision 7: live STOPPED / exists-False / DB NONE/STOPPED/FAILED / SHUTTING_DOWN
        elif live["exists"] and live["state"] == "STOPPED":
            # Start the stopped endpoint
            if not session or session["worker_status"] != WorkerStatus.STARTING:
                try:
                    await endpoints_client.start_endpoint(endpoint_id)
                    await db.update_worker_session(session_id, worker_status=WorkerStatus.STARTING)
                    await db.touch_worker_session(session_id)
                    nebius_log.info("Worker starting — session_id=%s endpoint_id=%s", session_id, endpoint_id)
                except endpoints_client.NebiusEndpointError as exc:
                    # Start failed — might be slow/transient, or endpoint might be gone
                    try:
                        await endpoints_client.get_endpoint(endpoint_id)
                        nebius_log.warning(
                            "Start command failed but endpoint still exists — treating as slow, "
                            "continuing to wait — session_id=%s endpoint_id=%s: %s",
                            session_id, endpoint_id, exc,
                        )
                        await db.update_worker_session(session_id, worker_status=WorkerStatus.STARTING)
                        await db.touch_worker_session(session_id)
                    except endpoints_client.NebiusEndpointError:
                        nebius_log.warning(
                            "Start failed and endpoint no longer exists — creating new one — "
                            "session_id=%s endpoint_id=%s: %s", session_id, endpoint_id, exc,
                        )
                        endpoint_id = await create_new_worker(session_id, device_type)
        else:
            # No endpoint at all (DB NONE, DB STOPPED/FAILED, or DB SHUTTING_DOWN with endpoint gone)
            # or endpoint gone mid-flight → create fresh
            endpoint_id = await create_new_worker(session_id, device_type)

    # --- Poll loop (outside lock) — wait for endpoint to reach RUNNING
    max_attempts = int(max(
        1, settings.nebius_endpoint_ready_timeout_seconds // settings.nebius_endpoint_poll_interval_seconds,
    ))
    start_attempts = 0  # Count all start attempts for bounded retry (Part 7 / D7)

    for _ in range(max_attempts):
        try:
            endpoint = await endpoints_client.get_endpoint(endpoint_id)
        except endpoints_client.NebiusEndpointError as exc:
            # Endpoint disappeared mid-wait — create fresh and continue poll
            nebius_log.warning(
                "Endpoint disappeared while waiting — session_id=%s endpoint_id=%s: %s. Creating new one.",
                session_id, endpoint_id, exc,
            )
            endpoint_id = await create_new_worker(session_id, device_type)
            continue

        state = endpoint.get("status", {}).get("state")

        # --- Bounded retry: start when settled to STOPPED, up to max_retries (Part 7 / D7)
        if state == "STOPPED" and start_attempts < settings.nebius_endpoint_start_max_retries:
            start_attempts += 1
            try:
                await endpoints_client.start_endpoint(endpoint_id)
                nebius_log.info(
                    "Endpoint settled to STOPPED — retrying start (attempt %d/%d) — "
                    "session_id=%s endpoint_id=%s",
                    start_attempts, settings.nebius_endpoint_start_max_retries, session_id, endpoint_id,
                )
            except endpoints_client.NebiusEndpointError as exc:
                nebius_log.warning(
                    "Retry start failed (attempt %d/%d) — session_id=%s endpoint_id=%s: %s",
                    start_attempts, settings.nebius_endpoint_start_max_retries, session_id, endpoint_id, exc,
                )

        # --- Check if reached RUNNING with URL
        if state == "RUNNING":
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

    # --- Timeout: provisioning never reached RUNNING
    await db.update_worker_session(session_id, worker_status=WorkerStatus.FAILED)
    nebius_log.error("Worker provisioning timed out — session_id=%s endpoint_id=%s", session_id, endpoint_id)
    raise WorkerProvisionError(f"Endpoint {endpoint_id} did not reach RUNNING within timeout")
