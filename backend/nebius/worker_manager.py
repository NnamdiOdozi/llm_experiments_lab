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
    running = await endpoints_client.find_running_endpoint(kwargs["name"])
    stopped = None if running is not None else await endpoints_client.find_endpoint(kwargs["name"], "STOPPED")
    if running is not None:
        # A live endpoint with this name is already RUNNING but wasn't in
        # our DB (e.g. created out-of-band) — adopt it instead of creating
        # a duplicate. Left at PROVISIONING here regardless; ensure_worker's
        # own polling loop confirms RUNNING and populates endpoint_url/
        # actual_platform/actual_preset within one poll cycle (a few
        # seconds), same as a freshly created endpoint — no need to
        # duplicate that population logic here.
        endpoint_id = running["metadata"]["id"]
        nebius_log.info(
            "Adopted existing live endpoint instead of creating a duplicate — "
            "session_id=%s endpoint_id=%s name=%s", session_id, endpoint_id, kwargs["name"],
        )
    elif stopped is not None:
        # A stopped endpoint with this name is just as good as a running one
        # — restart it rather than abandoning it and paying for a fresh
        # create. Only issues the start command here; ensure_worker's own
        # polling loop (below in that function) already waits for RUNNING
        # and populates endpoint_url regardless of how we got to
        # PROVISIONING, so no separate wait logic is needed here. See
        # docs/DESIGN_DECISIONS.md.
        endpoint_id = stopped["metadata"]["id"]
        await endpoints_client.start_endpoint(endpoint_id)
        nebius_log.info(
            "Restarted stopped endpoint instead of creating a new one — "
            "session_id=%s endpoint_id=%s name=%s", session_id, endpoint_id, kwargs["name"],
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
    """Return a READY worker_session dict (with endpoint_url) for the given device."""
    device_type = device_type_for(device)
    session_id = session_id_for(device_type)

    # Everything up to (not including) the poll-for-RUNNING loop below is
    # under this device's lock — the "check status, then decide/commit"
    # section, which is exactly the gap that let two overlapping Start
    # commands through. Deliberately NOT holding the lock across the
    # multi-minute poll loop: that would turn today's fast WorkerBusyError
    # (a second caller fails immediately, retries later) into a caller
    # silently blocking for up to nebius_endpoint_ready_timeout_seconds —
    # a real UX regression, and not what this fix is for. See
    # docs/DESIGN_DECISIONS.md §79a.
    # Set when the READY liveness check below confirms the endpoint still
    # exists and is genuinely RUNNING, just unreachable (dead tunnel) —
    # routes straight to a fresh create instead of the start_endpoint
    # attempt further down. Real bug caught by this fix's own test suite,
    # 2026-07-15: without this, that branch would call start_endpoint() on
    # an endpoint Nebius already reports as RUNNING — precisely the
    # "issuing start command on the already-started instance" overlap
    # Nebius support flagged as the ERROR-state trigger in the first
    # place (§79a). A genuinely-gone or SHUTTING_DOWN endpoint doesn't need
    # this flag; those are already routed correctly below.
    endpoint_confirmed_running_but_unreachable = False

    async with _lock_for(session_id):
        session = await db.get_worker_session(session_id)

        if session is not None and session["worker_status"] == WorkerStatus.READY:
            # Verify liveness before trusting the DB — a manual deletion via
            # the Nebius console (or any other out-of-band action) never
            # updates our DB, so a stale READY row can point at an endpoint
            # that's actually gone. Confirmed live 2026-07-12: after the
            # user deleted every endpoint via the console, the app kept
            # reusing the dead URL and 404ing on every single run until the
            # DB was manually fixed. If it's really gone, fall through to
            # the same start/create logic below instead of returning stale
            # info. See docs/DESIGN_DECISIONS.md.
            try:
                live = await endpoints_client.get_endpoint(session["nebius_endpoint_id"])
            except endpoints_client.NebiusEndpointError:
                live = None
            # Real incident, 2026-07-15: a CPU endpoint reported RUNNING (and
            # its own container logs showed a clean startup, no crash) while
            # its public tunnel URL returned a bare 404 for every path —
            # Nebius's gateway responding, not this app. State alone isn't
            # enough; the URL itself must actually answer. See §79's
            # endpoint-guard entry.
            reachable = (
                live is not None
                and live.get("status", {}).get("state") == "RUNNING"
                and await endpoints_client.probe_endpoint_url(session["endpoint_url"])
            )
            if reachable:
                await db.touch_worker_session(session_id)
                nebius_log.info(
                    "Worker reused — session_id=%s endpoint_id=%s", session_id, session["nebius_endpoint_id"],
                )
                return await db.get_worker_session(session_id)
            endpoint_confirmed_running_but_unreachable = (
                live is not None and live.get("status", {}).get("state") == "RUNNING"
            )
            nebius_log.warning(
                "Worker marked READY in DB but endpoint is gone, not running, or its "
                "tunnel isn't answering — session_id=%s endpoint_id=%s. Re-provisioning.",
                session_id, session["nebius_endpoint_id"],
            )

        if session is not None and session["worker_status"] in (WorkerStatus.PROVISIONING, WorkerStatus.STARTING):
            # Real incident, 2026-07-15: this trusted worker_status
            # completely blindly, unlike the READY path a few lines above
            # (which verifies against Nebius before trusting it). An out-
            # of-band action — user manually stops the endpoint mid-
            # transition — leaves worker_status wedged at STARTING/
            # PROVISIONING forever with nothing left to ever notice and
            # fix it (same root cause class as §79c, different trigger:
            # that one was an in-flight task getting cancelled: this one
            # is the endpoint itself changing state underneath a status
            # nothing re-checks). Confirmed live: after manually stopping
            # the CPU endpoint a second time, two separate new run attempts
            # both got rejected with "already being provisioned" — genuinely
            # nothing was provisioning anymore. Verify against Nebius's
            # actual state before trusting the busy claim. Only treat it as
            # stale for the CLEAR contradicting states (stopped/stopping/
            # error/gone) — an unrecognized state defaults to "assume still
            # genuinely busy" so this can't misfire against an actually-
            # legitimate concurrent request just because of an unfamiliar
            # status string. See docs/DESIGN_DECISIONS.md.
            stale = False
            if session.get("nebius_endpoint_id"):
                try:
                    live = await endpoints_client.get_endpoint(session["nebius_endpoint_id"])
                    state = live.get("status", {}).get("state")
                    stale = state in ("STOPPED", "STOPPING", "ERROR")
                except endpoints_client.NebiusEndpointError:
                    stale = True  # endpoint is genuinely gone
            if not stale:
                # A different request already has this device's worker mid-flight —
                # without this check, ensure_worker would fall through to the
                # create/start branch below and could spawn a second endpoint for
                # the same device type on top of one that's already coming up.
                nebius_log.info(
                    "Worker busy, debounced — session_id=%s status=%s", session_id, session["worker_status"],
                )
                raise WorkerBusyError(
                    f"{device_type} worker is already being provisioned (status={session['worker_status']}) "
                    "— please wait for it to finish before starting another run."
                )
            nebius_log.warning(
                "Worker marked %s in DB but its endpoint's real state contradicts that — "
                "session_id=%s endpoint_id=%s. Treating as stale, not busy.",
                session["worker_status"], session_id, session["nebius_endpoint_id"],
            )

        if (
            session is None
            or session["worker_status"] == WorkerStatus.SHUTTING_DOWN
            or endpoint_confirmed_running_but_unreachable
        ):
            # A SHUTTING_DOWN worker can't be started back up — attempting
            # start_endpoint() on it would just wait out its full timeout (up to
            # 300s) before self-healing to a fresh create anyway. Skip straight
            # to create instead of wasting that wait: a request shouldn't be
            # slowed down by a worker that's mid-deletion, only genuinely
            # blocked by one that's still becoming useful (PROVISIONING/STARTING
            # above). Same reasoning for a confirmed-RUNNING-but-unreachable
            # endpoint — starting an endpoint Nebius already reports as
            # RUNNING is the overlapping-operation problem this whole fix
            # exists to avoid, not a real recovery step.
            endpoint_id = await create_new_worker(session_id, device_type)
        else:
            endpoint_id = session["nebius_endpoint_id"]
            try:
                await endpoints_client.start_endpoint(endpoint_id)
                await db.update_worker_session(session_id, worker_status=WorkerStatus.STARTING)
                # Same idle-clock reset as create_new_worker() — a restarted
                # STOPPED worker must not inherit its idle clock from before
                # it stopped. See docs/DESIGN_DECISIONS.md.
                await db.touch_worker_session(session_id)
                nebius_log.info("Worker starting — session_id=%s endpoint_id=%s", session_id, endpoint_id)
            except endpoints_client.NebiusEndpointError as exc:
                # A start-command timeout does NOT mean the endpoint is gone —
                # confirmed live 2026-07-12: a real GPU cold start outlasted our
                # own client-side wait, the app assumed "deleted outside the
                # app" and abandoned a perfectly good endpoint that went on to
                # finish starting successfully on Nebius's side, creating a
                # wasteful duplicate. Check live status before giving up: only
                # self-heal to a fresh create if the endpoint is genuinely gone
                # (get_endpoint itself fails), not merely slow. See
                # docs/DESIGN_DECISIONS.md.
                try:
                    await endpoints_client.get_endpoint(endpoint_id)
                    nebius_log.warning(
                        "Start command timed out but endpoint still exists — treating as "
                        "slow, not deleted, and continuing to wait on it — "
                        "session_id=%s endpoint_id=%s error=%s", session_id, endpoint_id, exc,
                    )
                    await db.update_worker_session(session_id, worker_status=WorkerStatus.STARTING)
                    await db.touch_worker_session(session_id)
                except endpoints_client.NebiusEndpointError:
                    nebius_log.warning(
                        "Existing endpoint could not be started and no longer exists — "
                        "assuming it was deleted outside the app — session_id=%s "
                        "endpoint_id=%s error=%s. Creating a new one.",
                        session_id, endpoint_id, exc,
                    )
                    endpoint_id = await create_new_worker(session_id, device_type)

    max_attempts = max(
        1, settings.nebius_endpoint_ready_timeout_seconds // settings.nebius_endpoint_poll_interval_seconds,
    )
    # Real incident, 2026-07-15: user manually stopped the CPU endpoint via
    # Nebius directly, then started a new run seconds later. The initial
    # start_endpoint() call landed while the endpoint was still mid-
    # STOPPING (not yet settled) and Nebius explicitly rejected it ("rpc
    # error: code = Internal desc = internal error") — not a timeout, a
    # flat rejection. The existing exception handler above treats every
    # start_endpoint() failure the same way ("maybe just slow, keep
    # passively polling for RUNNING"), which is right for an ambiguous
    # timeout but wrong here: nothing was ever actually accepted, so
    # passively polling get_endpoint() for a start that never happened
    # can only ever time out — confirmed live, the endpoint sat cleanly
    # STOPPED (transition long since finished) while this loop kept
    # checking for RUNNING and finding nothing, for the full budget.
    # Retry the start command (once — bounded, not a tight retry storm,
    # to avoid feeding back into the overlapping-operation problem this
    # whole area exists to avoid) once the poll loop itself observes the
    # endpoint has settled into STOPPED, rather than only ever trying to
    # start it the one time before this loop began. See
    # docs/DESIGN_DECISIONS.md.
    retried_start = False
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
            endpoint_id = await create_new_worker(session_id, device_type)
            continue
        if not retried_start and endpoint.get("status", {}).get("state") == "STOPPED":
            retried_start = True
            try:
                await endpoints_client.start_endpoint(endpoint_id)
                nebius_log.info(
                    "Endpoint settled to STOPPED mid-wait — retrying start — "
                    "session_id=%s endpoint_id=%s", session_id, endpoint_id,
                )
            except endpoints_client.NebiusEndpointError as exc:
                nebius_log.warning(
                    "Retry start also failed — session_id=%s endpoint_id=%s error=%s. "
                    "Continuing to poll; will time out and fail normally if it never comes up.",
                    session_id, endpoint_id, exc,
                )
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
