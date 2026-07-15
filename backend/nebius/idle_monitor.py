"""Background task: stop CPU/GPU endpoints idle past their configured timeout.

Runs as an asyncio task started in backend/main.py's lifespan. Activity is
tracked via worker_sessions.last_activity_at, touched on every proxied
training call (backend/api/training.py's _proxy) plus an explicit "Remain
logged in" heartbeat the frontend can send. See
docs/NEBIUS_SERVERLESS_IMPLEMENTATION_PLAN.md and the 2026-07-11 session log.
"""

import asyncio
from datetime import datetime, timezone

from backend import db
from backend.logging_config import nebius_log
from backend.nebius import endpoints_client
from backend.nebius.endpoints_client import NebiusEndpointError
from backend.training.worker_status import WorkerStatus
from config.settings import settings


def seconds_since(timestamp_str: str) -> float:
    """Seconds elapsed since a SQLite CURRENT_TIMESTAMP string (UTC, no tz suffix)."""
    then = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - then).total_seconds()


async def reconcile_worker_sessions() -> None:
    """Self-heal: converge DB worker_status to live Nebius state.

    Part 5 / D6: Called from run_forever() each scan BEFORE stop_idle_workers().
    For each active session, fetch live Nebius state and update DB if it contradicts
    reality. This handles:
    - Manual deletion via Nebius console (endpoint gone → DB NONE/FAILED)
    - SHUTTING_DOWN never gets a writer (now it does: STOPPING → SHUTTING_DOWN)
    - Stale STARTING/PROVISIONING that the endpoint has already left
    - RUNNING-but-unreachable endpoints (log warning, next ensure_worker handles)

    Skips sessions currently locked by a provisioning task (owns the session,
    knows better than a stale reconcile).
    """
    from backend.nebius import worker_manager  # avoid circular import

    # Only reconcile active sessions (primary use case)
    sessions_to_check = await db.list_active_worker_sessions()

    for session in sessions_to_check:
        session_id = session["session_id"]

        # Skip if a provisioning task currently owns this session
        if worker_manager._lock_for(session_id).locked():
            continue

        try:
            live = await worker_manager._live_state(session)

            # Map live state to DB status
            if not live["exists"]:
                new_status = WorkerStatus.NONE
            elif live["state"] == "ERROR":
                new_status = WorkerStatus.FAILED
            elif live["state"] == "STOPPING":
                new_status = WorkerStatus.SHUTTING_DOWN
            elif live["state"] == "STOPPED":
                new_status = WorkerStatus.STOPPED
            elif live["state"] == "STARTING":
                new_status = WorkerStatus.STARTING
            elif live["state"] == "RUNNING" and live["reachable"]:
                new_status = WorkerStatus.READY
            elif live["state"] == "RUNNING" and not live["reachable"]:
                # Endpoint reachable live but tunnel dead — leave as-is, log warning
                nebius_log.warning(
                    "Reconciler found RUNNING endpoint with dead tunnel — "
                    "next ensure_worker call will handle — session_id=%s endpoint_id=%s",
                    session_id, session.get("nebius_endpoint_id"),
                )
                continue
            else:
                # Unknown state or None — leave as-is
                continue

            old_status = session.get("worker_status")
            if new_status != old_status:
                # Status changed — update and log transition
                update_dict = {"worker_status": new_status}

                # If transitioning to READY, also update endpoint_url and platform/preset
                if new_status == WorkerStatus.READY and live["endpoint"]:
                    url = endpoints_client.extract_public_url(live["endpoint"])
                    spec = live["endpoint"].get("spec", {})
                    if url:
                        update_dict["endpoint_url"] = url
                        update_dict["actual_platform"] = spec.get("platform")
                        update_dict["actual_preset"] = spec.get("preset")

                await db.update_worker_session(session_id, **update_dict)
                nebius_log.info(
                    "Reconciler converged session — session_id=%s endpoint_id=%s %s→%s",
                    session_id, session.get("nebius_endpoint_id"), old_status, new_status,
                )

        except Exception as exc:
            # One session's CLI failure doesn't kill the entire scan
            nebius_log.warning(
                "Reconciler failed for session_id=%s: %s", session_id, exc,
            )


async def stop_idle_workers() -> int:
    """Scan active worker sessions, stop any endpoint idle past its timeout.

    Only touches workers currently READY — one mid-provisioning or already
    shutting down is left alone regardless of last_activity_at.

    Part 6 / D5: TOCTOU fix — immediately before calling stop_endpoint, re-fetch
    the session from the DB and re-check status==READY AND idle >= timeout.
    A session might be claimed/touched between the scan read and the stop call.
    """
    stopped = 0
    for session in await db.list_active_worker_sessions():
        if session["worker_status"] != WorkerStatus.READY:
            continue
        idle = seconds_since(session["last_activity_at"])
        if idle >= session["idle_timeout_seconds"]:
            # Re-fetch immediately before stop (TOCTOU protection)
            fresh_session = await db.get_worker_session(session["session_id"])
            if not fresh_session or fresh_session["worker_status"] != WorkerStatus.READY:
                # Session no longer exists or status changed (was claimed) — skip it
                nebius_log.info(
                    "Idle scan re-check skipped (session status changed) — session_id=%s",
                    session["session_id"],
                )
                continue

            fresh_idle = seconds_since(fresh_session["last_activity_at"])
            if fresh_idle < fresh_session["idle_timeout_seconds"]:
                # Activity touched between scan read and stop call — skip it
                nebius_log.info(
                    "Idle scan re-check skipped (worker touched) — session_id=%s idle_seconds=%.0f",
                    session["session_id"], fresh_idle,
                )
                continue

            # Still idle and READY — proceed with stop
            try:
                await endpoints_client.stop_endpoint(fresh_session["nebius_endpoint_id"])
            except NebiusEndpointError as exc:
                # Real incident, 2026-07-14: user deleted this endpoint
                # manually (Nebius console) outside the app. Every scan
                # after that hit this exact "NotFound" error, which
                # previously propagated straight to run_forever()'s
                # generic except — the DB update below never ran, so
                # worker_status stayed READY forever even though the
                # endpoint was long gone. That's what made stop_training()
                # believe the (already-deleted) endpoint was still there
                # to proxy a stop request to, permanently stranding any
                # run that had been using it. An endpoint that's already
                # gone is, for our purposes, already stopped — treat
                # NotFound as success and update the DB anyway. Any other
                # CLI failure (real network/auth error) still re-raises,
                # so it's still visible in the log instead of being
                # silently swallowed. See docs/DESIGN_DECISIONS.md.
                if "notfound" not in str(exc).lower():
                    raise
                nebius_log.warning(
                    "Idle timeout — endpoint already gone (deleted outside the app?), "
                    "marking stopped anyway — session_id=%s endpoint_id=%s: %s",
                    fresh_session["session_id"], fresh_session["nebius_endpoint_id"], exc,
                )
            await db.update_worker_session(fresh_session["session_id"], worker_status=WorkerStatus.STOPPED)
            nebius_log.info(
                "Idle timeout — stopped worker session_id=%s endpoint_id=%s idle_seconds=%.0f timeout=%d",
                fresh_session["session_id"], fresh_session["nebius_endpoint_id"], fresh_idle,
                fresh_session["idle_timeout_seconds"],
            )
            stopped += 1
    return stopped


async def run_forever() -> None:
    """Loop entry point — call as a background asyncio task from app lifespan."""
    while True:
        await asyncio.sleep(settings.idle_scan_interval_seconds)
        try:
            # Part 5: Reconcile worker sessions to live state first, then stop idle ones
            await reconcile_worker_sessions()
            await stop_idle_workers()
        except Exception as exc:
            nebius_log.error("Idle monitor scan failed: %s", exc)
