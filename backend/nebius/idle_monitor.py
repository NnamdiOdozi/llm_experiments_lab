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
from backend.training.worker_status import WorkerStatus
from config.settings import settings


def seconds_since(timestamp_str: str) -> float:
    """Seconds elapsed since a SQLite CURRENT_TIMESTAMP string (UTC, no tz suffix)."""
    then = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - then).total_seconds()


async def stop_idle_workers() -> int:
    """Scan active worker sessions, stop any endpoint idle past its timeout.

    Only touches workers currently READY — one mid-provisioning or already
    shutting down is left alone regardless of last_activity_at.
    """
    stopped = 0
    for session in await db.list_active_worker_sessions():
        if session["worker_status"] != WorkerStatus.READY:
            continue
        idle = seconds_since(session["last_activity_at"])
        if idle >= session["idle_timeout_seconds"]:
            await endpoints_client.stop_endpoint(session["nebius_endpoint_id"])
            await db.update_worker_session(session["session_id"], worker_status=WorkerStatus.STOPPED)
            nebius_log.info(
                "Idle timeout — stopped worker session_id=%s endpoint_id=%s idle_seconds=%.0f timeout=%d",
                session["session_id"], session["nebius_endpoint_id"], idle, session["idle_timeout_seconds"],
            )
            stopped += 1
    return stopped


async def run_forever() -> None:
    """Loop entry point — call as a background asyncio task from app lifespan."""
    while True:
        await asyncio.sleep(settings.idle_scan_interval_seconds)
        try:
            await stop_idle_workers()
        except Exception as exc:
            nebius_log.error("Idle monitor scan failed: %s", exc)
