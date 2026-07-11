"""Worker status + heartbeat — feeds the frontend's idle-timeout warning banner.

Not a general Nebius API surface; just enough for the UI to show "this
worker will be stopped in N minutes" and for "Remain logged in" to reset
the idle clock. See docs/NEBIUS_SERVERLESS_IMPLEMENTATION_PLAN.md.
"""

from fastapi import APIRouter

from backend import db
from backend.nebius.idle_monitor import seconds_since
from backend.training.worker_status import device_type_for, session_id_for
from config.settings import settings

router = APIRouter(prefix="/api/nebius", tags=["nebius"])


def _warning_seconds(device_type: str) -> int:
    return settings.gpu_idle_warning_seconds if device_type == "gpu" else settings.cpu_idle_warning_seconds


@router.get("/workers/{device}")
async def get_worker_status(device: str):
    device_type = device_type_for(device)
    session = await db.get_worker_session(session_id_for(device_type))
    if session is None:
        return {
            "worker_status": "none", "seconds_idle": None,
            "idle_timeout_seconds": None, "warning_seconds": None,
        }
    return {
        "worker_status": session["worker_status"],
        "seconds_idle": seconds_since(session["last_activity_at"]),
        "idle_timeout_seconds": session["idle_timeout_seconds"],
        "warning_seconds": _warning_seconds(device_type),
    }


@router.post("/workers/{device}/heartbeat")
async def heartbeat(device: str):
    session_id = session_id_for(device_type_for(device))
    if await db.get_worker_session(session_id) is not None:
        await db.touch_worker_session(session_id)
    return {"ok": True}
