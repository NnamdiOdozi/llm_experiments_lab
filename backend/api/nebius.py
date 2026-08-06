"""Worker status + heartbeat — feeds the frontend's idle-timeout warning banner.

Not a general Nebius API surface; just enough for the UI to show "this
worker will be stopped in N minutes" and for "Remain logged in" to reset
the idle clock. See docs/NEBIUS_SERVERLESS_IMPLEMENTATION_PLAN.md.
"""

from fastapi import APIRouter

from backend import db
from backend.nebius import endpoints_client
from backend.nebius.idle_monitor import seconds_since
from backend.training.worker_status import device_type_for, session_id_for
from config.settings import settings

router = APIRouter(prefix="/api/nebius", tags=["nebius"])


def _warning_seconds(device_type: str) -> int:
    return settings.gpu_idle_warning_seconds if device_type == "gpu" else settings.cpu_idle_warning_seconds


def _configured_spec(device_type: str, gpu_flavor: str | None = None) -> dict:
    """What settings.py says a *future* create_endpoint call would request —
    NOT necessarily what's actually running (see actual_platform/actual_preset
    below and docs/DESIGN_DECISIONS.md §9). Shown on the landing/workspace
    pages so users can see the hardware without digging into config/settings.py.
    For GPU, returns H100 specs if gpu_flavor=='h100', else L40S specs."""
    if device_type == "gpu":
        if gpu_flavor == "h100":
            return {"platform": settings.nebius_gpu_h100_platform, "preset": settings.nebius_gpu_h100_preset}
        return {"platform": settings.nebius_gpu_platform, "preset": settings.nebius_gpu_preset}
    return {"platform": settings.nebius_cpu_platform, "preset": settings.nebius_cpu_preset}


@router.get("/workers/{device}")
async def get_worker_status(device: str, gpu_flavor: str | None = None):
    device_type = device_type_for(device)
    configured = _configured_spec(device_type, gpu_flavor)
    base = {
        "backend_mode": settings.training_backend,
        "configured_platform": configured["platform"],
        "configured_preset": configured["preset"],
    }
    session_id = session_id_for(device_type, gpu_flavor)
    session = await db.get_worker_session(session_id)
    if session is None:
        return {
            **base,
            "preset": None,
            "actual_platform": None,
            "worker_status": "none", "seconds_idle": None,
            "idle_timeout_seconds": None, "warning_seconds": None,
        }
    return {
        **base,
        # The endpoint's *actual* platform/preset, captured when it last became
        # READY — never settings.nebius_cpu_preset/nebius_gpu_preset, which is
        # only what a *future* create_endpoint call would request. A config
        # bump doesn't retroactively resize an already-running endpoint. See
        # docs/DESIGN_DECISIONS.md for the 2026-07-11 incident this fixed.
        "preset": session["actual_preset"],
        "actual_platform": session["actual_platform"],
        "worker_status": session["worker_status"],
        "seconds_idle": seconds_since(session["last_activity_at"]),
        "idle_timeout_seconds": session["idle_timeout_seconds"],
        "warning_seconds": _warning_seconds(device_type),
    }


@router.get("/workers/{device}/logs")
async def get_worker_logs(device: str, gpu_flavor: str | None = None):
    device_type = device_type_for(device)
    session = await db.get_worker_session(session_id_for(device_type, gpu_flavor))
    if session is None or session["nebius_endpoint_id"] is None:
        return {"logs": ""}
    logs = await endpoints_client.get_logs(session["nebius_endpoint_id"])
    return {"logs": logs}


@router.post("/workers/{device}/heartbeat")
async def heartbeat(device: str, gpu_flavor: str | None = None):
    device_type = device_type_for(device)
    session_id = session_id_for(device_type, gpu_flavor)
    if await db.get_worker_session(session_id) is not None:
        await db.touch_worker_session(session_id)
    return {"ok": True}
