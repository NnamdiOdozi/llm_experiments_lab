"""Remote worker lifecycle status — a separate state machine from RunStatus.

RunStatus (backend/training/status.py) tracks one training run's subprocess
lifecycle within a worker. WorkerStatus tracks the remote Nebius endpoint
itself — the thing that can host zero or more runs across a session. Do not
reuse RunStatus names for worker state; the two overlap textually
(QUEUED/RUNNING/COMPLETED/etc. exist in both concepts) but describe different
entities, and conflating them was flagged as a bug risk during design review.
See docs/NEBIUS_SERVERLESS_IMPLEMENTATION_PLAN.md.
"""

from enum import StrEnum


def device_type_for(device: str) -> str:
    return "gpu" if device.startswith("cuda") else "cpu"


def session_id_for(device_type: str) -> str:
    return f"worker-{device_type}"


class WorkerStatus(StrEnum):
    NONE = "none"
    PROVISIONING = "provisioning"
    STARTING = "starting"
    READY = "ready"
    IDLE = "idle"
    BUSY = "busy"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"
    FAILED = "failed"


# Status groupings — used by idle-timeout scanning and DB queries
ACTIVE_WORKER_STATUSES = frozenset({
    WorkerStatus.PROVISIONING,
    WorkerStatus.STARTING,
    WorkerStatus.READY,
    WorkerStatus.IDLE,
    WorkerStatus.BUSY,
    WorkerStatus.SHUTTING_DOWN,
})

TERMINAL_WORKER_STATUSES = frozenset({
    WorkerStatus.STOPPED,
    WorkerStatus.FAILED,
})
