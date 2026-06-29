"""Shared run status constants — single source of truth for backend + DB.

Import from here instead of using raw status strings.
"""

from enum import StrEnum


class RunStatus(StrEnum):
    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    PAUSE_REQUESTED = "pause_requested"
    CHECKPOINTING = "checkpointing"
    PAUSED = "paused"
    RESUMING = "resuming"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Status groupings — used by DB queries, runner cleanup, UI logic
ACTIVE_STATUSES = frozenset({
    RunStatus.QUEUED,
    RunStatus.STARTING,
    RunStatus.RUNNING,
    RunStatus.PAUSE_REQUESTED,
    RunStatus.CHECKPOINTING,
    RunStatus.RESUMING,
})

TERMINAL_STATUSES = frozenset({
    RunStatus.COMPLETED,
    RunStatus.FAILED,
    RunStatus.CANCELLED,
})

# Statuses where the worker process has exited but run isn't "done"
PAUSED_STATUSES = frozenset({
    RunStatus.PAUSED,
})
