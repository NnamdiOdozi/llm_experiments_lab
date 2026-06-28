"""Training run orchestrator — launches and manages subprocess workers.

Replaced the old threaded runner. Training now runs in separate processes
that communicate via files (status.json, metrics.jsonl, flag files).
"""

import ctypes
import ctypes.util
import json
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from backend.training import artifacts
from backend.db import sync_update_training_run
from backend.logging_config import training_log
from config.settings import settings


def _set_pdeathsig():
    """preexec_fn: worker receives SIGTERM when parent API process dies (Linux only)."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except (OSError, TypeError):
        pass  # Non-Linux — skip silently


# Keep RunStatus constants for backward compatibility with training.py imports
class RunStatus:
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


@dataclass
class ActiveRun:
    """Tracks a subprocess training worker."""
    run_id: int
    experiment_id: int
    process: subprocess.Popen
    run_dir: Path
    device: str
    template_key: str


# Global registry — tracks live worker processes
active_runs: dict[int, ActiveRun] = {}


def _cleanup_finished():
    """Remove finished processes from active_runs."""
    to_remove = []
    for run_id, run in active_runs.items():
        if run.process.poll() is not None:
            # Process exited — check if status was set properly
            status = artifacts.read_status(run_id)
            if status and status.get("status") not in ("completed", "failed", "cancelled", "paused"):
                artifacts.write_status(run_id, {**status, "status": "failed"})
                sync_update_training_run(run_id, status="failed",
                                         error_message="Worker process exited unexpectedly")
                training_log.warning("Worker died unexpectedly run_id=%d exit_code=%d",
                                     run_id, run.process.returncode)
            to_remove.append(run_id)
    for run_id in to_remove:
        del active_runs[run_id]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def start_run(run_id: int, experiment_id: int, config: dict, device: str = "cpu") -> None:
    """Write config and launch training subprocess."""
    _cleanup_finished()

    template_key = config.get("template", "transformer")
    rd = artifacts.run_dir(run_id)

    # Write config for worker
    worker_config = {**config, "run_id": run_id, "experiment_id": experiment_id, "device": device}
    (rd / "config.json").write_text(json.dumps(worker_config, indent=2))

    # Clean any stale flags
    artifacts.remove_flag(run_id, "stop")
    artifacts.remove_flag(run_id, "pause")

    # Launch worker subprocess — dies with parent via prctl
    proc = subprocess.Popen(
        [sys.executable, "-m", "backend.training.train_worker", "--run-dir", str(rd)],
        cwd=str(_project_root()),
        preexec_fn=_set_pdeathsig,
    )

    active_runs[run_id] = ActiveRun(
        run_id=run_id,
        experiment_id=experiment_id,
        process=proc,
        run_dir=rd,
        device=device,
        template_key=template_key,
    )

    training_log.info(
        "LAUNCHED worker run_id=%d pid=%d template=%s device=%s",
        run_id, proc.pid, template_key, device,
    )


def pause_run(run_id: int) -> bool:
    _cleanup_finished()
    run = active_runs.get(run_id)
    if run is None or run.process.poll() is not None:
        return False
    status = artifacts.read_status(run_id)
    if status is None or status.get("status") != "running":
        return False
    artifacts.write_flag(run_id, "pause")
    return True


def resume_run(run_id: int) -> bool:
    """Resume from checkpoint — launch new worker subprocess."""
    _cleanup_finished()
    status = artifacts.read_status(run_id)
    if status is None or status.get("status") != "paused":
        return False

    rd = artifacts.run_dir(run_id)

    # Clean flags
    artifacts.remove_flag(run_id, "pause")
    artifacts.remove_flag(run_id, "stop")

    # Read config for metadata
    config = json.loads((rd / "config.json").read_text())

    # Launch new worker with --resume — dies with parent via prctl
    proc = subprocess.Popen(
        [sys.executable, "-m", "backend.training.train_worker",
         "--run-dir", str(rd), "--resume"],
        cwd=str(_project_root()),
        preexec_fn=_set_pdeathsig,
    )

    active_runs[run_id] = ActiveRun(
        run_id=run_id,
        experiment_id=config.get("experiment_id", 0),
        process=proc,
        run_dir=rd,
        device=config.get("device", "cpu"),
        template_key=config.get("template", "transformer"),
    )

    training_log.info(
        "RESUMED worker run_id=%d pid=%d (from checkpoint)",
        run_id, proc.pid,
    )
    return True


def _force_stop_worker(run_id: int, process: subprocess.Popen):
    """Background thread: wait for cooperative stop, then terminate/kill if needed."""
    grace = settings.stop_grace_seconds
    kill_timeout = settings.stop_kill_seconds

    # Stage 1: wait for cooperative exit
    try:
        process.wait(timeout=grace)
        training_log.info("STOP cooperative exit run_id=%d", run_id)
        return
    except subprocess.TimeoutExpired:
        pass

    # Stage 2: SIGTERM
    training_log.warning("STOP force terminate run_id=%d (no exit after %ds)", run_id, grace)
    process.terminate()
    try:
        process.wait(timeout=kill_timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    # Stage 3: SIGKILL
    training_log.warning("STOP force kill run_id=%d (no exit after terminate)", run_id)
    process.kill()
    process.wait(timeout=5)


def stop_run(run_id: int) -> bool:
    run = active_runs.get(run_id)
    if run is not None:
        # Live process — signal via flag, then escalate in background
        artifacts.write_flag(run_id, "stop")
        artifacts.remove_flag(run_id, "pause")
        threading.Thread(
            target=_force_stop_worker,
            args=(run_id, run.process),
            daemon=True,
        ).start()
        return True

    # Process already exited (paused) — update status directly
    status = artifacts.read_status(run_id)
    if status and status.get("status") == "paused":
        status["status"] = "cancelled"
        artifacts.write_status(run_id, status)
        sync_update_training_run(run_id, status="cancelled")
        training_log.info("CANCELLED paused run_id=%d (no live process)", run_id)
        return True

    return False


def prompt_paused_model(run_id: int, prompt_text: str, max_new_tokens: int = 200) -> str | None:
    """Load checkpoint into API process, run inference, cleanup."""
    status = artifacts.read_status(run_id)
    if status is None or status.get("status") != "paused":
        return None

    rd = artifacts.run_dir(run_id)
    cp_path = artifacts.checkpoint_path(run_id)
    if not cp_path.exists():
        return None

    config = json.loads((rd / "config.json").read_text())
    template_key = config.get("template", "transformer")
    device = config.get("device", "cpu")

    import torch
    from backend.training.templates import TEMPLATE_REGISTRY

    # Load checkpoint (weights_only=False needed for optimizer state in checkpoint)
    cp = torch.load(cp_path, map_location=device, weights_only=False)
    # Use config from checkpoint — it has runtime updates (e.g. RNN vocab_size)
    model_config = cp.get("config", config)
    model = TEMPLATE_REGISTRY[template_key]["build_model"](model_config).to(device)
    model.load_state_dict(cp["model_state"])
    model.train(False)

    result = None

    if template_key in ("transformer", "moe"):
        from backend.training.templates.transformer.data import load_tiny_shakespeare, CharDataset
        text = load_tiny_shakespeare()
        dataset = CharDataset(text, config["model"]["block_size"], 1)
        encoded = dataset.encode(prompt_text)
        idx = torch.tensor([encoded], dtype=torch.long, device=device)
        with torch.no_grad():
            output = model.generate(idx, max_new_tokens=max_new_tokens)
        result = dataset.decode(output[0].tolist())

    elif template_key == "rnn":
        from backend.training.templates.rnn.data import load_dinos_dataset
        dataset = load_dinos_dataset(config["training"].get("seq_len", 50))
        try:
            result = model.generate(
                dataset.id_to_token,
                dataset.token_to_id,
                prefix=prompt_text.lower(),
                max_new_tokens=max_new_tokens,
                device=device,
            )
        except KeyError:
            result = "[Error: prompt contains characters not in vocabulary. Use lowercase letters only.]"

    del model, cp
    training_log.info("PROMPT run_id=%d template=%s prompt='%s'", run_id, template_key, prompt_text[:50])
    return result


def get_run_status(run_id: int) -> dict | None:
    """Read status from worker's status.json file."""
    _cleanup_finished()
    status = artifacts.read_status(run_id)

    if status is not None:
        # Check if process died unexpectedly
        run = active_runs.get(run_id)
        if run and run.process.poll() is not None:
            if status.get("status") not in ("completed", "failed", "cancelled", "paused"):
                status["status"] = "failed"
                artifacts.write_status(run_id, status)
                sync_update_training_run(run_id, status="failed",
                                         error_message="Worker process died unexpectedly")
        return status

    # No status file yet — check if run just launched
    run = active_runs.get(run_id)
    if run is not None:
        return {
            "run_id": run_id,
            "status": "queued",
            "current_step": 0,
            "total_steps": 0,
            "metrics_count": 0,
            "template": run.template_key,
            "elapsed_seconds": 0,
        }

    return None


def shutdown_all_workers():
    """Cleanly stop all active workers — called during API shutdown."""
    for run_id, run in list(active_runs.items()):
        if run.process.poll() is None:
            artifacts.write_flag(run_id, "stop")
            training_log.info("SHUTDOWN stopping run_id=%d pid=%d", run_id, run.process.pid)
    # Give workers grace period to exit cooperatively
    deadline = time.time() + settings.stop_grace_seconds
    for run_id, run in list(active_runs.items()):
        remaining = max(0, deadline - time.time())
        try:
            run.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            run.process.terminate()
            training_log.warning("SHUTDOWN terminated run_id=%d", run_id)
    active_runs.clear()
