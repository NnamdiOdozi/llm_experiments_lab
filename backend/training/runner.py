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
from backend.training.status import RunStatus, ACTIVE_STATUSES, TERMINAL_STATUSES, PAUSED_STATUSES
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


def _popen_kwargs() -> dict:
    """Platform-specific subprocess.Popen kwargs for child process management.

    Linux/WSL: preexec_fn with prctl(PR_SET_PDEATHSIG) — worker auto-dies when parent exits.
    Windows: CREATE_NEW_PROCESS_GROUP — worker gets own group for clean signaling.
    """
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"preexec_fn": _set_pdeathsig}


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
            if status and status.get("status") not in TERMINAL_STATUSES | PAUSED_STATUSES:
                artifacts.write_status(run_id, {**status, "status": RunStatus.FAILED})
                sync_update_training_run(run_id, status=RunStatus.FAILED,
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

    # Reset stale artifacts from previous run in same directory so the
    # WebSocket/poller doesn't briefly serve old status or metrics.
    artifacts.write_status(run_id, {"status": RunStatus.QUEUED, "current_step": 0, "total_steps": 0})
    metrics_file = artifacts.metrics_path(run_id)
    if metrics_file.exists():
        metrics_file.write_text("")

    # Launch worker subprocess — platform-aware process management
    proc = subprocess.Popen(
        [sys.executable, "-m", "backend.training.train_worker", "--run-dir", str(rd)],
        cwd=str(_project_root()),
        **_popen_kwargs(),
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
    if status is None or status.get("status") != RunStatus.RUNNING:
        return False
    artifacts.write_flag(run_id, "pause")
    return True


def resume_run(run_id: int, updated_config: dict | None = None) -> bool:
    """Resume from checkpoint — launch new worker subprocess.

    If updated_config is provided (from DB), hot-reloadable fields like
    max_iters and eval_interval are merged into the run's config.json
    so the new worker picks them up.
    """
    _cleanup_finished()
    status = artifacts.read_status(run_id)
    if status is None or status.get("status") != RunStatus.PAUSED:
        return False

    rd = artifacts.run_dir(run_id)

    # Clean flags
    artifacts.remove_flag(run_id, "pause")
    artifacts.remove_flag(run_id, "stop")

    # Read config for metadata
    config = json.loads((rd / "config.json").read_text())

    # Apply user edits made while paused (e.g. max_iters, eval_interval)
    if updated_config:
        for section in ("training", "inference"):
            if section in updated_config:
                config[section] = {**config.get(section, {}), **updated_config[section]}
        (rd / "config.json").write_text(json.dumps(config, indent=2))

    # Launch new worker with --resume — platform-aware process management
    proc = subprocess.Popen(
        [sys.executable, "-m", "backend.training.train_worker",
         "--run-dir", str(rd), "--resume"],
        cwd=str(_project_root()),
        **_popen_kwargs(),
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


def stop_run(run_id: int, db_status: str | None = None) -> bool:
    """db_status: the run's status per the DB row (caller already has it
    from stop_training's own db.get_training_run call) — used as a
    fallback when the on-disk status.json is missing or doesn't say
    PAUSED. Real bug found 2026-07-14: two legacy run directories
    (26, 27; predating status.json tracking, only had checkpoint.pt/
    metrics.jsonl/run_meta.json) had no status.json at all, so
    artifacts.read_status() returned None and this always fell through to
    "Run not found" — the DB itself said paused (what the user actually
    sees in Open Runs), but that was never consulted. See
    docs/DESIGN_DECISIONS.md.
    """
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
    if status and status.get("status") == RunStatus.PAUSED:
        status["status"] = RunStatus.CANCELLED
        artifacts.write_status(run_id, status)
        sync_update_training_run(run_id, status=RunStatus.CANCELLED)
        training_log.info("CANCELLED paused run_id=%d (no live process)", run_id)
        return True

    if status is None and db_status == RunStatus.PAUSED:
        artifacts.write_status(run_id, {"status": RunStatus.CANCELLED})
        sync_update_training_run(run_id, status=RunStatus.CANCELLED)
        training_log.info("CANCELLED paused run_id=%d (no live process, no status.json — DB said paused)", run_id)
        return True

    return False


def prompt_paused_model(run_id: int, prompt_text: str, max_new_tokens: int = 200) -> str | None:
    """Load checkpoint into API process, run inference, cleanup.

    Despite the name, this works for any run with a saved checkpoint, not
    just a paused one — it never touches the training subprocess, only the
    checkpoint file on disk. Every template saves a final checkpoint right
    before marking a run COMPLETED (see train_worker.py), so a finished
    run can be prompted exactly the same way a paused one already could.
    See docs/DESIGN_DECISIONS.md.
    """
    status = artifacts.read_status(run_id)
    if status is None or status.get("status") not in (RunStatus.PAUSED, RunStatus.COMPLETED):
        return None

    rd = artifacts.run_dir(run_id)
    cp_path = artifacts.checkpoint_path(run_id)
    if not cp_path.exists():
        return None

    config = json.loads((rd / "config.json").read_text())
    template_key = config.get("template", "transformer")
    device = config.get("device", "cpu")

    # Inference params (max_new_tokens, temperature) live in the experiment
    # config under the "inference" key — editable from the dashboard ConfigPanel.
    # Falls back to API-provided max_new_tokens / sensible defaults.
    inference_cfg = config.get("inference", {})
    max_tokens = inference_cfg.get("max_new_tokens", max_new_tokens)
    temperature = inference_cfg.get("temperature", 0.8)
    # "greedy" or "sample" (default) — same setting used everywhere decoding
    # happens (this Generate path, and step-through > / >> via
    # DiagnosticSession.decoding_mode), so behavior is consistent across
    # the whole app rather than diagnostics-only. See docs/DESIGN_DECISIONS.md.
    greedy = inference_cfg.get("decoding_mode", "sample") == "greedy"

    import torch
    from backend.training.templates import TEMPLATE_REGISTRY

    # Load checkpoint (weights_only=False needed for optimizer state in checkpoint)
    cp = torch.load(cp_path, map_location=device, weights_only=False)
    # Drop optimizer state immediately — not needed for inference and can
    # be large (same size as model weights for Adam).  Frees memory before
    # we allocate the model tensor buffers.
    cp.pop("optimizer_state", None)
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
            output = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, greedy=greedy)
        result = dataset.decode(output[0].tolist())

    elif template_key == "rnn":
        from backend.training.templates.rnn.data import load_dinos_dataset
        dataset = load_dinos_dataset(config["training"].get("seq_len", 50))
        try:
            result = model.generate(
                dataset.id_to_token,
                dataset.token_to_id,
                prefix=prompt_text.lower(),
                max_new_tokens=max_tokens,
                device=device,
                temperature=temperature,
                greedy=greedy,
            )
        except KeyError:
            result = "[Error: prompt contains characters not in vocabulary. Use lowercase letters only.]"

    del model, cp
    # Free CUDA cached memory so the resume worker subprocess can allocate.
    # Without this, the API process keeps a CUDA context that blocks the
    # worker from fitting the training model on the same GPU.
    if device != "cpu":
        import torch
        torch.cuda.empty_cache()
    training_log.info(
        "PROMPT run_id=%d template=%s prompt='%s' max_tokens=%d temperature=%.2f decoding_mode=%s",
        run_id, template_key, prompt_text[:50], max_tokens, temperature, "greedy" if greedy else "sample",
    )
    return result


def get_run_status(run_id: int) -> dict | None:
    """Read status from worker's status.json file."""
    _cleanup_finished()
    status = artifacts.read_status(run_id)

    if status is not None:
        # Check if process died unexpectedly
        run = active_runs.get(run_id)
        if run and run.process.poll() is not None:
            if status.get("status") not in TERMINAL_STATUSES | PAUSED_STATUSES:
                status["status"] = RunStatus.FAILED
                artifacts.write_status(run_id, status)
                sync_update_training_run(run_id, status=RunStatus.FAILED,
                                         error_message="Worker process died unexpectedly")
        # This whole function only ever reads a local subprocess worker's own
        # status.json — it has no concept of remote execution, so this is
        # always "local" from here. See docs/DESIGN_DECISIONS.md §10.
        status["execution_backend"] = "local"
        return status

    # No status file yet — check if run just launched
    run = active_runs.get(run_id)
    if run is not None:
        return {
            "run_id": run_id,
            "status": RunStatus.QUEUED,
            "current_step": 0,
            "total_steps": 0,
            "metrics_count": 0,
            "template": run.template_key,
            "elapsed_seconds": 0,
            "execution_backend": "local",
            "device": run.device,
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
