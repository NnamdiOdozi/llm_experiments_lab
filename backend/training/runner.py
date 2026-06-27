"""Background-thread training runner with pause/resume/prompt support.

Dispatches to the correct template (transformer or rnn) based on config["template"].
"""

import datetime
import importlib.metadata
import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from backend.training.templates import TEMPLATE_REGISTRY

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

from backend.training.templates.transformer.data import CharDataset, load_tiny_shakespeare
from backend.training.templates.rnn.data import DinosDataset, load_dinos_dataset
from backend.training.templates.rnn.model import one_hot_encode
from backend.db import sync_update_training_run
from backend.logging_config import training_log, error_log
from config.settings import settings


def _get_device_name(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
    except Exception:
        return "unknown"


def _get_package_versions() -> dict:
    pkgs = ["torch", "numpy", "fastapi", "uvicorn", "pydantic"]
    versions = {}
    for pkg in pkgs:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def _param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class RunStatus(str, Enum):
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
    """Mutable state for a training run that lives in memory."""
    run_id: int
    experiment_id: int
    config: dict
    template_key: str = "transformer"
    status: RunStatus = RunStatus.QUEUED
    model: Any = None
    dataset: Any = None
    optimizer: torch.optim.Optimizer | None = None
    current_step: int = 0
    metrics: list[dict] = field(default_factory=list)
    pause_requested: threading.Event = field(default_factory=threading.Event)
    stop_flag: bool = False
    thread: threading.Thread | None = None
    device: str = "cpu"
    started_at: float = 0.0


# Global registry of active runs (in-memory, Tier 1 only)
active_runs: dict[int, ActiveRun] = {}


def _metrics_path(run_id: int) -> Path:
    path = settings.data_dir / "runs" / str(run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path / "metrics.jsonl"


def _checkpoint_path(run_id: int) -> Path:
    path = settings.data_dir / "runs" / str(run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path / "checkpoint.pt"


def _write_metric(run: ActiveRun, metric_row: dict):
    """Append metric to in-memory list, disk file, and database."""
    metric_row["timestamp"] = datetime.datetime.now().isoformat()
    run.metrics.append(metric_row)
    with open(_metrics_path(run.run_id), "a") as f:
        f.write(json.dumps(metric_row) + "\n")

    # Persist to DB
    train_history = json.dumps([m for m in run.metrics if "train_loss" in m])
    val_history = json.dumps([m for m in run.metrics if "val_loss" in m])
    sync_update_training_run(
        run.run_id,
        current_step=run.current_step,
        train_loss_history=train_history,
        val_loss_history=val_history,
        final_train_loss=metric_row.get("train_loss"),
        final_val_loss=metric_row.get("val_loss"),
    )


def _write_run_meta(run: ActiveRun, template_key: str, dataset_name: str):
    """Write run_meta.json to run folder — self-contained run identity."""
    meta = {
        "run_id": run.run_id,
        "experiment_id": run.experiment_id,
        "template": template_key,
        "dataset": dataset_name,
        "device": run.device,
        "started_at": datetime.datetime.now().isoformat(),
        "seed": settings.random_seed,
        "param_count": _param_count(run.model),
        "config": run.config,
        "package_versions": _get_package_versions(),
        "git_commit": _get_git_commit(),
    }
    meta_path = settings.data_dir / "runs" / str(run.run_id) / "run_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def _save_checkpoint(run: ActiveRun, step: int):
    torch.save({
        "model_state": run.model.state_dict(),
        "optimizer_state": run.optimizer.state_dict(),
        "step": step,
        "config": run.config,
    }, _checkpoint_path(run.run_id))


def _set_status(run: ActiveRun, status: RunStatus):
    """Update run status in memory and DB."""
    old_status = run.status.value
    run.status = status
    updates = {"status": status.value, "current_step": run.current_step}
    if status == RunStatus.RUNNING:
        updates["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    elif status in (RunStatus.COMPLETED, RunStatus.FAILED):
        updates["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    sync_update_training_run(run.run_id, **updates)
    training_log.info(
        "STATUS run_id=%d %s → %s step=%d",
        run.run_id, old_status, status.value, run.current_step,
    )


def _check_pause(run: ActiveRun, step: int) -> bool:
    """Handle pause/stop. Returns True if run should terminate."""
    if run.pause_requested.is_set():
        _set_status(run, RunStatus.PAUSE_REQUESTED)
        training_log.info("PAUSING run_id=%d at step=%d — saving checkpoint", run.run_id, step)
        _set_status(run, RunStatus.CHECKPOINTING)
        _save_checkpoint(run, step)
        sync_update_training_run(run.run_id, checkpoint_path=str(_checkpoint_path(run.run_id)))
        _set_status(run, RunStatus.PAUSED)
        while run.pause_requested.is_set() and not run.stop_flag:
            time.sleep(0.1)
        if run.stop_flag:
            training_log.info("CANCELLED (from pause) run_id=%d at step=%d", run.run_id, step)
            _set_status(run, RunStatus.CANCELLED)
            return True
        _set_status(run, RunStatus.RESUMING)
        training_log.info("RESUMING run_id=%d from step=%d", run.run_id, step)
        _set_status(run, RunStatus.RUNNING)
    if run.stop_flag:
        _set_status(run, RunStatus.CANCELLED)
    return run.stop_flag


# ── Transformer training loop ──────────────────────────────────────

@torch.no_grad()
def _transformer_eval(model, dataset: CharDataset, device: str, num_iters: int) -> dict[str, float]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(num_iters)
        for k in range(num_iters):
            x, y = dataset.get_batch(split, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
            time.sleep(0.005)  # yield GIL 5ms so FastAPI can respond
        out[split] = losses.mean().item()
    model.train()
    return out


def _train_transformer(run: ActiveRun):
    config = run.config
    train_cfg = config["training"]
    device = run.device

    _set_status(run, RunStatus.STARTING)

    text = load_tiny_shakespeare()
    run.dataset = CharDataset(text, config["model"]["block_size"], train_cfg["batch_size"])

    template = TEMPLATE_REGISTRY["transformer"]
    run.model = template["build_model"](config).to(device)
    opt_cls = OPTIMIZERS.get(train_cfg.get("optimizer", "adamw"), torch.optim.AdamW)
    run.optimizer = opt_cls(run.model.parameters(), lr=train_cfg["learning_rate"])

    # Persist run metadata
    sync_update_training_run(run.run_id,
        config_snapshot=json.dumps(config),
        seed=settings.random_seed,
        template_key="transformer",
        dataset_name="tiny_shakespeare",
        metrics_path=str(_metrics_path(run.run_id)),
        device_name=_get_device_name(device),
        param_count=_param_count(run.model),
        package_versions=json.dumps(_get_package_versions()),
        git_commit=_get_git_commit(),
    )
    _write_run_meta(run, "transformer", "tiny_shakespeare")

    run.started_at = time.time()
    _set_status(run, RunStatus.RUNNING)
    max_iters = train_cfg["max_iters"]
    log_interval = train_cfg["eval_interval"]
    num_eval_iters = min(train_cfg.get("eval_iters", 10), 10)
    lr = train_cfg["learning_rate"]

    torch.manual_seed(settings.random_seed)

    for step in range(run.current_step, max_iters + 1):
        run.current_step = step

        if _check_pause(run, step):
            return

        xb, yb = run.dataset.get_batch("train", device)
        _, loss = run.model(xb, yb)
        run.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        run.optimizer.step()
        time.sleep(0.005)  # yield GIL 5ms so FastAPI can respond to status polls

        if step > 0 and step % log_interval == 0:
            losses = _transformer_eval(run.model, run.dataset, device, num_eval_iters)
            _write_metric(run, {
                "step": step,
                "train_loss": round(losses["train"], 4),
                "val_loss": round(losses["val"], 4),
                "learning_rate": lr,
                "elapsed_seconds": round(time.time() - run.started_at, 1),
                "param_count": _param_count(run.model),
            })

    _set_status(run, RunStatus.COMPLETED)
    cp = _checkpoint_path(run.run_id)
    _save_checkpoint(run, max_iters)
    sync_update_training_run(run.run_id, checkpoint_path=str(cp))


# ── RNN training loop ──────────────────────────────────────────────

def _train_rnn(run: ActiveRun):
    config = run.config
    train_cfg = config["training"]
    device = run.device

    _set_status(run, RunStatus.STARTING)

    seq_len = train_cfg.get("seq_len", 50)
    dataset = load_dinos_dataset(seq_len)
    run.dataset = dataset

    # Split into train/val (80/20)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    train_set = torch.utils.data.Subset(dataset, range(n_train))
    val_set = torch.utils.data.Subset(dataset, range(n_train, n_total))

    batch_size = train_cfg["batch_size"]
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)

    # Update vocab_size from actual data
    config["model"]["vocab_size"] = dataset.vocab_size

    template = TEMPLATE_REGISTRY["rnn"]
    run.model = template["build_model"](config).to(device)
    opt_cls = OPTIMIZERS.get(train_cfg.get("optimizer", "adam"), torch.optim.Adam)
    run.optimizer = opt_cls(run.model.parameters(), lr=train_cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss().to(device)

    # Persist run metadata
    sync_update_training_run(run.run_id,
        config_snapshot=json.dumps(config),
        seed=settings.random_seed,
        template_key="rnn",
        dataset_name="dinos",
        metrics_path=str(_metrics_path(run.run_id)),
        device_name=_get_device_name(device),
        param_count=_param_count(run.model),
        package_versions=json.dumps(_get_package_versions()),
        git_commit=_get_git_commit(),
    )
    _write_run_meta(run, "rnn", "dinos")

    run.started_at = time.time()
    _set_status(run, RunStatus.RUNNING)
    epochs = train_cfg.get("epochs", 50)
    clip = train_cfg.get("clip", 5)
    print_every = train_cfg.get("print_every", 10)
    n_chars = dataset.vocab_size
    lr = train_cfg["learning_rate"]

    torch.manual_seed(settings.random_seed)
    counter = 0

    for epoch in range(epochs):
        h = run.model.init_hidden(batch_size, device)

        for x, targets in train_loader:
            counter += 1
            run.current_step = counter

            if _check_pause(run, counter):
                return

            x_encoded = one_hot_encode(x, n_chars)
            inputs = torch.from_numpy(x_encoded).to(device)
            targets = targets.to(device)

            h = tuple(each.data for each in h)
            run.model.zero_grad()
            output, h = run.model(inputs, h)
            loss = criterion(output, targets.view(batch_size * seq_len))
            loss.backward()
            nn.utils.clip_grad_norm_(run.model.parameters(), clip)
            run.optimizer.step()
            time.sleep(0.005)  # yield GIL 5ms

            if counter % print_every == 0:
                # Validation
                val_h = run.model.init_hidden(batch_size, device)
                val_losses = []
                run.model.train(False)
                for vx, vy in val_loader:
                    vx_enc = one_hot_encode(vx, n_chars)
                    vx_t = torch.from_numpy(vx_enc).to(device)
                    vy = vy.to(device)
                    val_h = tuple(each.data for each in val_h)
                    vout, val_h = run.model(vx_t, val_h)
                    vloss = criterion(vout, vy.view(batch_size * seq_len))
                    val_losses.append(vloss.item())
                run.model.train(True)

                _write_metric(run, {
                    "step": counter,
                    "epoch": epoch + 1,
                    "train_loss": round(loss.item(), 4),
                    "val_loss": round(float(np.mean(val_losses)), 4),
                    "learning_rate": lr,
                    "elapsed_seconds": round(time.time() - run.started_at, 1),
                    "param_count": _param_count(run.model),
                })

    _set_status(run, RunStatus.COMPLETED)
    cp = _checkpoint_path(run.run_id)
    _save_checkpoint(run, counter)
    sync_update_training_run(run.run_id, checkpoint_path=str(cp))


# ── Public API ──────────────────────────────────────────────────────

TRAIN_DISPATCHERS = {
    "transformer": _train_transformer,
    "rnn": _train_rnn,
}


def _train_loop(run: ActiveRun):
    """Dispatch to the correct training loop based on template."""
    try:
        dispatcher = TRAIN_DISPATCHERS.get(run.template_key)
        if dispatcher is None:
            raise ValueError(f"Unknown template: {run.template_key}")
        dispatcher(run)
    except Exception as e:
        _set_status(run, RunStatus.FAILED)
        run.metrics.append({"error": str(e)})
        sync_update_training_run(run.run_id, error_message=str(e))
        error_log.error(
            "Training FAILED run_id=%d template=%s step=%d: %s",
            run.run_id, run.template_key, run.current_step, e,
            exc_info=True,
        )


def start_run(run_id: int, experiment_id: int, config: dict, device: str = "cpu") -> ActiveRun:
    """Start a training run in a background thread."""
    template_key = config.get("template", "transformer")
    run = ActiveRun(
        run_id=run_id,
        experiment_id=experiment_id,
        config=config,
        template_key=template_key,
        device=device,
    )
    active_runs[run_id] = run

    thread = threading.Thread(target=_train_loop, args=(run,), daemon=True)
    run.thread = thread
    thread.start()
    return run


def pause_run(run_id: int) -> bool:
    run = active_runs.get(run_id)
    if run is None or run.status != RunStatus.RUNNING:
        return False
    run.pause_requested.set()
    return True


def resume_run(run_id: int) -> bool:
    run = active_runs.get(run_id)
    if run is None or run.status != RunStatus.PAUSED:
        return False
    run.pause_requested.clear()
    return True


def stop_run(run_id: int) -> bool:
    run = active_runs.get(run_id)
    if run is None:
        return False
    run.stop_flag = True
    run.pause_requested.clear()
    return True


def prompt_paused_model(run_id: int, prompt_text: str, max_new_tokens: int = 200) -> str | None:
    """Run inference on a paused model. Returns generated text or None."""
    run = active_runs.get(run_id)
    if run is None or run.status != RunStatus.PAUSED or run.model is None:
        return None

    if run.template_key == "transformer":
        encoded = run.dataset.encode(prompt_text)
        idx = torch.tensor([encoded], dtype=torch.long, device=run.device)
        output = run.model.generate(idx, max_new_tokens=max_new_tokens)
        return run.dataset.decode(output[0].tolist())

    elif run.template_key == "rnn":
        try:
            return run.model.generate(
                run.dataset.id_to_token,
                run.dataset.token_to_id,
                prefix=prompt_text.lower(),
                max_new_tokens=max_new_tokens,
                device=run.device,
            )
        except KeyError:
            return f"[Error: prompt contains characters not in vocabulary. Use lowercase letters only.]"

    return None


def get_run_status(run_id: int) -> dict | None:
    run = active_runs.get(run_id)
    if run is None:
        return None
    total = run.config["training"].get("max_iters", run.config["training"].get("epochs", 0) * 100)
    elapsed = time.time() - run.started_at if run.started_at > 0 else 0
    return {
        "run_id": run.run_id,
        "status": run.status.value,
        "current_step": run.current_step,
        "total_steps": total,
        "metrics_count": len(run.metrics),
        "template": run.template_key,
        "elapsed_seconds": round(elapsed, 1),
    }
