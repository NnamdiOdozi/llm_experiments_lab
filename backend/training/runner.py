"""Background-thread training runner with pause/resume/prompt support.

Dispatches to the correct template (transformer or rnn) based on config["template"].
"""

import json
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
from backend.training.templates.transformer.data import CharDataset, load_tiny_shakespeare
from backend.training.templates.rnn.data import DinosDataset, load_dinos_dataset
from backend.training.templates.rnn.model import one_hot_encode
from backend.db import sync_update_training_run
from backend.logging_config import training_log, error_log
from config.settings import settings


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


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


def _save_checkpoint(run: ActiveRun, step: int):
    torch.save({
        "model_state": run.model.state_dict(),
        "optimizer_state": run.optimizer.state_dict(),
        "step": step,
        "config": run.config,
    }, _checkpoint_path(run.run_id))


def _set_status(run: ActiveRun, status: RunStatus):
    """Update run status in memory and DB."""
    run.status = status
    updates = {"status": status.value, "current_step": run.current_step}
    if status == RunStatus.RUNNING:
        updates["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    elif status in (RunStatus.COMPLETED, RunStatus.FAILED):
        updates["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    sync_update_training_run(run.run_id, **updates)
    training_log.info(
        "STATUS run_id=%d %s → %s step=%d",
        run.run_id, "?", status.value, run.current_step,
    )


def _check_pause(run: ActiveRun, step: int) -> bool:
    """Handle pause/stop. Returns True if run should terminate."""
    if run.pause_requested.is_set():
        training_log.info("PAUSING run_id=%d at step=%d — saving checkpoint", run.run_id, step)
        _set_status(run, RunStatus.PAUSED)
        _save_checkpoint(run, step)
        while run.pause_requested.is_set() and not run.stop_flag:
            time.sleep(0.1)
        if run.stop_flag:
            training_log.info("STOPPED (from pause) run_id=%d at step=%d", run.run_id, step)
            return True
        training_log.info("RESUMING run_id=%d from step=%d", run.run_id, step)
        _set_status(run, RunStatus.RUNNING)
    return run.stop_flag


# ── Transformer training loop ──────────────────────────────────────

def _transformer_eval(model, dataset: CharDataset, device: str, num_iters: int) -> dict[str, float]:
    model.train(False)
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(num_iters)
        for k in range(num_iters):
            x, y = dataset.get_batch(split, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train(True)
    return out


def _train_transformer(run: ActiveRun):
    config = run.config
    train_cfg = config["training"]
    device = run.device

    text = load_tiny_shakespeare()
    run.dataset = CharDataset(text, config["model"]["block_size"], train_cfg["batch_size"])

    template = TEMPLATE_REGISTRY["transformer"]
    run.model = template["build_model"](config).to(device)
    run.optimizer = torch.optim.AdamW(run.model.parameters(), lr=train_cfg["learning_rate"])

    _set_status(run, RunStatus.RUNNING)
    max_iters = train_cfg["max_iters"]
    log_interval = train_cfg["eval_interval"]
    num_eval_iters = train_cfg.get("eval_iters", 200)

    torch.manual_seed(1337)

    for step in range(run.current_step, max_iters + 1):
        if _check_pause(run, step):
            return

        if step % log_interval == 0:
            losses = _transformer_eval(run.model, run.dataset, device, num_eval_iters)
            _write_metric(run, {
                "step": step,
                "train_loss": round(losses["train"], 4),
                "val_loss": round(losses["val"], 4),
            })

        xb, yb = run.dataset.get_batch("train", device)
        _, loss = run.model(xb, yb)
        run.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        run.optimizer.step()
        run.current_step = step

    _set_status(run, RunStatus.COMPLETED)
    _save_checkpoint(run, max_iters)


# ── RNN training loop ──────────────────────────────────────────────

def _train_rnn(run: ActiveRun):
    config = run.config
    train_cfg = config["training"]
    device = run.device

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
    run.optimizer = torch.optim.Adam(run.model.parameters(), lr=train_cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss().to(device)

    _set_status(run, RunStatus.RUNNING)
    epochs = train_cfg.get("epochs", 50)
    clip = train_cfg.get("clip", 5)
    print_every = train_cfg.get("print_every", 10)
    n_chars = dataset.vocab_size

    torch.manual_seed(1337)
    counter = 0

    for epoch in range(epochs):
        h = run.model.init_hidden(batch_size, device)

        for x, targets in train_loader:
            counter += 1

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
            run.current_step = counter

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
                })

    _set_status(run, RunStatus.COMPLETED)
    _save_checkpoint(run, counter)


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
    return {
        "run_id": run.run_id,
        "status": run.status.value,
        "current_step": run.current_step,
        "total_steps": total,
        "metrics_count": len(run.metrics),
        "template": run.template_key,
    }
