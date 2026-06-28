"""Standalone training worker — runs as a subprocess.

Usage: python -m backend.training.train_worker --run-dir data/runs/42
"""

import argparse
import datetime
import importlib.metadata
import json
import os
import subprocess as sp
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from backend.training import artifacts
from backend.training.templates import TEMPLATE_REGISTRY
from backend.training.templates.transformer.data import CharDataset, load_tiny_shakespeare
from backend.training.templates.rnn.data import DinosDataset, load_dinos_dataset
from backend.training.templates.rnn.model import one_hot_encode
from backend.db import sync_update_training_run
from config.settings import settings

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


def _get_device_name(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def _get_git_commit() -> str:
    try:
        return sp.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=sp.DEVNULL, timeout=5,
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


# ── Worker state ────────────────────────────────────────────────────


class WorkerState:
    """Mutable state for the training worker process."""

    def __init__(self, run_id: int, config: dict, device: str, resume: bool = False):
        self.run_id = run_id
        self.config = config
        self.device = device
        self.template_key = config.get("template", "transformer")
        self.model = None
        self.optimizer = None
        self.dataset = None
        self.current_step = 0
        self.started_at = 0.0
        self.metrics: list[dict] = []
        self.resume = resume
        self.checkpoint_extra: dict = {}  # extra fields to include in checkpoint (e.g. epoch)

    def _total_steps(self) -> int:
        t = self.config.get("training", {})
        return t.get("max_iters", t.get("epochs", 0) * 100)

    def set_status(self, status: str):
        elapsed = time.time() - self.started_at if self.started_at > 0 else 0
        artifacts.write_status(self.run_id, {
            "run_id": self.run_id,
            "status": status,
            "current_step": self.current_step,
            "total_steps": self._total_steps(),
            "metrics_count": len(self.metrics),
            "template": self.template_key,
            "elapsed_seconds": round(elapsed, 1),
            "pid": os.getpid(),
        })
        updates: dict = {"status": status, "current_step": self.current_step}
        if status == "running":
            updates["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        elif status in ("completed", "failed"):
            updates["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        sync_update_training_run(self.run_id, **updates)

    def update_progress(self):
        """Update status.json with current step/metrics count without changing status."""
        elapsed = time.time() - self.started_at if self.started_at > 0 else 0
        status = artifacts.read_status(self.run_id)
        current_status = status["status"] if status else "running"
        artifacts.write_status(self.run_id, {
            "run_id": self.run_id,
            "status": current_status,
            "current_step": self.current_step,
            "total_steps": self._total_steps(),
            "metrics_count": len(self.metrics),
            "template": self.template_key,
            "elapsed_seconds": round(elapsed, 1),
            "pid": os.getpid(),
        })

    def write_metric(self, metric_row: dict):
        metric_row["timestamp"] = datetime.datetime.now().isoformat()
        self.metrics.append(metric_row)
        with open(artifacts.metrics_path(self.run_id), "a") as f:
            f.write(json.dumps(metric_row) + "\n")
        train_history = json.dumps([m for m in self.metrics if "train_loss" in m])
        val_history = json.dumps([m for m in self.metrics if "val_loss" in m])
        sync_update_training_run(
            self.run_id,
            current_step=self.current_step,
            train_loss_history=train_history,
            val_loss_history=val_history,
            final_train_loss=metric_row.get("train_loss"),
            final_val_loss=metric_row.get("val_loss"),
        )
        self.update_progress()

    def save_checkpoint(self, step: int, **extra):
        data = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step": step,
            "config": self.config,
            **extra,
        }
        torch.save(data, artifacts.checkpoint_path(self.run_id))

    def load_checkpoint(self) -> dict:
        """Load model/optimizer state from checkpoint. Call after model+optimizer are built.
        Returns the full checkpoint dict for callers that need extra fields (e.g. epoch).
        """
        cp_path = artifacts.checkpoint_path(self.run_id)
        cp = torch.load(cp_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(cp["model_state"])
        self.optimizer.load_state_dict(cp["optimizer_state"])
        self.current_step = cp["step"]
        # Reload previous metrics from disk
        mf = artifacts.metrics_path(self.run_id)
        if mf.exists():
            with open(mf) as f:
                self.metrics = [json.loads(line) for line in f if line.strip()]
        return cp

    def check_pause(self, step: int) -> bool:
        """Check pause/stop flags. Returns True if run should terminate."""
        if artifacts.has_flag(self.run_id, "pause"):
            self.set_status("pause_requested")
            self.set_status("checkpointing")
            self.save_checkpoint(step, **self.checkpoint_extra)
            sync_update_training_run(
                self.run_id,
                checkpoint_path=str(artifacts.checkpoint_path(self.run_id)),
            )
            self.set_status("paused")
            # Exit process — resume will launch a new worker from checkpoint
            sys.exit(0)
        if artifacts.has_flag(self.run_id, "stop"):
            self.set_status("cancelled")
            return True
        return False

    def write_run_meta(self, dataset_name: str):
        meta = {
            "run_id": self.run_id,
            "template": self.template_key,
            "dataset": dataset_name,
            "device": self.device,
            "started_at": datetime.datetime.now().isoformat(),
            "seed": settings.random_seed,
            "param_count": _param_count(self.model),
            "config": self.config,
            "package_versions": _get_package_versions(),
            "git_commit": _get_git_commit(),
        }
        meta_path = artifacts.run_dir(self.run_id) / "run_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


# ── Transformer ─────────────────────────────────────────────────────


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
        out[split] = losses.mean().item()
    model.train()
    return out


def train_transformer(ws: WorkerState):
    config = ws.config
    train_cfg = config["training"]
    device = ws.device

    ws.set_status("starting")

    text = load_tiny_shakespeare()
    ws.dataset = CharDataset(text, config["model"]["block_size"], train_cfg["batch_size"])

    template = TEMPLATE_REGISTRY["transformer"]
    ws.model = template["build_model"](config).to(device)
    opt_cls = OPTIMIZERS.get(train_cfg.get("optimizer", "adamw"), torch.optim.AdamW)
    ws.optimizer = opt_cls(ws.model.parameters(), lr=train_cfg["learning_rate"])

    if ws.resume:
        ws.load_checkpoint()

    sync_update_training_run(ws.run_id,
        config_snapshot=json.dumps(config),
        seed=settings.random_seed,
        template_key="transformer",
        dataset_name="tiny_shakespeare",
        metrics_path=str(artifacts.metrics_path(ws.run_id)),
        device_name=_get_device_name(device),
        param_count=_param_count(ws.model),
        package_versions=json.dumps(_get_package_versions()),
        git_commit=_get_git_commit(),
    )
    ws.write_run_meta("tiny_shakespeare")

    ws.started_at = time.time()
    ws.set_status("running")
    max_iters = train_cfg["max_iters"]
    log_interval = train_cfg["eval_interval"]
    num_eval_iters = min(train_cfg.get("eval_iters", 10), 10)
    lr = train_cfg["learning_rate"]

    torch.manual_seed(settings.random_seed)
    # Checkpoint is saved AFTER completing step N, so resume from N+1
    start_step = ws.current_step + 1 if ws.resume else 0

    for step in range(start_step, max_iters + 1):
        ws.current_step = step

        xb, yb = ws.dataset.get_batch("train", device)
        _, loss = ws.model(xb, yb)
        ws.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        ws.optimizer.step()

        # Check pause/stop AFTER training step so checkpoint = completed step
        if ws.check_pause(step):
            return

        if step > 0 and step % log_interval == 0:
            losses = _transformer_eval(ws.model, ws.dataset, device, num_eval_iters)
            ws.write_metric({
                "step": step,
                "train_loss": round(losses["train"], 4),
                "val_loss": round(losses["val"], 4),
                "learning_rate": lr,
                "elapsed_seconds": round(time.time() - ws.started_at, 1),
                "param_count": _param_count(ws.model),
            })
            ws.save_checkpoint(step)

    ws.save_checkpoint(max_iters)
    sync_update_training_run(ws.run_id, checkpoint_path=str(artifacts.checkpoint_path(ws.run_id)))
    ws.set_status("completed")


# ── MoE ─────────────────────────────────────────────────────────────


@torch.no_grad()
def _moe_eval(model, dataset, device: str, num_iters: int) -> dict[str, dict[str, float]]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(num_iters)
        drop_rates = torch.zeros(num_iters)
        for k in range(num_iters):
            x, y = dataset.get_batch(split, device)
            _, loss, drop_rate = model(x, y)
            losses[k] = loss.item()
            drop_rates[k] = drop_rate if isinstance(drop_rate, float) else drop_rate.item()
        out[split] = {"loss": losses.mean().item(), "drop_rate": drop_rates.mean().item()}
    model.train()
    return out


def train_moe(ws: WorkerState):
    config = ws.config
    train_cfg = config["training"]
    device = ws.device

    ws.set_status("starting")

    text = load_tiny_shakespeare()
    ws.dataset = CharDataset(text, config["model"]["block_size"], train_cfg["batch_size"])

    template = TEMPLATE_REGISTRY["moe"]
    ws.model = template["build_model"](config).to(device)
    opt_cls = OPTIMIZERS.get(train_cfg.get("optimizer", "adamw"), torch.optim.AdamW)
    ws.optimizer = opt_cls(ws.model.parameters(), lr=train_cfg["learning_rate"])

    if ws.resume:
        ws.load_checkpoint()

    sync_update_training_run(ws.run_id,
        config_snapshot=json.dumps(config),
        seed=settings.random_seed,
        template_key="moe",
        dataset_name="tiny_shakespeare",
        metrics_path=str(artifacts.metrics_path(ws.run_id)),
        device_name=_get_device_name(device),
        param_count=_param_count(ws.model),
        package_versions=json.dumps(_get_package_versions()),
        git_commit=_get_git_commit(),
    )
    ws.write_run_meta("tiny_shakespeare")

    ws.started_at = time.time()
    ws.set_status("running")
    max_iters = train_cfg["max_iters"]
    log_interval = train_cfg["eval_interval"]
    num_eval_iters = min(train_cfg.get("eval_iters", 10), 10)
    lr = train_cfg["learning_rate"]

    torch.manual_seed(settings.random_seed)
    start_step = ws.current_step + 1 if ws.resume else 0

    for step in range(start_step, max_iters + 1):
        ws.current_step = step

        xb, yb = ws.dataset.get_batch("train", device)
        _, loss, _ = ws.model(xb, yb)
        ws.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        ws.optimizer.step()

        # Check pause/stop AFTER training step so checkpoint = completed step
        if ws.check_pause(step):
            return

        if step > 0 and step % log_interval == 0:
            metrics = _moe_eval(ws.model, ws.dataset, device, num_eval_iters)
            ws.write_metric({
                "step": step,
                "train_loss": round(metrics["train"]["loss"], 4),
                "val_loss": round(metrics["val"]["loss"], 4),
                "train_drop_rate": round(metrics["train"]["drop_rate"] * 100, 1),
                "val_drop_rate": round(metrics["val"]["drop_rate"] * 100, 1),
                "learning_rate": lr,
                "elapsed_seconds": round(time.time() - ws.started_at, 1),
                "param_count": _param_count(ws.model),
            })
            ws.save_checkpoint(step)

    ws.save_checkpoint(max_iters)
    sync_update_training_run(ws.run_id, checkpoint_path=str(artifacts.checkpoint_path(ws.run_id)))
    ws.set_status("completed")


# ── RNN ─────────────────────────────────────────────────────────────


def train_rnn(ws: WorkerState):
    config = ws.config
    train_cfg = config["training"]
    device = ws.device

    ws.set_status("starting")

    seq_len = train_cfg.get("seq_len", 50)
    dataset = load_dinos_dataset(seq_len)
    ws.dataset = dataset

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    train_set = torch.utils.data.Subset(dataset, range(n_train))
    val_set = torch.utils.data.Subset(dataset, range(n_train, n_total))

    batch_size = train_cfg["batch_size"]
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=True,
    )

    config["model"]["vocab_size"] = dataset.vocab_size

    template = TEMPLATE_REGISTRY["rnn"]
    ws.model = template["build_model"](config).to(device)
    opt_cls = OPTIMIZERS.get(train_cfg.get("optimizer", "adam"), torch.optim.Adam)
    ws.optimizer = opt_cls(ws.model.parameters(), lr=train_cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss().to(device)

    if ws.resume:
        ws.load_checkpoint()

    sync_update_training_run(ws.run_id,
        config_snapshot=json.dumps(config),
        seed=settings.random_seed,
        template_key="rnn",
        dataset_name="dinos",
        metrics_path=str(artifacts.metrics_path(ws.run_id)),
        device_name=_get_device_name(device),
        param_count=_param_count(ws.model),
        package_versions=json.dumps(_get_package_versions()),
        git_commit=_get_git_commit(),
    )
    ws.write_run_meta("dinos")

    ws.started_at = time.time()
    ws.set_status("running")
    epochs = train_cfg.get("epochs", 50)
    clip = train_cfg.get("clip", 5)
    print_every = train_cfg.get("print_every", 10)
    n_chars = dataset.vocab_size
    lr = train_cfg["learning_rate"]

    torch.manual_seed(settings.random_seed)
    counter = 0
    start_epoch = 0
    resume_batch = 0  # batch index to resume from within the epoch
    if ws.resume:
        cp_path = artifacts.checkpoint_path(ws.run_id)
        cp_data = torch.load(cp_path, map_location="cpu", weights_only=False)
        start_epoch = cp_data.get("epoch", 0)
        resume_batch = cp_data.get("batch_in_epoch", 0)
        del cp_data
        counter = ws.current_step

    for epoch in range(start_epoch, epochs):
        # Deterministic shuffle per epoch — same seed reproduces same order
        epoch_gen = torch.Generator().manual_seed(settings.random_seed + epoch)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, drop_last=True,
            generator=epoch_gen,
        )
        ws.checkpoint_extra = {"epoch": epoch, "batch_in_epoch": 0}
        h = ws.model.init_hidden(batch_size, device)

        for batch_idx, (x, targets) in enumerate(train_loader):
            # Skip batches already completed in a resumed epoch
            if epoch == start_epoch and batch_idx < resume_batch:
                continue
            counter += 1
            ws.current_step = counter
            ws.checkpoint_extra = {"epoch": epoch, "batch_in_epoch": batch_idx + 1}

            x_encoded = one_hot_encode(x, n_chars)
            inputs = torch.from_numpy(x_encoded).to(device)
            targets = targets.to(device)

            h = tuple(each.data for each in h)
            ws.model.zero_grad()
            output, h = ws.model(inputs, h)
            loss = criterion(output, targets.view(batch_size * seq_len))
            loss.backward()
            nn.utils.clip_grad_norm_(ws.model.parameters(), clip)
            ws.optimizer.step()

            # Check pause/stop AFTER training step so checkpoint = completed step
            if ws.check_pause(counter):
                return

            if counter % print_every == 0:
                val_h = ws.model.init_hidden(batch_size, device)
                val_losses = []
                ws.model.train(False)
                for vx, vy in val_loader:
                    vx_enc = one_hot_encode(vx, n_chars)
                    vx_t = torch.from_numpy(vx_enc).to(device)
                    vy = vy.to(device)
                    val_h = tuple(each.data for each in val_h)
                    vout, val_h = ws.model(vx_t, val_h)
                    vloss = criterion(vout, vy.view(batch_size * seq_len))
                    val_losses.append(vloss.item())
                ws.model.train(True)

                ws.write_metric({
                    "step": counter,
                    "epoch": epoch + 1,
                    "train_loss": round(loss.item(), 4),
                    "val_loss": round(float(np.mean(val_losses)), 4),
                    "learning_rate": lr,
                    "elapsed_seconds": round(time.time() - ws.started_at, 1),
                    "param_count": _param_count(ws.model),
                })
                ws.save_checkpoint(counter, **ws.checkpoint_extra)

    ws.save_checkpoint(counter, **ws.checkpoint_extra)
    sync_update_training_run(ws.run_id, checkpoint_path=str(artifacts.checkpoint_path(ws.run_id)))
    ws.set_status("completed")


# ── Dispatch + entry point ──────────────────────────────────────────

TRAIN_DISPATCHERS = {
    "transformer": train_transformer,
    "moe": train_moe,
    "rnn": train_rnn,
}


def main():
    parser = argparse.ArgumentParser(description="Training worker subprocess")
    parser.add_argument("--run-dir", required=True, help="Path to run directory")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())

    run_id = config["run_id"]
    device = config["device"]
    template_key = config.get("template", "transformer")

    ws = WorkerState(run_id, config, device, resume=args.resume)

    try:
        dispatcher = TRAIN_DISPATCHERS.get(template_key)
        if dispatcher is None:
            raise ValueError(f"Unknown template: {template_key}")
        dispatcher(ws)
    except Exception as e:
        ws.set_status("failed")
        sync_update_training_run(ws.run_id, error_message=str(e))
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
