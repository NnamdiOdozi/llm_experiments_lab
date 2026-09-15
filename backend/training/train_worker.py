"""Standalone training worker — runs as a subprocess.

Usage: python -m backend.training.train_worker --run-dir data/runs/42
"""

import argparse
import datetime
import importlib.metadata
import json
import math
import os
import subprocess as sp
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn

from backend.training import artifacts
from backend.training.status import RunStatus
from backend.training.templates import TEMPLATE_REGISTRY
from backend.training.templates.transformer.data import CharDataset, load_tiny_shakespeare
from backend.training.templates.rnn.data import DinosDataset, load_dinos_dataset
from backend.training.templates.rnn.model import one_hot_encode
from backend.training.tokenizers.loader import load_tokenizer
from backend.training.tokenizers.dataset import TokenizedTextDataset
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


def _sample_resource_usage(device: str) -> dict:
    """CPU/RAM always (psutil); GPU utilization/memory/temp only for cuda
    devices (nvidia-smi). Best-effort — utilization is supplementary
    telemetry, not correctness-critical, so failures don't stop training.
    2026-07-12 addition, see docs/DESIGN_DECISIONS.md.
    """
    usage = {}
    try:
        usage["cpu_percent"] = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        usage["ram_used_mb"] = round(vm.used / 1024 / 1024, 1)
        usage["ram_total_mb"] = round(vm.total / 1024 / 1024, 1)
    except Exception as exc:
        print(f"[train_worker] psutil sampling failed: {exc}", file=sys.stderr)

    if device.startswith("cuda"):
        try:
            result = sp.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            gpu_util, mem_used, mem_total, temp = result.stdout.strip().split(", ")
            usage["gpu_utilization_pct"] = float(gpu_util)
            usage["gpu_memory_used_mb"] = float(mem_used)
            usage["gpu_memory_total_mb"] = float(mem_total)
            usage["gpu_temp_c"] = float(temp)
        except Exception as exc:
            print(f"[train_worker] nvidia-smi sampling failed: {exc}", file=sys.stderr)

    return usage


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
        # Elapsed seconds accumulated across pause/resume segments. A pause
        # writes the running total to the checkpoint; on resume this offset is
        # restored so elapsed_seconds keeps counting from the run's FIRST start,
        # not the resume moment. Without it, resume showed "elapsed ~0s" again.
        self.elapsed_before = 0.0
        self.metrics: list[dict] = []
        self.resume = resume
        self.checkpoint_extra: dict = {}  # extra fields to include in checkpoint (e.g. epoch)
        self._last_progress: float = 0.0  # wall-clock of last progress update

    def elapsed(self) -> float:
        """Cumulative training seconds: this segment (now - started_at) plus
        everything accumulated before the current pause/resume cycle."""
        return self.elapsed_before + (time.time() - self.started_at if self.started_at > 0 else 0)

    def _total_steps(self) -> int:
        t = self.config.get("training", {})
        return t.get("max_iters", t.get("epochs", 0) * 100)

    def yield_gpu(self, step: int):
        """Sync + micro-sleep every N steps so WSL2 display compositor isn't starved."""
        if settings.gpu_yield_enabled and self.device != "cpu" and step % settings.gpu_yield_interval == 0:
            torch.cuda.synchronize()
            time.sleep(settings.gpu_yield_sleep)

    def maybe_update_progress(self):
        """Throttled progress update — writes status.json at most every 2s."""
        now = time.time()
        if now - self._last_progress >= 2.0:
            self.update_progress()
            self._last_progress = now

    def sync_metadata(self, dataset_name: str):
        """Write run metadata to DB and disk — called once at training start."""
        sync_update_training_run(self.run_id,
            config_snapshot=json.dumps(self.config),
            seed=settings.random_seed,
            template_key=self.template_key,
            dataset_name=dataset_name,
            metrics_path=str(artifacts.metrics_path(self.run_id)),
            device_name=_get_device_name(self.device),
            param_count=_param_count(self.model),
            package_versions=json.dumps(_get_package_versions()),
            git_commit=_get_git_commit(),
        )
        self.write_run_meta(dataset_name)

    def set_status(self, status: str):
        elapsed = self.elapsed()
        artifacts.write_status(self.run_id, {
            "run_id": self.run_id,
            "status": status,
            "current_step": self.current_step,
            "total_steps": self._total_steps(),
            "metrics_count": len(self.metrics),
            "template": self.template_key,
            "elapsed_seconds": round(elapsed, 1),
            "pid": os.getpid(),
            "device": self.device,
        })
        updates: dict = {"status": status, "current_step": self.current_step}
        if status == RunStatus.RUNNING:
            updates["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        elif status in (RunStatus.COMPLETED, RunStatus.FAILED):
            updates["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        sync_update_training_run(self.run_id, **updates)

    def update_progress(self):
        """Update status.json with current step/metrics count without changing status."""
        elapsed = self.elapsed()
        status = artifacts.read_status(self.run_id)
        current_status = status["status"] if status else RunStatus.RUNNING
        artifacts.write_status(self.run_id, {
            "run_id": self.run_id,
            "status": current_status,
            "current_step": self.current_step,
            "total_steps": self._total_steps(),
            "metrics_count": len(self.metrics),
            "template": self.template_key,
            "elapsed_seconds": round(elapsed, 1),
            "pid": os.getpid(),
            "device": self.device,
        })

    def write_metric(self, metric_row: dict):
        metric_row["timestamp"] = datetime.datetime.now().isoformat()
        metric_row.update(_sample_resource_usage(self.device))
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
            # Cumulative training seconds so a resume keeps counting total
            # elapsed time instead of restarting at 0.
            "elapsed_accum": self.elapsed(),
            **extra,
        }
        # Add tokenizer metadata for checkpoint provenance (transformer/MoE/nanogpt only)
        if self.template_key in ("transformer", "moe", "nanogpt"):
            data_cfg = self.config.get("data", {})
            data["tokenizer_id"] = data_cfg.get("tokenizer", "char")
            data["tokenizer_version"] = data_cfg.get("tokenizer_version", "unknown")
            data["tokenizer_artifact_hash"] = data_cfg.get("tokenizer_hash", "unknown")
            data["vocab_size"] = data_cfg.get("vocab_size", 65)
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
        # Restore cumulative elapsed time from before the pause. Older
        # checkpoints predate this field, so default to 0 (fresh-run behavior).
        self.elapsed_before = cp.get("elapsed_accum", 0.0)
        # Reload previous metrics from disk
        mf = artifacts.metrics_path(self.run_id)
        if mf.exists():
            with open(mf) as f:
                self.metrics = [json.loads(line) for line in f if line.strip()]
        return cp

    def check_pause(self, step: int) -> bool:
        """Check pause/stop flags. Returns True if run should terminate."""
        if artifacts.has_flag(self.run_id, "pause"):
            self.set_status(RunStatus.PAUSE_REQUESTED)
            self.set_status(RunStatus.CHECKPOINTING)
            self.save_checkpoint(step, **self.checkpoint_extra)
            sync_update_training_run(
                self.run_id,
                checkpoint_path=str(artifacts.checkpoint_path(self.run_id)),
            )
            self.set_status(RunStatus.PAUSED)
            # Exit process — resume will launch a new worker from checkpoint
            sys.exit(0)
        if artifacts.has_flag(self.run_id, "stop"):
            self.set_status(RunStatus.CANCELLED)
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


def _configure_optimizer(model: nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    """Build the optimizer for a model from training config.

    When weight_decay is unset or 0, falls back to the plain
    AdamW(lr) used by every existing template — identical behavior.
    When weight_decay > 0, groups parameters into decay (>=2D weights)
    vs no-decay (1D params / embeddings / biases) groups and applies
    betas if provided, matching nanoGPT's configure_optimizers.
    """
    opt_cls = OPTIMIZERS.get(train_cfg.get("optimizer", "adamw"), torch.optim.AdamW)
    lr = train_cfg["learning_rate"]
    weight_decay = train_cfg.get("weight_decay", 0.0)
    if weight_decay <= 0.0:
        return opt_cls(model.parameters(), lr=lr)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    betas = tuple(train_cfg.get("betas", (0.9, 0.95)))
    return opt_cls(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
    )


def _get_lr(step: int, train_cfg: dict) -> float:
    """Per-step LR: linear warmup then cosine decay to min_lr.

    Returns the configured learning_rate when no schedule is set
    (lr_decay_iters absent), so transformer/MoE/RNN keep a constant LR.
    Mirrors nanoGPT's get_lr.
    """
    lr = train_cfg["learning_rate"]
    if train_cfg.get("lr_decay_iters") is None:
        return lr
    lr_decay_iters = train_cfg["lr_decay_iters"]
    warmup_iters = train_cfg.get("warmup_iters", 0)
    min_lr = train_cfg.get("min_lr", 0.0)
    if step < warmup_iters:
        return lr * (step + 1) / (warmup_iters + 1)
    if step > lr_decay_iters:
        return min_lr
    decay_ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def _train_char_lm(ws: WorkerState, template_key: str, eval_fn, build_metric_row):
    """Unified training loop for character-level LMs (transformer/MoE).

    Args:
        ws: WorkerState with config, device, etc.
        template_key: "transformer" or "moe" — determines model registry lookup
        eval_fn: Callable that returns dict with loss (and drop_rate for MoE)
        build_metric_row: Callable that builds metric dict from eval results
    """
    config = ws.config
    train_cfg = config["training"]
    device = ws.device

    ws.set_status(RunStatus.STARTING)

    # Load tokenizer and dataset using the new abstraction
    tokenizer = load_tokenizer(config.get("data", {"tokenizer": "char"}))
    text = load_tiny_shakespeare()
    ws.dataset = TokenizedTextDataset(tokenizer, text, config["model"]["block_size"], train_cfg["batch_size"])

    # Sync vocab_size from dataset to model config
    config["model"]["vocab_size"] = ws.dataset.vocab_size

    template = TEMPLATE_REGISTRY[template_key]
    ws.model = template["build_model"](config).to(device)
    ws.optimizer = _configure_optimizer(ws.model, train_cfg)

    if ws.resume:
        ws.load_checkpoint()

    ws.sync_metadata("tiny_shakespeare")

    ws.started_at = time.time()
    ws.set_status(RunStatus.RUNNING)
    max_iters = train_cfg["max_iters"]
    log_interval = train_cfg["eval_interval"]
    num_eval_iters = min(train_cfg.get("eval_iters", 10), 10)
    # Gradient accumulation (default 1 = one micro-batch per step, the
    # behavior every existing template has always had). When > 1, several
    # micro-batches are forward/backward'd with gradients summed before a
    # single optimizer step, matching nanoGPT's gradient_accumulation_steps.
    accum_steps = max(1, train_cfg.get("grad_accum_steps", 1))
    lr = train_cfg["learning_rate"]

    torch.manual_seed(settings.random_seed)
    # Checkpoint is saved AFTER completing step N, so resume from N+1.
    # Fresh runs start at 1, not 0 — "step N" means "N completed steps"
    # (that's what the resume arithmetic above already assumes), so a fresh
    # run starting at 0 did max_iters + 1 optimizer steps: an off-by-one
    # where "500 of 500" had actually trained 501 times. Fable review,
    # 2026-07-15 — see docs/DESIGN_DECISIONS.md §73.
    start_step = ws.current_step + 1 if ws.resume else 1

    for step in range(start_step, max_iters + 1):
        ws.current_step = step

        # Apply the (possibly scheduled) LR before the step's forward pass.
        # The eval/metric path below uses the same value so the logged
        # learning_rate reflects what actually trained this step.
        current_lr = _get_lr(step, train_cfg)
        for g in ws.optimizer.param_groups:
            g["lr"] = current_lr

        ws.optimizer.zero_grad(set_to_none=True)
        for _ in range(accum_steps):
            xb, yb = ws.dataset.get_batch("train", device)
            fwd_output = ws.model(xb, yb)
            # Unpack forward output: transformer returns (logits, loss),
            # MoE returns (logits, loss, drop_rate)
            if len(fwd_output) == 3:
                _, loss, _ = fwd_output
            else:
                _, loss = fwd_output
            # Scale by accum_steps so summed gradients equal the mean loss
            # across micro-batches (nanoGPT does the same).
            (loss / accum_steps).backward()
        ws.optimizer.step()
        ws.yield_gpu(step)
        ws.maybe_update_progress()

        # Check pause/stop AFTER training step so checkpoint = completed step
        if ws.check_pause(step):
            return

        if step > 0 and step % log_interval == 0:
            eval_results = eval_fn(ws.model, ws.dataset, device, num_eval_iters)
            metric_row = build_metric_row(step, eval_results, current_lr, ws.elapsed(), ws.model)
            ws.write_metric(metric_row)
            ws.save_checkpoint(step)

    ws.save_checkpoint(max_iters)
    sync_update_training_run(ws.run_id, checkpoint_path=str(artifacts.checkpoint_path(ws.run_id)))
    ws.set_status(RunStatus.COMPLETED)


def _build_transformer_metric_row(step: int, eval_results: dict, lr: float, elapsed: float, model) -> dict:
    """Build metric row dict for transformer eval results. `elapsed` is the
    CUMULATIVE training time (across pause/resume), computed by ws.elapsed()."""
    return {
        "step": step,
        "train_loss": round(eval_results["train"], 4),
        "val_loss": round(eval_results["val"], 4),
        "learning_rate": lr,
        "elapsed_seconds": round(elapsed, 1),
        "param_count": _param_count(model),
    }


def _build_moe_metric_row(step: int, eval_results: dict, lr: float, elapsed: float, model) -> dict:
    """Build metric row dict for MoE eval results, including drop rates.
    `elapsed` is the cumulative training time (across pause/resume)."""
    return {
        "step": step,
        "train_loss": round(eval_results["train"]["loss"], 4),
        "val_loss": round(eval_results["val"]["loss"], 4),
        "train_drop_rate": round(eval_results["train"]["drop_rate"] * 100, 1),
        "val_drop_rate": round(eval_results["val"]["drop_rate"] * 100, 1),
        "learning_rate": lr,
        "elapsed_seconds": round(elapsed, 1),
        "param_count": _param_count(model),
    }


def train_transformer(ws: WorkerState):
    """Train a transformer LM on tiny_shakespeare."""
    _train_char_lm(ws, "transformer", _transformer_eval, _build_transformer_metric_row)


def train_moe(ws: WorkerState):
    """Train a MoE LM on tiny_shakespeare."""
    _train_char_lm(ws, "moe", _moe_eval, _build_moe_metric_row)


def train_nanogpt(ws: WorkerState):
    """Train a nanoGPT LM on tiny_shakespeare.

    Shares the transformer's (logits, loss) return signature and its
    eval/metric builders; the nanoGPT-specific optimizer (weight decay,
    betas), LR schedule, and gradient accumulation are driven entirely by
    optional training-config keys handled inside _train_char_lm.
    """
    _train_char_lm(ws, "nanogpt", _transformer_eval, _build_transformer_metric_row)


# ── RNN ─────────────────────────────────────────────────────────────


def train_rnn(ws: WorkerState):
    config = ws.config
    train_cfg = config["training"]
    device = ws.device

    ws.set_status(RunStatus.STARTING)

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

    ws.sync_metadata("dinos")

    ws.started_at = time.time()
    ws.set_status(RunStatus.RUNNING)
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
            ws.yield_gpu(counter)
            ws.maybe_update_progress()

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
                    "elapsed_seconds": round(ws.elapsed(), 1),
                    "param_count": _param_count(ws.model),
                })
                ws.save_checkpoint(counter, **ws.checkpoint_extra)

    ws.save_checkpoint(counter, **ws.checkpoint_extra)
    sync_update_training_run(ws.run_id, checkpoint_path=str(artifacts.checkpoint_path(ws.run_id)))
    ws.set_status(RunStatus.COMPLETED)


# ── Dispatch + entry point ──────────────────────────────────────────

TRAIN_DISPATCHERS = {
    "transformer": train_transformer,
    "moe": train_moe,
    "nanogpt": train_nanogpt,
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
        ws.set_status(RunStatus.FAILED)
        sync_update_training_run(ws.run_id, error_message=str(e))
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
