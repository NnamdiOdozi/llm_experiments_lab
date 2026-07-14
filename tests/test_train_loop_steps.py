"""Training-loop step-count test (Fable review, 2026-07-15 — DESIGN_DECISIONS
§73): fresh runs previously looped `range(0, max_iters + 1)` — max_iters + 1
optimizer steps, so "500 of 500" had actually trained 501 times. "Step N"
means "N completed steps" (the resume arithmetic already assumed this), so a
fresh run must start at step 1 and do exactly max_iters optimizer steps.

Runs a real (tiny) train_transformer end-to-end with the dataset download,
DB writes, and optimizer intercepted — everything lands under tmp_path, no
stray files or processes.
"""

import torch

from backend.training import train_worker
from config.settings import settings


def test_fresh_run_does_exactly_max_iters_optimizer_steps(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path)  # status/checkpoint under tmp
    monkeypatch.setattr(train_worker, "load_tiny_shakespeare", lambda: "abcdefgh\n" * 200)
    monkeypatch.setattr(train_worker, "sync_update_training_run", lambda run_id, **kw: None)

    calls = {"n": 0}

    class CountingAdamW(torch.optim.AdamW):
        def step(self, *args, **kwargs):
            calls["n"] += 1
            return super().step(*args, **kwargs)

    monkeypatch.setitem(train_worker.OPTIMIZERS, "adamw", CountingAdamW)

    config = {
        "template": "transformer",
        "model": {
            "vocab_size": 16, "block_size": 8, "n_embd": 8, "n_head": 2,
            "n_layer": 1, "dropout": 0.0, "pos_encoding": "learned", "activation": "gelu",
        },
        "training": {
            "batch_size": 2, "learning_rate": 1e-3, "max_iters": 5,
            "eval_interval": 100, "eval_iters": 1, "optimizer": "adamw",
        },
    }
    ws = train_worker.WorkerState(run_id=999999, config=config, device="cpu")
    train_worker.train_transformer(ws)

    assert calls["n"] == 5, f"expected exactly max_iters=5 optimizer steps, got {calls['n']}"
    assert ws.current_step == 5  # display still ends at "5 of 5"
