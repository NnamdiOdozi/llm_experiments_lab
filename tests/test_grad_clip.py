"""Gradient clipping is applied when grad_clip is set, and only then.

Karpathy's nanoGPT clips at 1.0 between the backward pass and the optimizer
step; our char-LM loop did not clip at all (the RNN path always has). At 124M
params with a 50k vocab an unlucky batch can spike the run, so nanoGPT now sets
grad_clip=1.0. The call is config-driven, so templates that do not set the key
are unaffected — these tests pin both halves of that.

See docs/NANOGPT_VS_KARPATHY.md §3 and DEFERRED_WORK.md (item closed 2026-09-16).
"""

import torch

from backend.training import train_worker
from config.presets import PRESETS
from config.settings import settings

BASE_MODEL = {
    "vocab_size": 16, "block_size": 8, "n_embd": 8, "n_head": 2,
    "n_layer": 1, "dropout": 0.0, "pos_encoding": "learned", "activation": "gelu",
}
BASE_TRAINING = {
    "batch_size": 2, "learning_rate": 1e-3, "max_iters": 3,
    "eval_interval": 100, "eval_iters": 1, "optimizer": "adamw",
}


def _run(monkeypatch, tmp_path, training_extra):
    """Run 3 tiny steps, returning the grad_clip values clip_grad_norm_ saw."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(train_worker, "load_tiny_shakespeare", lambda: "abcdefgh\n" * 200)
    monkeypatch.setattr(train_worker, "sync_update_training_run", lambda run_id, **kw: None)

    seen = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm, *a, **kw):
        seen.append(max_norm)
        return real_clip(params, max_norm, *a, **kw)

    monkeypatch.setattr(train_worker.nn.utils, "clip_grad_norm_", spy)

    config = {
        "template": "transformer",
        "model": dict(BASE_MODEL),
        "training": {**BASE_TRAINING, **training_extra},
    }
    ws = train_worker.WorkerState(run_id=999998, config=config, device="cpu")
    train_worker.train_transformer(ws)
    return seen


def test_clipping_applied_once_per_step_when_configured(monkeypatch, tmp_path):
    seen = _run(monkeypatch, tmp_path, {"grad_clip": 1.0})
    assert seen == [1.0, 1.0, 1.0], f"expected one clip per step at 1.0, got {seen}"


def test_no_clipping_when_grad_clip_absent(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, {}) == []


def test_no_clipping_when_grad_clip_is_zero(monkeypatch, tmp_path):
    """0.0 must mean 'off', not 'clip everything to zero' (which would kill training)."""
    assert _run(monkeypatch, tmp_path, {"grad_clip": 0.0}) == []


def test_nanogpt_preset_clips_at_karpathys_default():
    assert PRESETS["nanogpt"]["training"]["grad_clip"] == 1.0


def test_other_presets_are_unchanged_by_this():
    """nanoGPT-only by design — the baseline's loss curve must not shift."""
    for key in ("baseline", "moe"):
        assert "grad_clip" not in PRESETS[key]["training"], f"{key} unexpectedly clips"
