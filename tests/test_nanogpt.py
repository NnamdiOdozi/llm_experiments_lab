"""nanoGPT template tests: structural GPT-2 properties, optimizer/LR helpers,
preset/registry wiring, and an end-to-end training-loop step-count check.

The loop test mirrors tests/test_train_loop_steps.py (real train_nanogpt with
the dataset download, DB writes, and optimizer intercepted so everything lands
under tmp_path). See docs/DESIGN_DECISIONS.md.
"""

import math
import time

import torch

from backend.training import train_worker
from config.presets import PRESETS
from config.settings import settings

NANO_CFG = {
    "model": {
        "vocab_size": 65, "block_size": 32, "n_embd": 64, "n_head": 4,
        "n_layer": 4, "dropout": 0.0, "pos_encoding": "learned",
        "activation": "gelu", "bias": False, "tie_weights": True,
    },
}


def _build():
    from backend.training.templates.nanogpt.model import build_model_from_config
    return build_model_from_config(NANO_CFG)


# ── Model structure ───────────────────────────────────────────────────

def test_weight_tying_shares_lm_head_with_token_emb():
    m = _build()
    assert m.lm_head.weight is m.token_emb.weight


def test_gpt2_style_bias_false_everywhere():
    m = _build()
    assert m.lm_head.bias is None
    assert m.ln_f.bias is None
    assert m.blocks[0].ln1.bias is None
    assert m.blocks[0].attn.out_proj.bias is None
    assert m.blocks[0].ffn.net[0].bias is None
    assert m.blocks[0].ffn.net[2].bias is None


def test_scaled_residual_init():
    m = _build()
    scale = 0.02 / math.sqrt(2 * NANO_CFG["model"]["n_layer"])
    # Residual projections carry the scaled std, not the base 0.02.
    std = m.blocks[0].ffn.net[2].weight.detach().std()
    assert abs(std.item() - scale) < 0.005
    # Non-residual params keep the base 0.02 init.
    tok_std = m.token_emb.weight.detach().std()
    assert abs(tok_std.item() - 0.02) < 0.005


def test_same_config_keys_as_transformer():
    # nanoGPT must accept the exact same model config keys as the tiny
    # transformer (user requirement) so the ConfigPanel affordances match.
    from backend.training.templates.transformer.model import build_model_from_config as bt
    from backend.training.templates.nanogpt.model import build_model_from_config as bn
    full = dict(NANO_CFG["model"])
    full.pop("bias")
    full.pop("tie_weights")
    bt({"model": dict(full)})   # must not raise
    bn({"model": dict(full)})   # must not raise (defaults bias=False, tie=True)


def test_forward_and_generate():
    m = _build()
    logits, loss = m(torch.randint(0, 65, (2, 16)), torch.randint(0, 65, (2, 16)))
    assert logits.shape == (2, 16, 65)
    assert isinstance(loss.item(), float)
    out = m.generate(torch.tensor([[0, 1, 2]]), max_new_tokens=5, temperature=0.8)
    assert out.shape == (1, 8)


# ── Optimizer + LR helpers ────────────────────────────────────────────

def test_plain_optimizer_when_no_weight_decay():
    m = _build()
    opt = train_worker._configure_optimizer(m, {"optimizer": "adamw", "learning_rate": 3e-4})
    assert len(opt.param_groups) == 1


def test_weight_decay_splits_into_decay_no_decay_groups():
    m = _build()
    opt = train_worker._configure_optimizer(
        m, {"optimizer": "adamw", "learning_rate": 3e-4, "weight_decay": 0.1, "betas": [0.9, 0.95]}
    )
    assert len(opt.param_groups) == 2
    g0, g1 = opt.param_groups
    assert g0["weight_decay"] == 0.1
    assert g1["weight_decay"] == 0.0
    assert tuple(g0["betas"]) == (0.9, 0.95)


def test_lr_schedule_warmup_cosine_min():
    cfg = {"learning_rate": 1e-3, "warmup_iters": 10, "lr_decay_iters": 100, "min_lr": 1e-4}
    assert train_worker._get_lr(5, cfg) < 1e-3          # warmup ramp
    assert train_worker._get_lr(9, cfg) > 1e-4
    assert abs(train_worker._get_lr(55, cfg) - 5.5e-4) < 1e-6  # cosine midpoint
    assert train_worker._get_lr(101, cfg) == 1e-4       # clamped to min_lr


def test_lr_constant_without_schedule():
    assert train_worker._get_lr(50, {"learning_rate": 1e-3}) == 1e-3


# ── Registry / preset wiring ──────────────────────────────────────────

def test_nanogpt_registered_everywhere():
    from backend.training.templates import TEMPLATE_REGISTRY
    assert "nanogpt" in TEMPLATE_REGISTRY
    assert "nanogpt" in train_worker.TRAIN_DISPATCHERS
    assert "nanogpt" in PRESETS
    assert PRESETS["nanogpt"]["template"] == "nanogpt"


def test_preset_builds_and_ties():
    from backend.training.templates.nanogpt.model import build_model_from_config
    m = build_model_from_config(PRESETS["nanogpt"])
    assert m.lm_head.weight is m.token_emb.weight


# ── End-to-end training loop ──────────────────────────────────────────

def test_train_nanogpt_does_exactly_max_iters_steps_with_grad_accum(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(train_worker, "load_tiny_shakespeare", lambda: "abcdefgh\n" * 200)
    monkeypatch.setattr(train_worker, "sync_update_training_run", lambda run_id, **kw: None)

    calls = {"n": 0}

    class CountingAdamW(torch.optim.AdamW):
        def step(self, *args, **kwargs):
            calls["n"] += 1
            return super().step(*args, **kwargs)

    monkeypatch.setitem(train_worker.OPTIMIZERS, "adamw", CountingAdamW)

    config = {
        "template": "nanogpt",
        "model": {
            "vocab_size": 16, "block_size": 8, "n_embd": 8, "n_head": 2,
            "n_layer": 1, "dropout": 0.0, "pos_encoding": "learned",
            "activation": "gelu", "bias": False, "tie_weights": True,
        },
        "training": {
            "batch_size": 2, "learning_rate": 1e-3, "weight_decay": 0.1,
            "betas": [0.9, 0.95], "warmup_iters": 2, "lr_decay_iters": 4,
            "min_lr": 1e-4, "grad_accum_steps": 2,
            "max_iters": 4, "eval_interval": 100, "eval_iters": 1,
            "optimizer": "adamw",
        },
    }
    ws = train_worker.WorkerState(run_id=999999, config=config, device="cpu")
    train_worker.train_nanogpt(ws)

    # 2 micro-batches per optimizer step but still exactly max_iters steps.
    assert calls["n"] == 4, f"expected max_iters=4 optimizer steps, got {calls['n']}"
    assert ws.current_step == 4
    assert ws.model.lm_head.weight is ws.model.token_emb.weight


# ── GPT-2 tokenizer (124.4M goal) ────────────────────────────────────

def test_gpt2_tokenizer_reports_50304_and_roundtrips():
    from backend.training.tokenizers.bpe import Gpt2Tokenizer
    t = Gpt2Tokenizer("data/tokenizers/gpt2.json")
    assert t.vocab_size == 50304                      # padded for 124.4M
    ids = t.encode("To be or not to be")
    assert max(ids) < 50257                           # real GPT-2 ids only
    assert t.decode(ids) == "To be or not to be"


def test_gpt2_tokenizer_padding_id_does_not_crash():
    from backend.training.tokenizers.bpe import Gpt2Tokenizer
    t = Gpt2Tokenizer("data/tokenizers/gpt2.json")
    meta = t.id_to_token(50303)                       # a padding row
    assert meta["raw"].startswith("<pad:")


def test_nanogpt_preset_reaches_1244m_with_gpt2():
    from backend.training.templates.nanogpt.model import build_model_from_config
    p = PRESETS["nanogpt"]
    assert p["model"]["vocab_size"] == 50304
    assert p["data"]["tokenizer"] == "gpt2"
    m = build_model_from_config(p)
    n = sum(x.numel() for x in m.parameters())
    assert abs(n / 1e6 - 124.4) < 0.5, f"expected ~124.4M, got {n/1e6:.1f}M"


# ── Cumulative training time across pause/resume ─────────────────────

def test_elapsed_accumulates_across_pause_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    from backend.training.templates.transformer.model import build_model_from_config

    cfg = {
        "template": "transformer",
        "model": {"vocab_size": 16, "block_size": 8, "n_embd": 8, "n_head": 2, "n_layer": 1, "dropout": 0.0},
    }

    # Segment 1: run ~5s, then pause (save checkpoint).
    ws = train_worker.WorkerState(run_id=888888, config=cfg, device="cpu")
    ws.model = build_model_from_config(cfg)
    ws.optimizer = torch.optim.AdamW(ws.model.parameters(), lr=1e-3)
    ws.started_at = time.time() - 5.0
    ws.current_step = 10
    ws.save_checkpoint(10)

    # Segment 2: resume from that checkpoint in a NEW worker.
    ws2 = train_worker.WorkerState(run_id=888888, config=cfg, device="cpu", resume=True)
    ws2.model = build_model_from_config(cfg)
    ws2.optimizer = torch.optim.AdamW(ws2.model.parameters(), lr=1e-3)
    ws2.load_checkpoint()
    ws2.started_at = time.time()  # fresh segment clock, like the training loop

    assert ws2.current_step == 10
    assert ws2.elapsed_before >= 5.0, f"elapsed_before not restored: {ws2.elapsed_before}"
    # Cumulative total = pre-pause time + new segment — definitely above the
    # 5s pre-pause value, and never reset toward 0.
    assert ws2.elapsed() >= ws2.elapsed_before
    assert ws2.elapsed() > 5.0