"""Hardcoded experiment presets for Tier 1 (replaces auth + templates system).

Each preset includes an "inference" section with generation parameters
(max_new_tokens, temperature) that are editable from the dashboard and
used by prompt_paused_model() when generating text from a checkpoint.
"""

# --- Transformer templates ---

BASELINE_CONFIG = {
    "name": "Baseline Tiny Transformer",
    "description": "Default 4-layer decoder-only transformer on Tiny Shakespeare.",
    "template": "transformer",
    "model": {
        "vocab_size": 65,
        "block_size": 128,
        "n_embd": 192,
        "n_head": 6,
        "n_layer": 4,
        "dropout": 0.1,
        "pos_encoding": "learned",
        "activation": "gelu",
    },
    "training": {
        "batch_size": 64,
        "learning_rate": 3e-4,
        "max_iters": 3000,
        "eval_interval": 10,
        "eval_iters": 2,
        "optimizer": "adamw",
    },
    "inference": {
        "max_new_tokens": 100,
        "temperature": 0.8,
    },
    "dataset": "tiny_shakespeare",
    "data": {
        "dataset": "tiny_shakespeare",
        "tokenizer": "char",
        "tokenizer_artifact": None,
        "vocab_size": 65,
    },
}

MOE_CONFIG = {
    "name": "Mixture of Experts (DeepSeek-style)",
    "description": "4-layer MoE transformer with 8 half-sized experts, top-2 routing, RoPE.",
    "template": "moe",
    "model": {
        "vocab_size": 65,
        "block_size": 128,
        "n_embd": 192,
        "n_head": 6,
        "n_layer": 4,
        "dropout": 0.1,
        "pos_encoding": "rope",
        "activation": "gelu",
        "num_experts": 8,
        "top_k": 2,
        "capacity_factor": 1.25,
    },
    "training": {
        "batch_size": 64,
        "learning_rate": 3e-4,
        "max_iters": 3000,
        "eval_interval": 10,
        "eval_iters": 2,
        "optimizer": "adamw",
    },
    "inference": {
        "max_new_tokens": 100,
        "temperature": 0.8,
    },
    "dataset": "tiny_shakespeare",
    "data": {
        "dataset": "tiny_shakespeare",
        "tokenizer": "char",
        "tokenizer_artifact": None,
        "vocab_size": 65,
    },
}

LR_SENSITIVITY_CONFIG = {
    "name": "High Learning Rate",
    "description": "Same baseline but with 10x learning rate to observe instability.",
    "template": "transformer",
    "model": {
        "vocab_size": 65,
        "block_size": 128,
        "n_embd": 192,
        "n_head": 6,
        "n_layer": 4,
        "dropout": 0.1,
        "pos_encoding": "learned",
        "activation": "gelu",
    },
    "training": {
        "batch_size": 64,
        "learning_rate": 3e-3,
        "max_iters": 3000,
        "eval_interval": 10,
        "eval_iters": 2,
        "optimizer": "adamw",
    },
    "inference": {
        "max_new_tokens": 100,
        "temperature": 0.8,
    },
    "dataset": "tiny_shakespeare",
    "data": {
        "dataset": "tiny_shakespeare",
        "tokenizer": "char",
        "tokenizer_artifact": None,
        "vocab_size": 65,
    },
}

# --- nanoGPT template ---

# Karpathy's nanoGPT default hyperparameters (GPT-2 style): 124M params,
# block_size 1024, weight tying, bias=False, weight-decay optimizer with a
# decay/no-decay split, and a warmup + cosine LR schedule. Every field is
# editable from the dashboard ConfigPanel. vocab_size is synced at train
# time from the chosen tokenizer (char=65 / bpe_1k=1024 / bpe_4k=4096) —
# the sidepane tokenizer dropdown sets it. Defaults to char so the preset
# runs out of the box on CPU; switch to BPE for the closer-to-nanoGPT setup.
NANOGPT_CONFIG = {
    "name": "nanoGPT (GPT-2 mini)",
    "description": "Karpathy's nanoGPT defaults: 12-layer GPT-2 style LM (block_size 1024, n_embd 768) with weight tying, bias-free layers, weight-decay AdamW, warmup+cosine LR. ~124M params at nanoGPT's full BPE vocab; ~86M with the default char tokenizer.",
    "template": "nanogpt",
    "model": {
        "vocab_size": 65,
        "block_size": 1024,
        "n_embd": 768,
        "n_head": 12,
        "n_layer": 12,
        "dropout": 0.0,
        "pos_encoding": "learned",
        "activation": "gelu",
        "bias": False,
        "tie_weights": True,
    },
    "training": {
        "batch_size": 64,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "betas": [0.9, 0.95],
        "warmup_iters": 200,
        "lr_decay_iters": 3000,
        "min_lr": 3e-5,
        "grad_accum_steps": 4,
        "max_iters": 3000,
        "eval_interval": 10,
        "eval_iters": 2,
        "optimizer": "adamw",
    },
    "inference": {
        "max_new_tokens": 100,
        "temperature": 0.8,
    },
    "dataset": "tiny_shakespeare",
    "data": {
        "dataset": "tiny_shakespeare",
        "tokenizer": "char",
        "tokenizer_artifact": None,
        "vocab_size": 65,
    },
}

# --- RNN templates ---

RNN_BASELINE_CONFIG = {
    "name": "Baseline CharRNN (LSTM)",
    "description": "2-layer LSTM character-level LM on dinosaur names dataset.",
    "template": "rnn",
    "model": {
        "vocab_size": 29,  # determined by dinos.txt vocab
        "n_hidden": 256,
        "n_layers": 2,
        "dropout": 0.5,
    },
    "training": {
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 50,
        "seq_len": 50,
        "clip": 5,
        "print_every": 10,
        "optimizer": "adam",
    },
    "inference": {
        "max_new_tokens": 100,
        "temperature": 0.8,
    },
    "dataset": "dinos",
    "data": {
        "dataset": "dinos",
        "tokenizer": "char",
        "tokenizer_artifact": None,
        "vocab_size": 29,
    },
}

PRESETS = {
    "baseline": BASELINE_CONFIG,
    "moe": MOE_CONFIG,
    "lr_sensitivity": LR_SENSITIVITY_CONFIG,
    "rnn_baseline": RNN_BASELINE_CONFIG,
    "nanogpt": NANOGPT_CONFIG,
}
