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
        "max_iters": 2000,
        "eval_interval": 10,
        "eval_iters": 2,
        "optimizer": "adamw",
    },
    "inference": {
        "max_new_tokens": 100,
        "temperature": 0.8,
    },
    "dataset": "tiny_shakespeare",
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
        "max_iters": 2000,
        "eval_interval": 10,
        "eval_iters": 2,
        "optimizer": "adamw",
    },
    "inference": {
        "max_new_tokens": 100,
        "temperature": 0.8,
    },
    "dataset": "tiny_shakespeare",
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
        "max_iters": 2000,
        "eval_interval": 10,
        "eval_iters": 2,
        "optimizer": "adamw",
    },
    "inference": {
        "max_new_tokens": 100,
        "temperature": 0.8,
    },
    "dataset": "tiny_shakespeare",
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
}

PRESETS = {
    "baseline": BASELINE_CONFIG,
    "moe": MOE_CONFIG,
    "lr_sensitivity": LR_SENSITIVITY_CONFIG,
    "rnn_baseline": RNN_BASELINE_CONFIG,
}
