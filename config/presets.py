"""Hardcoded experiment presets for Tier 1 (replaces auth + templates system)."""

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
        "max_iters": 5000,
        "eval_interval": 100,
        "eval_iters": 10,
        "optimizer": "adamw",
    },
    "dataset": "tiny_shakespeare",
}

ROPE_CONFIG = {
    "name": "RoPE Positional Encoding",
    "description": "Same architecture but with Rotary Position Embeddings instead of learned.",
    "template": "transformer",
    "model": {
        "vocab_size": 65,
        "block_size": 128,
        "n_embd": 192,
        "n_head": 6,
        "n_layer": 4,
        "dropout": 0.1,
        "pos_encoding": "rope",
        "activation": "gelu",
    },
    "training": {
        "batch_size": 64,
        "learning_rate": 3e-4,
        "max_iters": 5000,
        "eval_interval": 100,
        "eval_iters": 10,
        "optimizer": "adamw",
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
        "max_iters": 5000,
        "eval_interval": 100,
        "eval_iters": 10,
        "optimizer": "adamw",
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
    "dataset": "dinos",
}

PRESETS = {
    "baseline": BASELINE_CONFIG,
    "rope": ROPE_CONFIG,
    "lr_sensitivity": LR_SENSITIVITY_CONFIG,
    "rnn_baseline": RNN_BASELINE_CONFIG,
}
