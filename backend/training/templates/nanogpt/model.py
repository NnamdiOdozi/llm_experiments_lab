"""NanoGPT LM — GPT-2 style, reuses the TinyTransformer building blocks.

This is a faithful port of Karpathy's nanoGPT (see
https://github.com/karpathy/nanoGPT) on top of the existing
TinyTransformerLM architecture. The decoder-only transformer blocks,
causal masked attention, and forward/generate loops are inherited
unchanged. The nanoGPT-specific differences are structural, not
hyperparameter-only:

  * weight tying: the token embedding table and the lm_head share the
    same weight tensor (GPT-2 style, ~saves one vocab_size x n_embd
    matrix of parameters);
  * bias=False everywhere (GPT-2 style) — the qkv projection was
    already bias-free in the base transformer;
  * scaled residual init: the residual projection layers (attention
    out_proj and the FFN's second Linear) are re-initialized with a
    smaller std, 0.02 / sqrt(2 * n_layer), per the GPT-2 paper.

The heavier training-side differences (weight-decay optimizer with a
decay/no-decay parameter split, warmup + cosine LR schedule, gradient
accumulation) live in train_worker.py, keyed off optional training
config fields so the base transformer/MoE/RNN loops are unaffected.
"""

import math

import torch
import torch.nn as nn

from backend.training.templates.transformer.model import TinyTransformerLM


class NanoGPTModel(TinyTransformerLM):
    """GPT-2 style decoder-only LM with weight tying and scaled init."""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        dropout: float,
        pos_encoding: str = "learned",
        activation: str = "gelu",
        bias: bool = False,
        tie_weights: bool = True,
    ):
        # The base class builds the full transformer (token_emb, pos_emb,
        # blocks, ln_f, lm_head) and applies the standard normal(0, 0.02)
        # init. bias here defaults to False (GPT-2 style).
        super().__init__(
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            block_size=block_size,
            dropout=dropout,
            pos_encoding=pos_encoding,
            activation=activation,
            bias=bias,
        )

        if tie_weights:
            # Share the token embedding weight with the lm_head. Must
            # happen after super().__init__ so the lm_head weight tensor
            # already exists; assigning makes them the SAME tensor, so a
            # backward pass updates one tensor reflected in both places.
            self.lm_head.weight = self.token_emb.weight

        # Scaled residual init, per the GPT-2 paper: the residual-stream
        # projections get a smaller std so the residual stream doesn't
        # accumulate too large a variance across many layers. Overrides
        # the base class's flat 0.02 init on just these two modules.
        if n_layer > 1:
            scale = 0.02 / math.sqrt(2 * n_layer)
            for pn, p in self.named_parameters():
                if pn.endswith("out_proj.weight") or pn.endswith("net.2.weight"):
                    nn.init.normal_(p, mean=0.0, std=scale)


def build_model_from_config(config: dict) -> NanoGPTModel:
    """Instantiate a NanoGPTModel from an experiment config dict."""
    model_cfg = config["model"]
    return NanoGPTModel(
        vocab_size=model_cfg["vocab_size"],
        n_embd=model_cfg["n_embd"],
        n_head=model_cfg["n_head"],
        n_layer=model_cfg["n_layer"],
        block_size=model_cfg["block_size"],
        dropout=model_cfg["dropout"],
        pos_encoding=model_cfg.get("pos_encoding", "learned"),
        activation=model_cfg.get("activation", "gelu"),
        bias=model_cfg.get("bias", False),
        tie_weights=model_cfg.get("tie_weights", True),
    )