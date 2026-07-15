"""Tiny MoE Transformer LM — Mixture of Experts with DeepSeek-style fine-grained experts.

Reuses MultiHeadSelfAttention and RotaryPositionalEncoding from the transformer template.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.training.templates.transformer.model import (
    MultiHeadSelfAttention,
    ACTIVATIONS,
)


class ExpertFeedForward(nn.Module):
    """Single expert FFN with configurable hidden dim."""

    def __init__(self, n_embd: int, hid_dim: int, dropout: float, activation: str = "gelu"):
        super().__init__()
        act_cls = ACTIVATIONS.get(activation, nn.GELU)
        self.net = nn.Sequential(
            nn.Linear(n_embd, hid_dim),
            act_cls(),
            nn.Linear(hid_dim, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoE(nn.Module):
    """Top-K Sparse Mixture of Experts with capacity-based token dropping."""

    def __init__(
        self,
        n_embd: int,
        dropout: float,
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        activation: str = "gelu",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        # DeepSeek-style: half-sized expert hidden dim
        exp_hid_dim = 2 * n_embd

        self.experts = nn.ModuleList([
            ExpertFeedForward(n_embd, exp_hid_dim, dropout, activation)
            for _ in range(num_experts)
        ])
        self.router = nn.Linear(n_embd, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        B, T, C = x.shape
        x_flat = x.reshape(B * T, C)
        total_tokens = x_flat.shape[0]

        capacity = int(total_tokens * self.top_k / self.num_experts * self.capacity_factor)

        logits = self.router(x_flat)
        probs = torch.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(probs, k=self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x_flat)
        total_dropped = 0

        for expert_idx, expert in enumerate(self.experts):
            token_idx, route_idx = torch.where(topk_indices == expert_idx)
            num_assigned = token_idx.numel()
            if num_assigned == 0:
                continue
            if num_assigned > capacity:
                total_dropped += num_assigned - capacity
                token_idx = token_idx[:capacity]
                route_idx = route_idx[:capacity]
            if token_idx.numel() == 0:
                continue

            expert_input = x_flat[token_idx]
            expert_weight = topk_weights[token_idx, route_idx].unsqueeze(-1)
            expert_output = expert(expert_input) * expert_weight
            output.index_add_(0, token_idx, expert_output)

        output = output.view(B, T, C)
        drop_rate = total_dropped / (total_tokens * self.top_k)
        return output, drop_rate


class BlockMoe(nn.Module):
    """Pre-norm Transformer block with MoE replacing dense FFN."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float,
        pos_encoding: str = "rope",
        activation: str = "gelu",
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadSelfAttention(n_embd, n_head, block_size, dropout, pos_encoding)
        self.ln2 = nn.LayerNorm(n_embd)
        self.moe = MoE(n_embd, dropout, num_experts, top_k, capacity_factor, activation)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        x = x + self.attn(self.ln1(x))
        moe_out, drop_rate = self.moe(self.ln2(x))
        x = x + moe_out
        return x, drop_rate


class TinyMoeLM(nn.Module):
    """Decoder-only Transformer LM with Mixture of Experts blocks."""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        dropout: float,
        pos_encoding: str = "rope",
        activation: str = "gelu",
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.block_size = block_size
        self.pos_encoding = pos_encoding

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        if pos_encoding == "learned":
            self.pos_emb = nn.Embedding(block_size, n_embd)

        self.blocks = nn.ModuleList([
            BlockMoe(n_embd, n_head, block_size, dropout, pos_encoding,
                     activation, num_experts, top_k, capacity_factor)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets=None):
        B, T = idx.shape
        device = idx.device

        tok = self.token_emb(idx)
        if self.pos_encoding == "learned":
            pos = self.pos_emb(torch.arange(T, device=device))
            x = tok + pos
        else:
            x = tok

        layer_drop_rates = []
        for block in self.blocks:
            x, drop_rate = block(x)
            layer_drop_rates.append(drop_rate)

        avg_drop_rate = sum(layer_drop_rates) / len(layer_drop_rates)

        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, avg_drop_rate

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8, greedy: bool = False) -> torch.Tensor:
        """Autoregressive generation.  temperature < 1 → sharper (more greedy),
        temperature > 1 → flatter (more random).

        greedy=True always picks the single highest-probability token
        (argmax) — temperature has no effect in this mode, see
        transformer/model.py's generate() docstring for why. See
        docs/DESIGN_DECISIONS.md.
        """
        self.train(False)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if greedy:
                next_idx = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)  # scale before softmax
                next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx

    def register_diagnostic_hooks(self, session_id: str):
        """Register forward hooks at diagnostic node points for a session.

        Delegates to the shared hook factory in backend.training.diagnostics,
        which handles both transformer and MoE outputs (unwrapping MoE tuples).
        For MoE, the MoE block is treated as a single opaque node (block.{i}.moe)
        with shape capture only; per-expert routing breakdown is deferred to Phase 2.

        See docs/DESIGN_DECISIONS.md §66.
        """
        from backend.training.diagnostics import make_hook_for_diagnostics

        # Collect and return the hook handles — discarded handles meant
        # delete_session() could never detach hooks. The diagnostics-side
        # wrapper stores these on the session so eviction can actually detach them.
        handles = []

        # Register embedding hook
        handles.append(self.token_emb.register_forward_hook(make_hook_for_diagnostics("embedding", session_id)))

        # Register block hooks
        for i, block in enumerate(self.blocks):
            handles.append(block.ln1.register_forward_hook(make_hook_for_diagnostics(f"block.{i}.ln1", session_id)))
            handles.append(block.attn.register_forward_hook(make_hook_for_diagnostics(f"block.{i}.attention", session_id)))
            handles.append(block.ln2.register_forward_hook(make_hook_for_diagnostics(f"block.{i}.ln2", session_id)))
            handles.append(block.moe.register_forward_hook(make_hook_for_diagnostics(f"block.{i}.moe", session_id)))

        # Register final norm and lm_head hooks
        handles.append(self.ln_f.register_forward_hook(make_hook_for_diagnostics("final_norm", session_id)))
        handles.append(self.lm_head.register_forward_hook(make_hook_for_diagnostics("lm_head", session_id)))
        return handles


def build_model_from_config(config: dict) -> TinyMoeLM:
    """Instantiate a TinyMoeLM from an experiment config dict."""
    m = config["model"]
    return TinyMoeLM(
        vocab_size=m["vocab_size"],
        n_embd=m["n_embd"],
        n_head=m["n_head"],
        n_layer=m["n_layer"],
        block_size=m["block_size"],
        dropout=m["dropout"],
        pos_encoding=m.get("pos_encoding", "rope"),
        activation=m.get("activation", "gelu"),
        num_experts=m.get("num_experts", 8),
        top_k=m.get("top_k", 2),
        capacity_factor=m.get("capacity_factor", 1.25),
    )
