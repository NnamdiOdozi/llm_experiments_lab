"""Tiny Transformer LM — adapted from course notebook for configurable use."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionalEncoding(nn.Module):
    """Rotary Position Embeddings (RoPE) applied to Q and K tensors."""

    def __init__(self, head_size: int, block_size: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_size, 2).float() / head_size))
        positions = torch.arange(block_size).float()
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_head, T, head_size) — apply RoPE rotation."""
        T = x.shape[2]
        cos = self.cos_cached[:T].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:T].unsqueeze(0).unsqueeze(0)
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional RoPE."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float,
        pos_encoding: str = "learned",
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.n_embd = n_embd
        self.pos_encoding = pos_encoding

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd)

        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if pos_encoding == "rope":
            self.rope = RotaryPositionalEncoding(self.head_size, block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)

        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        if self.pos_encoding == "rope":
            q = self.rope(q)
            k = self.rope(k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = torch.softmax(scores, dim=-1)
        att = self.attn_dropout(att)

        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))


ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


class FeedForward(nn.Module):
    """Position-wise FFN: Linear -> activation -> Linear -> Dropout."""

    def __init__(self, n_embd: int, dropout: float, activation: str = "gelu"):
        super().__init__()
        act_cls = ACTIVATIONS.get(activation, nn.GELU)
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            act_cls(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float,
        pos_encoding: str = "learned",
        activation: str = "gelu",
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadSelfAttention(n_embd, n_head, block_size, dropout, pos_encoding)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    """Configurable decoder-only Transformer language model."""

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
    ):
        super().__init__()
        self.block_size = block_size
        self.pos_encoding = pos_encoding

        self.token_emb = nn.Embedding(vocab_size, n_embd)

        if pos_encoding == "learned":
            self.pos_emb = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout, pos_encoding, activation) for _ in range(n_layer)]
        )
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

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8) -> torch.Tensor:
        """Autoregressive generation.  temperature < 1 → sharper (more greedy),
        temperature > 1 → flatter (more random)."""
        self.train(False)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # scale logits before softmax
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx


def build_model_from_config(config: dict) -> TinyTransformerLM:
    """Instantiate a TinyTransformerLM from an experiment config dict."""
    model_cfg = config["model"]
    return TinyTransformerLM(
        vocab_size=model_cfg["vocab_size"],
        n_embd=model_cfg["n_embd"],
        n_head=model_cfg["n_head"],
        n_layer=model_cfg["n_layer"],
        block_size=model_cfg["block_size"],
        dropout=model_cfg["dropout"],
        pos_encoding=model_cfg.get("pos_encoding", "learned"),
        activation=model_cfg.get("activation", "gelu"),
    )
