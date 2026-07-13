"""Diagnostic session management for model introspection during pause/completion.

Maintains in-memory session state (model, tokenizer, tensor captures) keyed by
session_id. Each diagnostic session loads a checkpoint, tokenizes a prompt,
and captures tensor shapes + summary stats at predefined hook points during
forward passes. Phase 4 adds Q/K/V detail capture and session persistence.
"""

import math
import uuid
import torch
from typing import Optional
from dataclasses import dataclass, field
from backend.logging_config import training_log


@dataclass
class NodeCapture:
    """Captured tensor information at a node (hook point)."""
    input_shape: list[int]
    output_shape: list[int]
    summary: dict  # {mean, std, l2_norm, min, max}


@dataclass
class DiagnosticSnapshot:
    """Complete forward-pass snapshot — all node captures + lm_head top-k."""
    schema_version: int = 1
    diagnostic_session_id: str = ""
    generation_step: int = 0
    input_tokens: list[dict] = field(default_factory=list)  # [{position, id, text}, ...]
    generated_token: Optional[dict] = None  # {position, id, text}
    nodes: dict[str, dict] = field(default_factory=dict)  # node_id -> {input_shape, output_shape, summary}
    attention: dict = field(default_factory=lambda: {"available": False, "reason": "Not requested"})
    activation_summaries: dict = field(default_factory=lambda: {"available": False, "reason": "Not requested"})
    lm_head: dict = field(default_factory=dict)  # {logits_shape, selected_position, top_k}
    complete: bool = True

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "diagnostic_session_id": self.diagnostic_session_id,
            "generation_step": self.generation_step,
            "input_tokens": self.input_tokens,
            "generated_token": self.generated_token,
            "nodes": self.nodes,
            "attention": self.attention,
            "activation_summaries": self.activation_summaries,
            "lm_head": self.lm_head,
            "complete": self.complete,
        }


@dataclass
class DiagnosticSession:
    """In-memory session state for a diagnostic interaction."""
    session_id: str
    model: torch.nn.Module
    tokenizer: object  # CharDataset or equivalent
    device: str
    prompt_tokens: list[int]  # initial prompt encoded
    run_id: Optional[int] = None  # training run this session belongs to
    token_history: list[int] = field(default_factory=list)  # tokens generated so far
    captured_tensors: dict = field(default_factory=dict)  # node_id -> NodeCapture
    generation_step: int = 0
    last_snapshot: Optional[DiagnosticSnapshot] = None
    hook_handles: list = field(default_factory=list)  # for deregistering hooks


# Global registry: session_id -> DiagnosticSession
_diagnostic_sessions: dict[str, DiagnosticSession] = {}

# run_id -> most recent session_id for that run. Populated for BOTH local
# sessions (session lives in this same process) and remote/serverless
# sessions (session lives in the trainer container's process — only the id
# string is recorded here so the main server knows which session_id to ask
# the trainer about via the existing local/remote diagnostics_get route).
# Chatbot grounding (get_diagnostic_snapshot tool) reads this to find "the
# current diagnostic session" for a run without the user having to supply
# a session_id in chat. See docs/DESIGN_DECISIONS.md.
_run_to_session: dict[int, str] = {}


def get_latest_session_id_for_run(run_id: int) -> Optional[str]:
    """Most recent diagnostic session_id started for this run, or None."""
    return _run_to_session.get(run_id)


def record_session_for_run(run_id: int, session_id: str) -> None:
    """Record which session_id belongs to a run — called both when a local
    session is created here and when a remote diagnostics/start call is
    proxied (that session lives in the trainer container, not here)."""
    _run_to_session[run_id] = session_id


def new_session_id() -> str:
    """Generate a unique session ID."""
    return f"diag-{uuid.uuid4().hex[:8]}"


def create_diagnostic_session(
    model: torch.nn.Module,
    tokenizer: object,
    device: str,
    prompt_tokens: list[int],
    run_id: Optional[int] = None,
) -> str:
    """Create a new diagnostic session and register it.

    Args:
        model: Loaded PyTorch model in eval mode
        tokenizer: CharDataset or equivalent with encode/decode
        device: "cpu" or "cuda"
        prompt_tokens: Initial prompt tokens
        run_id: Training run this session belongs to (for chatbot grounding lookup)

    Returns:
        session_id
    """
    session_id = new_session_id()
    session = DiagnosticSession(
        session_id=session_id,
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt_tokens=prompt_tokens,
        run_id=run_id,
        token_history=[],
    )
    _diagnostic_sessions[session_id] = session
    if run_id is not None:
        record_session_for_run(run_id, session_id)
    training_log.info("Created diagnostic session %s for run_id=%s", session_id, run_id)
    return session_id


def get_session(session_id: str) -> Optional[DiagnosticSession]:
    """Retrieve a diagnostic session by ID."""
    return _diagnostic_sessions.get(session_id)


def delete_session(session_id: str) -> None:
    """Delete a diagnostic session and deregister all hooks."""
    session = _diagnostic_sessions.pop(session_id, None)
    if session is not None:
        # Deregister hooks
        for handle in session.hook_handles:
            handle.remove()
        training_log.info("Deleted diagnostic session %s", session_id)


def _compute_summary(tensor: torch.Tensor) -> dict:
    """Compute summary statistics for a tensor."""
    with torch.no_grad():
        flat = tensor.reshape(-1).float()
        return {
            "mean": float(flat.mean().item()),
            "std": float(flat.std().item()),
            "l2_norm": float(torch.norm(flat, p=2).item()),
            "min": float(flat.min().item()),
            "max": float(flat.max().item()),
        }


def register_diagnostic_hooks(model: torch.nn.Module, session_id: str) -> None:
    """Register forward hooks at diagnostic node points.

    Delegates to the model's register_diagnostic_hooks method, which knows
    about the model's internal structure.

    Args:
        model: The loaded model (must have register_diagnostic_hooks method)
        session_id: Session to receive captures
    """
    if hasattr(model, 'register_diagnostic_hooks'):
        model.register_diagnostic_hooks(session_id)


def _compute_attention_weights(
    session: "DiagnosticSession",
    layer: int,
    head: int,
    qkv_detail: bool = False,
) -> Optional[dict]:
    """Explicit (non-fused) QK^T -> scale -> causal mask -> softmax attention path.

    The model's normal forward pass uses fused/fast attention, which per
    Trainer_to_Frontend_Metrics.md doesn't expose weights — this recomputes
    just the requested layer/head manually, only when asked for (Phase 2).
    Phase 4: when qkv_detail=True, returns Q/K/V vectors for the last token
    position, one head only.

    Attribute names (qkv, n_head, head_size, blocks[i].attn/ln1) match
    backend/training/templates/transformer/model.py's MultiHeadSelfAttention
    exactly — verified against that file, not assumed.
    """
    try:
        model = session.model
        if not hasattr(model, "blocks") or layer < 0 or layer >= len(model.blocks):
            return None
        block = model.blocks[layer]
        attn = block.attn
        if head < 0 or head >= attn.n_head:
            return None

        all_tokens = session.prompt_tokens + session.token_history
        idx = torch.tensor([all_tokens], dtype=torch.long, device=session.device)

        with torch.inference_mode():
            x = model.token_emb(idx)
            if hasattr(model, "pos_emb"):
                x = x + model.pos_emb(torch.arange(x.shape[1], device=session.device))
            for i in range(layer):
                x = model.blocks[i](x)

            B, T, C = x.shape
            x_ln = block.ln1(x)
            q, k, v = attn.qkv(x_ln).split(C, dim=-1)
            q = q.view(B, T, attn.n_head, attn.head_size).transpose(1, 2)
            k = k.view(B, T, attn.n_head, attn.head_size).transpose(1, 2)
            v = v.view(B, T, attn.n_head, attn.head_size).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(attn.head_size)
            causal_mask = torch.tril(torch.ones(T, T, device=session.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)[0, head].tolist()

            token_labels = [session.tokenizer.decode([tid]) for tid in all_tokens]
            result = {
                "available": True,
                "layer": layer,
                "head": head,
                "weights": weights,
                "token_labels": token_labels,
            }

            # Phase 4: Q/K/V detail for the last token position only, this head
            if qkv_detail:
                last_token_pos = T - 1
                q_last = q[0, head, last_token_pos, :].tolist()
                k_last = k[0, head, last_token_pos, :].tolist()
                v_last = v[0, head, last_token_pos, :].tolist()
                result["qkv_detail"] = {
                    "position": last_token_pos,
                    "q": q_last,
                    "k": k_last,
                    "v": v_last,
                }

            return result
    except Exception as e:
        training_log.warning("Attention capture failed for layer=%d head=%d qkv_detail=%s: %s", layer, head, qkv_detail, e)
        return None


def _compute_activation_extras(vector: torch.Tensor) -> dict:
    """top_abs_components + value_slice for one representative tensor.

    Reuses `logits_last` (already computed for top-k) rather than a new
    capture mechanism, per the contract: "using tensors already captured."
    """
    with torch.no_grad():
        flat = vector.reshape(-1).float()
        top_k = min(5, flat.numel())
        top_vals, top_idxs = torch.topk(torch.abs(flat), k=top_k)
        return {
            "available": True,
            "top_abs_components": [
                {"index": int(i.item()), "value": float(flat[i].item())} for i in top_idxs
            ],
            "value_slice": [float(v) for v in flat[:8].tolist()],
        }


def _execute_forward_pass(
    session: "DiagnosticSession",
    top_k: int,
    attention_params: Optional[tuple[int, int]],
    qkv_detail: bool = False,
    append_token: bool = True,
) -> DiagnosticSnapshot:
    """Shared core for run_diagnostic_step and the Phase 3 final-token capture
    — one forward pass, tensor capture at hooked nodes, top-k, attention (if
    requested), activation extras. `append_token=False` captures the current
    state without advancing token_history (used for /generate's final frame,
    which has already appended its own tokens via its own sampling loop).
    """
    session.captured_tensors.clear()

    with torch.inference_mode():
        all_tokens = session.prompt_tokens + session.token_history
        idx = torch.tensor([all_tokens], dtype=torch.long, device=session.device)

        if "Moe" in session.model.__class__.__name__:
            logits, _, _ = session.model(idx)  # MoE returns (logits, loss, drop_rate)
        else:
            logits, _ = session.model(idx)

        logits_last = logits[0, -1, :]
        probs = torch.softmax(logits_last, dim=-1)
        topk_probs, topk_ids = torch.topk(probs, k=min(top_k, logits_last.shape[0]))

        if append_token:
            next_token_id = topk_ids[0].item()
            session.token_history.append(next_token_id)
            generated_token = {
                "position": len(session.prompt_tokens) + len(session.token_history) - 1,
                "id": next_token_id,
                "text": session.tokenizer.decode([next_token_id]),
            }
        else:
            last_id = session.token_history[-1] if session.token_history else session.prompt_tokens[-1]
            generated_token = {
                "position": len(session.prompt_tokens) + len(session.token_history) - 1,
                "id": last_id,
                "text": session.tokenizer.decode([last_id]),
            }

        input_tokens = [
            {"position": pos, "id": tid, "text": session.tokenizer.decode([tid])}
            for pos, tid in enumerate(session.prompt_tokens)
        ]

        nodes_dict = {
            node_id: {"input_shape": c.input_shape, "output_shape": c.output_shape, "summary": c.summary}
            for node_id, c in session.captured_tensors.items()
        }

        lm_head_data = {
            "logits_shape": list(logits.shape),
            "selected_position": len(session.prompt_tokens) + len(session.token_history) - 1,
            "top_k": [
                {
                    "rank": i + 1,
                    "token_id": int(tid.item()),
                    "token": session.tokenizer.decode([tid.item()]),
                    "logit": float(logits_last[tid].item()),
                    "probability": float(p.item()),
                }
                for i, (p, tid) in enumerate(zip(topk_probs, topk_ids))
            ],
        }

        attention_data = {"available": False, "reason": "Not requested"}
        if attention_params is not None:
            layer, head = attention_params
            result = _compute_attention_weights(session, layer, head, qkv_detail=qkv_detail)
            attention_data = result if result is not None else {"available": False, "reason": "Capture failed"}

        activation_data = _compute_activation_extras(logits_last)

        return DiagnosticSnapshot(
            diagnostic_session_id=session.session_id,
            generation_step=session.generation_step,
            input_tokens=input_tokens,
            generated_token=generated_token,
            nodes=nodes_dict,
            attention=attention_data,
            activation_summaries=activation_data,
            lm_head=lm_head_data,
        )


def run_diagnostic_step(
    session_id: str,
    top_k: int = 5,
    attention_params: Optional[tuple[int, int]] = None,
    qkv_detail: bool = False,
) -> Optional[DiagnosticSnapshot]:
    """Execute one autoregressive forward pass, appending one new token.

    Returns:
        DiagnosticSnapshot with shapes, summaries, top-k logits, and (if
        attention_params given) an explicit attention heatmap, or None on error.
    """
    session = get_session(session_id)
    if session is None:
        training_log.warning("Diagnostic step: session %s not found", session_id)
        return None

    session.generation_step += 1
    try:
        snapshot = _execute_forward_pass(session, top_k, attention_params, qkv_detail=qkv_detail, append_token=True)
        session.last_snapshot = snapshot
        return snapshot
    except Exception as e:
        training_log.error("Diagnostic step failed: %s", e, exc_info=True)
        return None


def run_diagnostic_step_internal(
    session_id: str,
    top_k: int = 5,
    attention_params: Optional[tuple[int, int]] = None,
    qkv_detail: bool = False,
    skip_token_generation: bool = True,
) -> Optional[DiagnosticSnapshot]:
    """Capture a snapshot for the CURRENT session state without appending a
    new token — used by Phase 3's /generate to produce the final snapshot
    after its own sampling loop has already advanced token_history.
    """
    session = get_session(session_id)
    if session is None:
        training_log.warning("Diagnostic step (internal): session %s not found", session_id)
        return None
    try:
        snapshot = _execute_forward_pass(session, top_k, attention_params, qkv_detail=qkv_detail, append_token=not skip_token_generation)
        session.last_snapshot = snapshot
        return snapshot
    except Exception as e:
        training_log.error("Diagnostic step (internal) failed: %s", e, exc_info=True)
        return None
