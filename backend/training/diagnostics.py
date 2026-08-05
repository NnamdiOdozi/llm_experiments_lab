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

# Per-position diagnostic data (lm_head.top_k_by_position, attention.qkv_detail)
# is capped to the most recent N positions — a long-running step-through
# session can accumulate far more tokens than are useful to browse in the
# Inspector's position stepper, and every position multiplies the response
# payload (each position needs its own decoded top-5 + Q/K/V vectors). See
# docs/DESIGN_DECISIONS.md.
DIAGNOSTIC_POSITION_WINDOW = 12

# torch.softmax(logits / temperature, ...) divides by this directly —
# temperature=0 produces inf/nan and crashes torch.multinomial mid-
# generation. Direct user request, 2026-07-15: clamp to a tiny epsilon at
# the point of use rather than rejecting the config value outright — 0 (or
# a negative value, from e.g. a stray minus key) is silently treated as
# "as sharp/greedy-like as sampling allows" instead of erroring. See
# docs/DESIGN_DECISIONS.md.
MIN_TEMPERATURE = 1e-6


@dataclass
class NodeCapture:
    """Captured tensor information at a node (hook point)."""
    input_shape: list[int]
    output_shape: list[int]
    summary: dict  # {mean, std, l2_norm, min, max}
    # Raw per-position output vectors, last DIAGNOSTIC_POSITION_WINDOW
    # positions only: {"positions": [...], "vectors": [[...], ...]}. None
    # for non-3D outputs (e.g. lm_head's [B,T,vocab_size] is handled
    # separately via DiagnosticSnapshot.lm_head, not here). Captured for
    # every node on every step (not just the currently-selected one) — user
    # explicitly chose this over per-node-on-demand for instant node
    # switching with no re-fetch, accepting the larger response size. See
    # docs/DESIGN_DECISIONS.md.
    position_vectors: Optional[dict] = None
    # Same shape/window as position_vectors, but for the node's INPUT
    # rather than output — e.g. what went into a LayerNorm, not just what
    # came out. Real gap flagged live (2026-07-15): output-only vectors
    # were shown everywhere, without the corresponding input to compare
    # against. See docs/DESIGN_DECISIONS.md.
    input_position_vectors: Optional[dict] = None


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
    # Windowed [{position, id, token}, ...] — same window as position_vectors
    # on every node. See docs/DESIGN_DECISIONS.md.
    position_tokens: list[dict] = field(default_factory=list)
    # Eagerly-captured attention weights for ALL layers × ALL heads on every step,
    # windowed to the most recent DIAGNOSTIC_POSITION_WINDOW positions.
    # Schema: {available, window_start, total_positions, token_labels, n_layer, n_head, weights}
    # Enables instant block/head switching in the Inspector without round-trip peeks.
    attention_maps: dict = field(default_factory=lambda: {"available": False, "reason": "Not captured"})
    # Full-context attention matrix for canvas heatmap — requested on-demand via
    # attention_full=True in peek, populated only when explicitly requested. Contains
    # full T×T matrix (capped to block_size) for one selected (layer, head) pair.
    # Schema: {available, layer, head, weights, token_labels, total_positions, block_size}
    # See docs/DESIGN_DECISIONS.md.
    attention_full: dict = field(default_factory=lambda: {"available": False, "reason": "Not requested"})
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
            "position_tokens": self.position_tokens,
            "attention_maps": self.attention_maps,
            "attention_full": self.attention_full,
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
    # Read from config.inference.temperature at session start — same source
    # and value model.generate() (the Generate button) uses. Live-overridable
    # afterward: the /step and /generate routes mutate this in place when
    # the request includes an explicit temperature/decoding_mode, so the
    # user can adjust mid-prompting without losing token_history by
    # restarting the session. Direct user request, 2026-07-15. See
    # docs/DESIGN_DECISIONS.md.
    temperature: float = 0.8
    # "greedy" or "sample" — read from config.inference.decoding_mode at
    # session start, same setting model.generate() (Generate button) uses.
    # Live-overridable — see temperature above.
    decoding_mode: str = "sample"
    # Shifts the position_vectors/input_position_vectors window captured by
    # every node's forward hook — same semantics as attention_window_offset
    # (0 = most recent DIAGNOSTIC_POSITION_WINDOW positions, positive N
    # shifts back N). Unlike attention_window_offset (a plain function
    # parameter threaded through _execute_forward_pass, since attention is
    # computed by an explicit separate call, not a hook), node capture
    # happens inside forward hooks — registered once at session start, with
    # no way to receive per-call arguments beyond PyTorch's (module, input,
    # output) signature. Session-level mutable state is the only way to
    # reach them: the /step, /peek, and /generate routes mutate this in
    # place right before calling session.model(...), and each hook closure
    # reads it back via get_session(session_id) at capture time. Direct
    # user request, 2026-07-15: "there should be a stepper that allows that
    # window to slide backwards in time" for generic nodes (LayerNorm,
    # MLP, embedding, final_norm), not just attention. See
    # docs/DESIGN_DECISIONS.md.
    node_window_offset: int = 0
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
    temperature: float = 0.8,
    decoding_mode: str = "sample",
) -> str:
    """Create a new diagnostic session and register it.

    Args:
        model: Loaded PyTorch model in eval mode
        tokenizer: CharDataset or equivalent with encode/decode
        device: "cpu" or "cuda"
        prompt_tokens: Initial prompt tokens
        run_id: Training run this session belongs to (for chatbot grounding lookup)
        temperature: config.inference.temperature — same value model.generate() uses
        decoding_mode: "greedy" or "sample" — config.inference.decoding_mode

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
        temperature=temperature,
        decoding_mode=decoding_mode,
        token_history=[],
    )
    _diagnostic_sessions[session_id] = session
    if run_id is not None:
        # Evict this run's PREVIOUS session (if any) — each session holds a
        # full model loaded from checkpoint, and nothing else ever pruned
        # the registry (delete_session had zero callers), so every prompt
        # session leaked a model's worth of RAM until process restart.
        # Per-run eviction on new-session start, NOT a TTL, is deliberate:
        # the frontend relies on a closed session staying peekable
        # (PausePrompt.closeSession keeps the id alive for Inspector's
        # post-close peeks) right up until a new prompt replaces it — which
        # is exactly this moment. Fable review, 2026-07-14. See
        # docs/DESIGN_DECISIONS.md §66.
        previous = _run_to_session.get(run_id)
        if previous is not None and previous != session_id:
            delete_session(previous)
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


def _get_position_vectors(tensor: torch.Tensor, offset: int = 0) -> Optional[dict]:
    """Raw per-position vectors, DIAGNOSTIC_POSITION_WINDOW positions wide.

    Handles any 3D [B, T, *] tensor (2D inputs like embedding's token-id
    lookup return None, matching the fact there's no meaningful "input vector"
    for a token id). `offset` shifts the window earlier — same semantics as
    attention's window_offset (0 = most recent, positive N = shift back N).

    Used by both transformer and MoE templates via make_hook_for_diagnostics.
    See docs/DESIGN_DECISIONS.md.
    """
    if tensor.dim() != 3:
        return None
    with torch.no_grad():
        T = tensor.shape[1]
        window = min(T, DIAGNOSTIC_POSITION_WINDOW)
        end = max(window, T - max(offset, 0))
        start = end - window
        return {"positions": list(range(start, end)), "vectors": tensor[0, start:end, :].tolist()}


def make_hook_for_diagnostics(node_id: str, session_id: str):
    """Factory for forward hooks that capture tensor info for a diagnostic session.

    The returned hook handles both plain tensors (transformer) and tuples where
    the first element is a tensor (MoE blocks return (x, drop_rate)) — it unpacks
    tuples before computing summaries and position vectors.

    Used by both transformer and MoE templates. Closures reference session_id so
    they store captures in the correct session's captured_tensors dict.
    """
    def hook(module, input_tuple, output):
        session = get_session(session_id)
        if session is None:
            return

        offset = session.node_window_offset
        input_tensor = input_tuple[0] if input_tuple else None
        input_shape = list(input_tensor.shape) if isinstance(input_tensor, torch.Tensor) else []
        input_position_vectors = (
            _get_position_vectors(input_tensor, offset) if isinstance(input_tensor, torch.Tensor) else None
        )

        # Handle both plain tensors (transformer) and tuples (MoE)
        if isinstance(output, tuple) and len(output) > 0:
            x = output[0]
            output_shape = list(x.shape) if isinstance(x, torch.Tensor) else []
            summary = _compute_summary(x) if isinstance(x, torch.Tensor) else {}
            position_vectors = _get_position_vectors(x, offset) if isinstance(x, torch.Tensor) else None
        elif isinstance(output, torch.Tensor):
            output_shape = list(output.shape)
            summary = _compute_summary(output)
            position_vectors = _get_position_vectors(output, offset)
        else:
            output_shape = []
            summary = {}
            position_vectors = None

        session.captured_tensors[node_id] = NodeCapture(
            input_shape=input_shape,
            output_shape=output_shape,
            summary=summary,
            position_vectors=position_vectors,
            input_position_vectors=input_position_vectors,
        )
    return hook


def register_diagnostic_hooks(model: torch.nn.Module, session_id: str) -> None:
    """Register forward hooks at diagnostic node points.

    Delegates to the model's register_diagnostic_hooks method, which knows
    about the model's internal structure. The returned handles are stored
    on the session so delete_session() can actually deregister them —
    previously both templates discarded the handles, so hook_handles was
    always empty. Fable review, 2026-07-14 — see docs/DESIGN_DECISIONS.md §66.

    Args:
        model: The loaded model (must have register_diagnostic_hooks method)
        session_id: Session to receive captures
    """
    if hasattr(model, 'register_diagnostic_hooks'):
        handles = model.register_diagnostic_hooks(session_id)
        session = get_session(session_id)
        if session is not None and handles:
            session.hook_handles.extend(handles)


def _compute_attention_weights(
    session: "DiagnosticSession",
    layer: int,
    head: int,
    qkv_detail: bool = False,
    window_offset: int = 0,
    full_matrix: bool = False,
) -> Optional[dict]:
    """Explicit (non-fused) QK^T -> scale -> causal mask -> softmax attention path.

    The model's normal forward pass uses fused/fast attention, which per
    Trainer_to_Frontend_Metrics.md doesn't expose weights — this recomputes
    just the requested layer/head manually, only when asked for (Phase 2).

    When full_matrix=False (default, windowed mode): The heatmap itself
    (`weights`/`token_labels`) is windowed to the last DIAGNOSTIC_POSITION_WINDOW
    positions on both axes (a square block, not just the rows) — previously
    unwindowed, so a long session rendered an ever-growing T x T grid that got
    "very busy very quickly" (real user report, 2026-07-13). `window_offset`
    shifts which block of the sequence is shown: 0 = most recent (default),
    positive N = shift the window back N positions, so the frontend can step
    earlier/later through history instead of only ever seeing the tail. qkv_detail
    (when requested) shares the exact same window — one stepper controls both. See
    docs/DESIGN_DECISIONS.md.

    When full_matrix=True (canvas mode): Returns the ENTIRE T×T matrix (or
    min(T, block_size) if session exceeded block_size) un-windowed, for the
    canvas heatmap. token_labels and total_positions describe the full context
    the model saw. window_offset is ignored in this mode. qkv_detail is not
    populated (Q/K/V remain windowed in the Q/K/V browser).

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
        # Only use the last block_size tokens (what the model actually saw).
        # See _execute_forward_pass comment in backend/training/diagnostics.py §75
        # for why this matters: a session can exceed block_size, but the model
        # never receives input longer than block_size due to the windowing in
        # model.generate()'s idx[:, -self.block_size:].
        block_size = session.model.block_size
        visible_tokens = all_tokens[-block_size:]
        idx = torch.tensor([visible_tokens], dtype=torch.long, device=session.device)

        with torch.inference_mode():
            x = model.token_emb(idx)
            if hasattr(model, "pos_emb"):
                x = x + model.pos_emb(torch.arange(x.shape[1], device=session.device))
            for i in range(layer):
                x = model.blocks[i](x)
                # BlockMoe.forward returns (x, drop_rate), not a bare tensor
                # — without unwrapping, any MoE layer >= 1 crashed on
                # `B, T, C = x.shape` below and the broad except turned it
                # into a permanent "Capture failed" for blocks 2..N. Fable
                # review, 2026-07-14 — see docs/DESIGN_DECISIONS.md §67.
                if isinstance(x, tuple):
                    x = x[0]

            B, T, C = x.shape
            # Use shared core (§67/§68 fixes).
            q, k, v = _layer_qkv(block, x)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(attn.head_size)
            causal_mask = torch.tril(torch.ones(T, T, device=session.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask, float("-inf"))
            full_weights = torch.softmax(scores, dim=-1)[0, head]

            if full_matrix:
                # Full matrix mode: return entire T×T, un-windowed
                weights = full_weights.tolist()
                token_labels = [session.tokenizer.decode([tid]) for tid in visible_tokens]
                result = {
                    "available": True,
                    "layer": layer,
                    "head": head,
                    "weights": weights,
                    "token_labels": token_labels,
                    "total_positions": T,
                    "block_size": block_size,
                }
                # No qkv_detail in full_matrix mode
                return result
            else:
                # Windowed mode (classic): 12-position window for heatmap + qkv_detail
                # Shared window for the heatmap and qkv_detail — offset=0 shows
                # the most recent `window` positions; positive offset shifts the
                # window's end earlier by that many positions. Clamped so the
                # window never goes out of [0, T].
                window = min(T, DIAGNOSTIC_POSITION_WINDOW)
                end = max(window, T - max(window_offset, 0))
                start = end - window

                weights = full_weights[start:end, start:end].tolist()
                token_labels = [session.tokenizer.decode([tid]) for tid in visible_tokens[start:end]]
                result = {
                    "available": True,
                    "layer": layer,
                    "head": head,
                    "weights": weights,
                    "token_labels": token_labels,
                    "window_start": start,
                    "total_positions": T,
                }

                # Q/K/V detail — one vector per position, for the frontend's
                # position stepper — over the exact same window as the heatmap.
                if qkv_detail:
                    result["qkv_detail"] = {
                        "positions": list(range(start, end)),
                        "tokens": [session.tokenizer.decode([tid]) for tid in visible_tokens[start:end]],
                        "q": q[0, head, start:end, :].tolist(),
                        "k": k[0, head, start:end, :].tolist(),
                        "v": v[0, head, start:end, :].tolist(),
                    }

                return result
    except Exception as e:
        training_log.warning("Attention capture failed for layer=%d head=%d qkv_detail=%s full_matrix=%s: %s", layer, head, qkv_detail, full_matrix, e)
        return None


def _layer_qkv(block, x):
    """Shared core: given a block and its input x (post-prior-blocks), extract
    Q/K/V with ln1 applied and RoPE if configured (DESIGN_DECISIONS §67/§68).

    Returns (q, k, v) shaped [B, n_head, T, head_size], ready for attention.
    No error handling here — exceptions bubble up to the caller.
    """
    attn = block.attn
    B, T, C = x.shape
    x_ln = block.ln1(x)
    q, k, v = attn.qkv(x_ln).split(C, dim=-1)
    q = q.view(B, T, attn.n_head, attn.head_size).transpose(1, 2)
    k = k.view(B, T, attn.n_head, attn.head_size).transpose(1, 2)
    v = v.view(B, T, attn.n_head, attn.head_size).transpose(1, 2)
    if getattr(attn, "pos_encoding", "learned") == "rope":
        q = attn.rope(q)
        k = attn.rope(k)
    return q, k, v


def _compute_all_attention(
    session: "DiagnosticSession",
    window_offset: int = 0,
) -> dict:
    """Eagerly compute attention weights for ALL layers × ALL heads in ONE pass.

    Single forward propagation through the model; at each layer, extract Q/K/V
    and compute attention for all heads before advancing to the next layer.

    Returns a dict with keys:
    - available: True if successful
    - window_start, total_positions: window metadata
    - token_labels: tokens in the windowed range
    - n_layer, n_head: counts
    - weights: [layer][head][w][w] list of windowed softmax weights
    Or {"available": False, "reason": "..."} on error.
    """
    try:
        model = session.model
        if not hasattr(model, "blocks") or len(model.blocks) == 0:
            return {"available": False, "reason": "Model has no blocks"}

        all_tokens = session.prompt_tokens + session.token_history
        n_layer = len(model.blocks)
        device = session.device

        with torch.inference_mode():
            idx = torch.tensor([all_tokens], dtype=torch.long, device=device)

            # Embed and apply positional encoding (once)
            x = model.token_emb(idx)
            if hasattr(model, "pos_emb"):
                x = x + model.pos_emb(torch.arange(x.shape[1], device=device))

            T = x.shape[1]
            # Shared windowing: same logic as _compute_attention_weights
            window = min(T, DIAGNOSTIC_POSITION_WINDOW)
            end = max(window, T - max(window_offset, 0))
            start = end - window

            token_labels = [session.tokenizer.decode([tid]) for tid in all_tokens[start:end]]

            # Weights array: [layer][head][window][window]
            weights_all_layers = []
            # QKV array: [layer][head] = {q, k, v per-position arrays}
            qkv_all_layers = []

            # Single pass through all blocks
            for layer in range(n_layer):
                block = model.blocks[layer]
                attn = block.attn

                # Extract Q/K/V using shared core (§67/§68 fixes applied here)
                q, k, v = _layer_qkv(block, x)
                head_size = attn.head_size
                n_head = attn.n_head

                # Compute attention for all heads in this layer
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_size)
                causal_mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
                scores = scores.masked_fill(~causal_mask, float("-inf"))
                full_weights = torch.softmax(scores, dim=-1)[0]  # [n_head, T, T]

                # Window and convert to list for each head
                weights_this_layer = [
                    full_weights[head, start:end, start:end].tolist()
                    for head in range(n_head)
                ]
                weights_all_layers.append(weights_this_layer)

                # Capture windowed Q/K/V for all heads (same window as weights/heatmap).
                # q/k/v are [B, n_head, T, head_size]; one window slice per head,
                # rounded to 4dp. NOTE: don't name a loop variable q/k/v/x here —
                # those are the live tensors.
                def _rounded_rows(t):
                    return [[round(val, 4) for val in row] for row in t.tolist()]
                qkv_this_layer = [
                    {
                        "q": _rounded_rows(q[0, head, start:end, :]),
                        "k": _rounded_rows(k[0, head, start:end, :]),
                        "v": _rounded_rows(v[0, head, start:end, :]),
                    }
                    for head in range(n_head)
                ]
                qkv_all_layers.append(qkv_this_layer)

                # Advance to next layer (must handle MoE tuple return; see DESIGN_DECISIONS §67)
                x = block(x)
                if isinstance(x, tuple):
                    x = x[0]

            return {
                "available": True,
                "window_start": start,
                "total_positions": T,
                "token_labels": token_labels,
                "positions": list(range(start, end)),
                "n_layer": n_layer,
                "n_head": n_head,
                "weights": weights_all_layers,
                "qkv": qkv_all_layers,
            }
    except Exception as e:
        training_log.warning("All-attention capture failed: %s", e)
        return {"available": False, "reason": "Capture failed"}


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
    attention_window_offset: int = 0,
    attention_full: bool = False,
) -> DiagnosticSnapshot:
    """Shared core for run_diagnostic_step and the Phase 3 final-token capture
    — one forward pass, tensor capture at hooked nodes, top-k, attention (if
    requested), activation extras. `append_token=False` captures the current
    state without advancing token_history (used for /generate's final frame,
    which has already appended its own tokens via its own sampling loop).
    When attention_full=True, computes full T×T matrix for the selected
    (layer, head) pair (on-demand, for canvas heatmap).
    """
    session.captured_tensors.clear()

    with torch.inference_mode():
        all_tokens = session.prompt_tokens + session.token_history
        # Real bug, 2026-07-15: this path never windowed to block_size, unlike
        # both templates' own model.generate() (`idx[:, -self.block_size:]`)
        # — a long diagnostic session (prompt + token_history > block_size)
        # crashed inside the model with a tensor-size mismatch on whichever
        # buffer is sized for exactly block_size (RoPE table / causal mask).
        # Reassigning all_tokens itself (not just a separate slice for idx)
        # keeps every downstream use in this function — top_k_by_position,
        # position_tokens, node capture — consistent with the same window
        # the model actually saw. See docs/DESIGN_DECISIONS.md.
        all_tokens = all_tokens[-session.model.block_size:]
        idx = torch.tensor([all_tokens], dtype=torch.long, device=session.device)

        if "Moe" in session.model.__class__.__name__:
            logits, _, _ = session.model(idx)  # MoE returns (logits, loss, drop_rate)
        else:
            logits, _ = session.model(idx)

        if append_token:
            # Pre-append distribution — all_tokens/idx above do NOT yet
            # include the token being sampled here, so this IS the exact
            # distribution generated_token gets drawn from. See the
            # top_k_by_position comment below for why this pre-append
            # timing is deliberate, not incidental.
            logits_last = logits[0, -1, :]
            probs = torch.softmax(logits_last, dim=-1)
            # Same recipe as model.generate() (the Generate button), and
            # same decoding_mode setting — greedy always picks the single
            # highest-probability token (temperature has no effect, order
            # is scale-invariant); sample scales logits by temperature then
            # draws from the resulting distribution. top_k above always
            # shows the model's raw (unscaled) confidence regardless of
            # which mode is active. See docs/DESIGN_DECISIONS.md.
            if session.decoding_mode == "greedy":
                next_token_id = torch.argmax(logits_last).item()
            else:
                sample_probs = torch.softmax(logits_last / max(session.temperature, MIN_TEMPERATURE), dim=-1)
                next_token_id = torch.multinomial(sample_probs, num_samples=1).item()
            session.token_history.append(next_token_id)
            generated_token = {
                "position": len(session.prompt_tokens) + len(session.token_history) - 1,
                "id": next_token_id,
                "text": session.tokenizer.decode([next_token_id]),
            }
        else:
            last_id = session.token_history[-1] if session.token_history else session.prompt_tokens[-1]
            # Real bug, 2026-07-15: this branch is used for >>'s final-frame
            # capture (called AFTER >>'s own loop already appended last_id
            # to token_history/all_tokens) and for /peek. Previously reused
            # logits[0, -1, :] here too — but by this point all_tokens
            # already ends in last_id, so that position predicts what comes
            # NEXT (one token ahead of last_id), not the distribution that
            # actually produced last_id. Inspector's LM Head "selected"
            # highlight compares top_k's token ids against
            # generated_token.id — with the wrong position, they could
            # never match, so nothing ever highlighted green after >>
            # (single-step > was unaffected — see the append_token branch
            # above). logits[0, -2, :] is the fix: a causal model's output
            # at index i is always "prediction after seeing input[0..i]", so
            # position T-2 (predicting position T-1) is exactly the
            # distribution all_tokens[T-1] (== last_id) was actually drawn
            # from — already computed in this same forward pass, no extra
            # cost. See docs/DESIGN_DECISIONS.md.
            source_position = -2 if logits.shape[1] >= 2 else -1
            logits_last = logits[0, source_position, :]
            probs = torch.softmax(logits_last, dim=-1)
            generated_token = {
                "position": len(session.prompt_tokens) + len(session.token_history) - 1,
                "id": last_id,
                "text": session.tokenizer.decode([last_id]),
            }

        topk_probs, topk_ids = torch.topk(probs, k=min(top_k, logits_last.shape[0]))

        input_tokens = [
            {"position": pos, "id": tid, "text": session.tokenizer.decode([tid])}
            for pos, tid in enumerate(session.prompt_tokens)
        ]

        # Per-position top-k for the frontend's position stepper — reuses
        # `logits` (already computed above for every position, `top_k`
        # above only ever kept the last one) rather than a second forward
        # pass. NOT capped to DIAGNOSTIC_POSITION_WINDOW (unlike
        # position_vectors/qkv_detail below) — see the top_k_pos_start
        # comment a few lines down for why.
        #
        # Note: this uses the PRE-append `all_tokens`/`logits` (computed
        # above, before token_history gets the new token appended a few
        # lines up in the append_token branch) — so its window ends one
        # position earlier than attention.qkv_detail's, which is computed
        # AFTER the append via its own fresh forward pass. That's correct,
        # not a bug: the just-generated token has real Q/K/V (it's part of
        # the sequence now, shows in the heatmap) but no "what comes next"
        # prediction yet — computing one would need another forward pass,
        # and that prediction is exactly what the NEXT step naturally
        # produces. Each dataset carries its own explicit "position" labels
        # (not a shared implicit index) for exactly this reason — the
        # frontend stepper for each section reads that section's own
        # positions, not a forced-shared range. See docs/DESIGN_DECISIONS.md.
        T_total = logits.shape[1]
        pos_window = min(T_total, DIAGNOSTIC_POSITION_WINDOW)
        pos_start = T_total - pos_window

        # Real bug, 2026-07-15: the fix a few lines up (source_position)
        # only corrected the FLAT lm_head.top_k field — Inspector's LM Head
        # panel actually reads top_k_by_position instead (LmHeadStepper in
        # Inspector.tsx defaults to its LAST entry to decide the "selected/
        # generated" highlight). That loop always ran through the full
        # pos_start..T_total range regardless of branch, so for
        # append_token=False its last entry was still the same one-ahead
        # "predict what comes after the complete sequence" position — the
        # comment above (pos_start/T_total) describing "this window ends
        # one position earlier... that's correct, not a bug" is only true
        # for append_token=True (where all_tokens/logits are the PRE-append
        # state). For append_token=False, all_tokens already ends in the
        # last generated token, so the same off-by-one applies here too.
        # top_k_end mirrors source_position's fallback exactly. pos_start
        # (used below for position_tokens) is deliberately NOT changed —
        # that field describes INPUT tokens, correct at every position
        # regardless of branch, no off-by-one there. See
        # docs/DESIGN_DECISIONS.md.
        top_k_end = T_total if (append_token or T_total < 2) else T_total - 1
        # Direct user request, 2026-07-15: no window here — unlike
        # position_vectors/qkv_detail (real per-position vectors, capped to
        # DIAGNOSTIC_POSITION_WINDOW for response-size/cost reasons), each
        # top_k_by_position entry is just a top-5 list of small scalars, so
        # showing every position all the way back to the start of the
        # captured sequence (not just the last 12) is cheap and was purely
        # an unnecessary restriction. See docs/DESIGN_DECISIONS.md.
        top_k_pos_start = 0

        # Direct user request, 2026-07-15: highlight the actually-selected
        # token at EVERY browsable position, not just the most recent one
        # — previously the frontend only ever compared against
        # generated_token.id (a single token, only meaningful for the
        # latest position) and hardcoded isMostRecent as a prerequisite.
        # Ground truth for "what token actually came next" after each
        # position is reconstructible here: for every already-historical
        # position, it's simply the next entry in all_tokens; for the
        # newest position under append_token=True specifically, all_tokens
        # is the PRE-append array, so its own "next" is next_token_id
        # (not yet appended). Attaching it per-entry lets the frontend do a
        # simple, uniform token_id comparison at any position, no more
        # isMostRecent special-casing. See docs/DESIGN_DECISIONS.md.
        full_next_tokens = all_tokens[1:] + [next_token_id] if append_token else all_tokens[1:]

        top_k_by_position = []
        for pos in range(top_k_pos_start, top_k_end):
            pos_logits = logits[0, pos, :]
            pos_probs = torch.softmax(pos_logits, dim=-1)
            pos_topk_probs, pos_topk_ids = torch.topk(pos_probs, k=min(top_k, pos_logits.shape[0]))
            top_k_by_position.append({
                "position": pos,
                "token": session.tokenizer.decode([all_tokens[pos]]),
                "actual_next_token_id": full_next_tokens[pos] if pos < len(full_next_tokens) else None,
                "top_k": [
                    {
                        "rank": r + 1,
                        "token_id": int(tid.item()),
                        "token": session.tokenizer.decode([tid.item()]),
                        "logit": float(pos_logits[tid].item()),
                        "probability": float(p.item()),
                    }
                    for r, (p, tid) in enumerate(zip(pos_topk_probs, pos_topk_ids))
                ],
            })

        # Windowed token id + text per position — same pre-append window as
        # top_k_by_position/position_vectors above. Exists specifically for
        # the embedding node's Inspector view, which needs the actual input
        # token id at each position to build a one-hot vector (direct user
        # request 2026-07-15: "position, then the character, and then the
        # one-hot vector"). Not attached per-node like position_vectors —
        # every node in a single forward pass shares this exact same
        # window, so one shared list avoids repeating identical token text
        # at every one of ~18 nodes. See docs/DESIGN_DECISIONS.md.
        position_tokens = [
            {"position": pos, "id": all_tokens[pos], "token": session.tokenizer.decode([all_tokens[pos]])}
            for pos in range(pos_start, T_total)
        ]

        nodes_dict = {
            node_id: {
                "input_shape": c.input_shape, "output_shape": c.output_shape, "summary": c.summary,
                "position_vectors": c.position_vectors,
                "input_position_vectors": c.input_position_vectors,
            }
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
            "top_k_by_position": top_k_by_position,
        }

        attention_data = {"available": False, "reason": "Not requested"}
        if attention_params is not None:
            layer, head = attention_params
            result = _compute_attention_weights(session, layer, head, qkv_detail=qkv_detail, window_offset=attention_window_offset)
            attention_data = result if result is not None else {"available": False, "reason": "Capture failed"}

        # Eagerly capture all-layer × all-head attention on every step/peek.
        # Round to 4 decimals to keep payload reasonable (~25-35KB).
        all_attention_maps = _compute_all_attention(session, window_offset=attention_window_offset)
        if all_attention_maps.get("available"):
            # Round all weights to 4 decimals
            for layer_weights in all_attention_maps.get("weights", []):
                for head_weights in layer_weights:
                    for row in head_weights:
                        for i in range(len(row)):
                            row[i] = round(row[i], 4)

        # Full-context matrix for canvas heatmap (on-demand, only when requested)
        full_matrix_data = {"available": False, "reason": "Not requested"}
        if attention_full and attention_params is not None:
            layer, head = attention_params
            result = _compute_attention_weights(session, layer, head, qkv_detail=False, window_offset=0, full_matrix=True)
            full_matrix_data = result if result is not None else {"available": False, "reason": "Capture failed"}

        activation_data = _compute_activation_extras(logits_last)

        return DiagnosticSnapshot(
            diagnostic_session_id=session.session_id,
            generation_step=session.generation_step,
            input_tokens=input_tokens,
            generated_token=generated_token,
            nodes=nodes_dict,
            attention=attention_data,
            attention_maps=all_attention_maps,
            attention_full=full_matrix_data,
            activation_summaries=activation_data,
            lm_head=lm_head_data,
            position_tokens=position_tokens,
        )


def run_diagnostic_step(
    session_id: str,
    top_k: int = 5,
    attention_params: Optional[tuple[int, int]] = None,
    qkv_detail: bool = False,
    attention_window_offset: int = 0,
    attention_full: bool = False,
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
        snapshot = _execute_forward_pass(
            session, top_k, attention_params, qkv_detail=qkv_detail, append_token=True,
            attention_window_offset=attention_window_offset, attention_full=attention_full,
        )
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
    attention_window_offset: int = 0,
    attention_full: bool = False,
) -> Optional[DiagnosticSnapshot]:
    """Capture a snapshot for the CURRENT session state without appending a
    new token — used by Phase 3's /generate to produce the final snapshot
    after its own sampling loop has already advanced token_history, and by
    /peek to recompute attention (including a shifted window) without
    advancing anything.
    """
    session = get_session(session_id)
    if session is None:
        training_log.warning("Diagnostic step (internal): session %s not found", session_id)
        return None
    try:
        snapshot = _execute_forward_pass(
            session, top_k, attention_params, qkv_detail=qkv_detail, append_token=not skip_token_generation,
            attention_window_offset=attention_window_offset, attention_full=attention_full,
        )
        session.last_snapshot = snapshot
        return snapshot
    except Exception as e:
        training_log.error("Diagnostic step (internal) failed: %s", e, exc_info=True)
        return None
