# Shared UI–Trainer Diagnostic Contract (Phase 1)

## Outstanding / not yet scoped (as of Phase 4, 2026-07-13)

Phases 1-4 cover the original spec doc's full 10-step implementation order.
Nothing below is scheduled — check here before starting new diagnostics work
so nothing already identified gets silently rebuilt or forgotten:

- **GET history route** — `/diagnostics/{session_id}` history/list endpoint.
  The Phase 4 write path (`diagnostic_sessions` table) exists; no read route
  to list past sessions for a run yet.
- **Events tab** — right pane still shows "Coming soon." Needs richer
  training-lifecycle event types (`worker.starting`, `checkpoint.saved`, etc.)
  threaded through `train_worker.py`/`runner.py`, not just the WS envelope
  wrapper already in place. Bigger lift than the other three items here.
- **MoE per-expert routing detail** — `block.{i}.moe` is still one opaque
  node; no per-token expert-assignment visualization.
- **RNN live diagnostics** — step-through (`>`/`>>`) is explicitly rejected
  for the RNN template (`"Step-through diagnostics not yet supported..."`).
  CharRNN's `forward(x, hc)` signature (one-hot input + threaded hidden
  state) is fundamentally different from transformer/moe's `forward(idx)`,
  so this needs its own design, not a copy of the existing path. Static
  architecture diagram already works for RNN.

Formalizes the coordination note agreed before backend and frontend work started in
parallel on 2026-07-13. **Read this before writing any diagnostic code.** Do not
invent a different schema independently — if something here is wrong or missing,
fix this file first, then build against the fix.

Source docs: `Trainer_to_Frontend_Metrics.md`, `Model_Diagram.md`. Those docs were
written without sight of this repo — where they conflict with what actually exists
here (see `docs/DESIGN_DECISIONS.md` §23 once written), this contract wins.

## Meaning of `>` and `>>` (confirmed with user 2026-07-13)

Both controls **only exist while a run is paused (or completed)** — never during
active training. They operate on a diagnostic session, not the training loop.

- **`>`** — one complete autoregressive forward pass: takes the current token
  sequence (prompt + any tokens already generated this session), runs the model
  once, selects exactly one next token, appends it, and returns one self-contained
  diagnostic snapshot for that pass. Pressing it repeatedly is what makes
  generation autoregressive — each press feeds the previous output back in. It does
  **not** mean stepping through one layer at a time; all layers run in that one
  pass, and the user inspects captured data from the full pass afterward (Embedding,
  Block 2, Attention, LM Head, etc. — all from the *same* snapshot).
- **`>>`** — continues generation from the current session state up to
  `max_new_tokens` or a stop condition, without waiting for a click per token.
  **Deferred to Phase 2** (see Scope below) — not built this round.

## Node IDs

Must be identical between backend hook names and frontend diagram node keys.
Confirmed against actual hook points in `backend/training/templates/transformer/model.py`
and `.../moe/model.py`:

```text
embedding
block.{i}.ln1
block.{i}.attention
block.{i}.ln2
block.{i}.mlp      # dense FFN — transformer template
block.{i}.moe      # MoE template only, replaces .mlp
final_norm
lm_head
```

`{i}` is zero-indexed, `0 .. n_layer-1`. Each pre-norm block has **two** LayerNorms
(pre-attention, pre-MLP) — `ln1`/`ln2` — this is one addition beyond the originally
pasted node list, needed because Model_Diagram.md's own per-block diagram shows both.

## Request/response shapes (Phase 1)

`POST /api/training/{local_run_id}/diagnostics/start`:

```json
{
  "prompt": "The king said",
  "top_k": 5,
  "max_prompt_tokens": 32
}
```
Rejected (400) unless the run's status is `paused` or `completed`. Returns
`{"diagnostic_session_id": "diag-17", "tokens": [...]}`.

`POST /api/training/{local_run_id}/diagnostics/{session_id}/step` — no body needed
(Phase 1 always captures shapes + top-k; capture-flag negotiation deferred):

```json
{
  "schema_version": 1,
  "diagnostic_session_id": "diag-17",
  "generation_step": 3,
  "input_tokens": [{"position": 0, "id": 51, "text": "The"}],
  "generated_token": {"position": 3, "id": 91, "text": " to"},
  "nodes": {
    "embedding": {
      "input_shape": [1, 3],
      "output_shape": [1, 3, 192],
      "summary": {"mean": 0.004, "std": 0.131, "l2_norm": 3.27, "min": -0.51, "max": 0.62}
    }
  },
  "attention": {"available": false, "reason": "Deferred to phase 2"},
  "activation_summaries": {"available": false, "reason": "Deferred to phase 2"},
  "lm_head": {
    "logits_shape": [1, 3, 65],
    "selected_position": 2,
    "top_k": [{"rank": 1, "token_id": 91, "token": " to", "logit": 6.21, "probability": 0.31}]
  },
  "complete": true
}
```

`GET /api/training/{local_run_id}/diagnostics/{session_id}` — returns the last
snapshot (for reconnect/refresh; does not advance the session).

See fixtures for full worked examples: `docs/fixtures/architecture_manifest.sample.json`,
`docs/fixtures/diagnostic_snapshot.sample.json`.

## Atomicity rule (frontend)

Never display partial data from two different forward passes together. Keep the
previous snapshot visible with a loading indicator until the new response arrives,
then replace the whole snapshot object atomically — not field by field.

## Local/remote dual-path (backend)

Follow the exact pattern already used by every other route in `backend/api/training.py`
(`_is_remote(db_run)` → either run locally or `await _proxy(...)`). The trainer
container runs the same FastAPI app, so writing these routes once in `training.py`
is sufficient — no separate trainer-side router. Confirmed via the existing
`prompt_paused_model()` route, which already works this way.

## Phase 1 scope boundary

**In scope:** architecture manifest (static, config-derived), tensor shapes at each
node, LM-head top-k, one-step (`>`) diagnostic sessions, in-memory session state
(no new DB table).

**Explicitly deferred (Phase 2+):** attention matrices/heatmap, Q/K/V detail,
activation summaries beyond shape, `>>` continue-generation with token streaming,
diagnostic session DB persistence, MoE per-expert routing detail (MoE block is one
opaque `block.{i}.moe` node with shape info only in Phase 1).

## Acceptance flow, Phase 1 (both agents tested against this)

```text
pause → enter prompt → click > → receive token and snapshot
→ inspect several nodes → click > again → snapshot advances by exactly one step
```

---

## Phase 2 scope: attention heatmap + richer activation summaries (2026-07-13)

Basic per-node stats (mean/std/l2_norm/min/max/shape, in `nodes.{id}.summary`)
were already live in Phase 1 — that satisfies the doc's "activation summaries"
item mostly. What's actually new in Phase 2:

**Attention** — one selected layer + head, explicit (non-fused) QKᵀ → scale →
causal mask → softmax → ×V path, per `Trainer_to_Frontend_Metrics.md`'s
warning that fused/Flash attention doesn't expose weights. Only computed when
requested (adds real cost); default the model's normal fast attention path
stays untouched for training and for steps that don't request it.

`POST /diagnostics/{session_id}/step` request body (was empty in Phase 1):
```json
{"attention_layer": 1, "attention_head": 0}
```
Response's `attention` field, when requested and available:
```json
{
  "available": true,
  "layer": 1, "head": 0,
  "weights": [[0.8, 0.2, 0.0], [0.3, 0.5, 0.2], [0.1, 0.3, 0.6]],
  "token_labels": ["The", " king", " said"]
}
```
`weights[i][j]` = how much position `i` attends to position `j` (row-normalized,
causal — upper triangle is 0). Omit `attention_layer`/`attention_head` in the
request → response stays `{"available": false, "reason": "Not requested"}`,
distinct from Phase 1's blanket "Deferred to phase 2" reason string.

**Activation summaries (richer)** — `activation_summaries` field, filled in
per node when requested, using tensors already captured by the existing
forward hooks (no new capture mechanism):
```json
{"available": true, "top_abs_components": [{"index": 42, "value": 3.1}, ...], "value_slice": [0.1, -0.2, 0.05, ...]}
```
`value_slice` = first 8 values of the flattened output tensor (illustrative,
not exhaustive — per the original doc: "Do not send every value").

**Not in Phase 2:** Q/K/V raw values (shapes only, already available via
existing per-node `summary`), MoE per-expert routing detail, `>>` (see below).

## Phase 3 scope: `>>` continue-generation (2026-07-13)

New endpoint, SSE (reuses the pattern already established in
`backend/api/chatbot.py`'s `event_stream()` — `data: {json}\n\n` frames,
`event: done` terminator — rather than the WS envelope, since this is a
one-shot user-triggered stream, not the continuous training-metrics channel):

```text
POST /api/training/{local_run_id}/diagnostics/{session_id}/generate
```
Request: `{"max_new_tokens": 50}`. Streams one event per generated token:
```text
event: token
data: {"position": 4, "id": 12, "text": " to", "generation_step": 4}

event: done
data: {"final_snapshot": {...full DiagnosticSnapshot, same shape as a `/step` response...}}
```
Full diagnostic capture (shapes + whatever attention/activation flags were
requested) only happens for the **final** token, not every token in between
— per the original doc's explicit cost warning. Intermediate `token` events
carry just the token itself, no per-node capture.

Frontend: `>>` button (currently disabled "Coming soon") subscribes to this
stream, appends each token to the displayed generated sequence as it arrives,
and swaps in the final full snapshot atomically when `done` fires — same
atomicity rule as `>|`. If the user clicks `>` again after a `>>` run
finishes, it continues from the session's now-longer token history, exactly
like today's `>`-only flow.

**Not in Phase 3:** Events tab stays "Coming soon" — wiring the richer
training-lifecycle event vocabulary (`worker.starting`, `checkpoint.saved`,
etc. — currently the WS envelope only emits `metric`/`status`/`done`/`error`)
is a bigger, separate lift through `train_worker.py`/`runner.py` and is
explicitly deferred past Phase 3, not silently dropped — flag this to the
user rather than letting it look finished.

## Phase 4 scope: Q/K/V detail + optional persistence (2026-07-13)

Lowest-priority tier per the original spec doc — build only what's listed.

**Q/K/V detail** — extends the Phase 2 attention request. When
`attention_layer`/`attention_head` are given AND a new flag `qkv_detail: true`
is set, the step response's `attention` object gains:
```json
{
  "qkv_detail": {
    "position": 2,
    "q": [0.12, -0.34, ...],
    "k": [0.08, 0.21, ...],
    "v": [-0.15, 0.42, ...]
  }
}
```
One token position only (the last position — the one about to attend), one
head's Q/K/V vectors (length = head_size), not the full sequence — per the
original doc: "Do not send every Q/K/V value for every token and head."
Reuse `_compute_attention_weights`'s existing partial-forward-pass machinery
(it already computes `q`, `k`, `v` internally) — don't recompute from scratch.
Default (`qkv_detail` omitted or false): unchanged Phase 2 behavior, no cost.

**Session persistence (optional)** — a new DB table
`diagnostic_sessions(id, run_id, prompt, generated_output, generation_params_json,
top_k_summary_json, created_at)`, written once when a `/generate` stream's
`done` event fires (not on every `/step` — only final outcomes, per the
original doc: "Do not necessarily persist full attention matrices or
activation tensors... reproducible while the endpoint and checkpoint remain
available"). No new route needed to fetch history in this phase — just the
write path; a GET-history route is a natural Phase 5, not required now.

## Scope-fencing rule for both agents (added after a Phase 1 incident, reinforced after Phase 2)

**Only touch the files explicitly listed in your task brief.** If you think
a fix or improvement to some other function would help, do not make it —
write it in your final report instead and let the reviewer decide. Phase 1's
backend agent rewrote several unrelated functions while adding new routes
(silently deleting a concurrency-limit safety check in the process) — see
`docs/DESIGN_DECISIONS.md` §25. Phase 2's backend agent's worktree started
from a snapshot that didn't reliably reflect the real current
`backend/api/training.py`/`diagnostics.py`, so it partly reconstructed
`get_architecture_manifest()` from imagination instead of truly extending
the real file — missing the RNN branch and calling the non-existent
`model.numel()`. **Before writing anything, `Read` the actual current file
at the exact path given below — do not reconstruct logic from this contract
description alone. If what you read doesn't match what this contract
describes, the real file wins; flag the mismatch in your report.** See
`docs/DESIGN_DECISIONS.md` §26 for the full incident.
