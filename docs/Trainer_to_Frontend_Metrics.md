Briefing Note 2: Trainer-to-Frontend Metrics and Diagnostic Data Pipeline
Objective

Design the data path from the trainer container running on a Nebius endpoint to the React frontend.

The system must support two very different traffic types:

Continuous lightweight training telemetry
    loss, step, throughput, status and events

On-demand diagnostic snapshots
    tensor shapes, attention, logits and activation summaries

Do not send both through one uncontrolled firehose.

The earlier project document proposed Jobs writing a growing JSONL file to Object Storage and FastAPI polling it. That made sense for non-addressable Jobs, but it is no longer the preferred live transport now that the trainer is an interactive HTTPS endpoint.

Object Storage may still hold checkpoints and durable artifacts. It should not be the primary live metrics bus for the endpoint design.

Required architecture
React frontend
    ↓
Controller FastAPI
    ↓ authenticated server-to-server request
Nebius trainer endpoint

React should never call trainer endpoints directly.

The controller owns:

local run IDs;
experiment history;
user/session authorisation;
endpoint leases;
mapping from local run ID to remote run ID;
persistent metrics;
public streaming connection to the browser.

The trainer endpoint owns:

live PyTorch model;
optimizer state;
active training process;
pause/resume state;
diagnostic hooks;
temporary runtime snapshots.
Three categories of data
1. Architecture metadata

This is mostly static and should not be streamed repeatedly.

Examples:

number of layers
embedding dimension
head count
attention type
positional encoding
MLP/MoE structure
parameter counts
formula identifiers
expected static shape rules

Return it once when the experiment or run is opened:

GET /api/training/{local_run_id}/architecture

The controller may construct it from the stored config, or request an authoritative manifest from the trainer.

2. Training metrics and lifecycle events

These are small and continuous:

step
total_steps
train_loss
validation_loss
learning_rate
tokens_per_second
elapsed_seconds
status
checkpoint event
pause/resume event

Send these through an event stream.

3. Diagnostic snapshots

These may be much larger and should be generated only following an explicit paused-model request.

Examples:

token IDs
actual tensor shapes
top-k logits
one selected attention matrix
activation summaries
selected Q/K/V information

Use request/response or a short-lived diagnostic stream. Do not include them in the normal loss stream.

SSE versus WebSocket

For one-way training metrics, Server-Sent Events are a good fit:

GET /api/training/{run_id}/events

Advantages:

built-in browser support;
automatic reconnect;
simple text event protocol;
appropriate for server-to-browser streaming.

Pause, resume and prompt commands can remain ordinary HTTP POST requests.

However, if the repository already has a working WebSocket implementation, do not rewrite it merely for architectural elegance. Extend the existing mechanism while preserving the same event envelope.

The key requirement is a stable event protocol, not a particular transport.

Standard event envelope

Every event should have a common structure:

{
  "schema_version": 1,
  "event_id": 184,
  "timestamp": "2026-07-13T10:24:13.120Z",
  "local_run_id": 42,
  "remote_run_id": "r-8",
  "type": "training.metric",
  "payload": {
    "step": 240,
    "total_steps": 2000,
    "train_loss": 2.184,
    "val_loss": 2.302,
    "learning_rate": 0.0003
  }
}

Suggested event types:

worker.starting
worker.ready
run.queued
run.started
training.metric
run.pause_requested
run.paused
run.resume_requested
run.resumed
checkpoint.started
checkpoint.saved
diagnostic.started
diagnostic.ready
generation.token
run.completed
run.failed
run.cancelled
worker.idle
worker.stopping

Use monotonically increasing event IDs per local run.

The browser should be able to reconnect and request events after the last event it received.

Controller responsibilities

The controller must translate remote trainer events into local application events.

Flow:

1. Create local training_runs row.
2. Allocate trainer endpoint.
3. Call remote trainer /runs/start.
4. Store remote_run_id and endpoint_id.
5. Subscribe to or poll the remote event stream.
6. Persist important metrics and state changes locally.
7. Relay events to React using local_run_id.

The browser should never need the remote endpoint URL or remote run ID.

When the trainer endpoint is replaced, local run history must remain intact.

Trainer event production

Inside the training loop, emit lightweight metrics every configurable number of steps.

For a tiny model, every 5–20 steps is reasonable. Do not emit after every batch by default.

Example internal call:

event_sink.emit(
    "training.metric",
    {
        "step": step,
        "total_steps": total_steps,
        "train_loss": float(loss),
        "learning_rate": optimizer.param_groups[0]["lr"],
        "tokens_per_second": tokens_per_second,
    },
)

The event sink should be simple and non-blocking. If the client briefly disconnects, training must continue.

Use a bounded in-memory event buffer plus persisted controller metrics. Do not let a slow browser block the training loop.

Pausing and diagnostics

The diagnostic system must run against a stable model state.

Sequence:

User presses Pause
    ↓
controller sends pause request
    ↓
trainer completes the current safe step
    ↓
trainer checkpoints optimizer/model state
    ↓
trainer reports PAUSED
    ↓
diagnostic controls become available

Do not run full diagnostics concurrently with an optimizer update.

For diagnostic inference:

model.eval()
with torch.inference_mode():
    ...

After the diagnostic operation, preserve the training model and optimizer state. On resume, restore training mode:

model.train()

Be explicit in the UI that dropout is disabled during eval() diagnostic inference unless a future “sample with training dropout” option is deliberately added.

Diagnostic API

Recommended controller-facing API:

POST /api/training/{local_run_id}/diagnostics/start
POST /api/training/{local_run_id}/diagnostics/{session_id}/step
POST /api/training/{local_run_id}/diagnostics/{session_id}/generate
GET  /api/training/{local_run_id}/diagnostics/{session_id}

Start request:

{
  "prompt": "The king said",
  "top_k": 5,
  "max_prompt_tokens": 32,
  "capture": {
    "shapes": true,
    "attention": {
      "enabled": true,
      "layers": [0],
      "heads": [0]
    },
    "activations": {
      "enabled": true,
      "nodes": ["embedding", "block.0.attn", "lm_head"],
      "summary_only": true
    },
    "qkv": {
      "enabled": false
    }
  }
}

The trainer should reject this unless the run is paused or complete.

Diagnostic snapshot schema

Example:

{
  "diagnostic_session_id": "diag-17",
  "generation_step": 0,
  "prompt": "The king said",
  "tokens": [
    {"position": 0, "id": 51, "text": "The"},
    {"position": 1, "id": 82, "text": " king"}
  ],
  "nodes": {
    "embedding": {
      "input_shape": [1, 3],
      "output_shape": [1, 3, 192],
      "summary": {
        "mean": 0.004,
        "std": 0.131,
        "l2_norm": 3.27,
        "min": -0.51,
        "max": 0.62
      }
    }
  },
  "lm_head": {
    "logits_shape": [1, 3, 1024],
    "selected_position": 2,
    "top_k": [
      {"rank": 1, "token_id": 91, "token": " to", "logit": 6.21, "probability": 0.31}
    ]
  }
}
Capturing tensor shapes

Tensor shapes are inexpensive and should be the first diagnostic feature.

Use forward hooks or explicit instrumentation at known model boundaries:

embedding
each transformer block input/output
attention input/output
MLP input/output
final layer norm
LM head

Do not attach anonymous hooks to every PyTorch module and return hundreds of internal implementation nodes. Define a curated educational set of capture points.

The node_id values must match the architecture manifest consumed by React.

Attention weights: feasibility warning

Attention visualisation is not always available automatically.

PyTorch fused scaled-dot-product attention and Flash Attention often optimise away or do not return the full attention probability matrix. Therefore, diagnostic mode may need a special attention path:

QKᵀ
scale
causal mask
softmax
multiply by V

For the tiny model and a short prompt, manually calculating this in diagnostic mode is acceptable.

Do not disable fast attention during normal training merely to support visualisation. Use the slower explicit path only when a paused diagnostic request asks for attention weights.

Attention payload grows as:

layers × heads × sequence²

Therefore:

cap diagnostic prompt length;
return selected layers and heads;
avoid capturing all matrices by default;
consider float rounding or compact encoding;
reject excessively large requests.
Q, K and V

For beginners, shapes are more valuable than full Q/K/V tensors.

Default response:

Q shape
K shape
V shape
head dimension
summary statistics

Optional advanced response may include values for:

one layer;
one head;
one token position;
a short vector slice.

Do not send every Q/K/V value for every token and head.

Activations and vectors

Never stream complete hidden-state tensors continuously.

For selected nodes, provide:

shape
mean
std
minimum
maximum
L2 norm
top absolute components
small value slice

If the frontend later needs full data for a heatmap, capture one selected token or one selected layer only.

Payload caps should be enforced server-side, not trusted to frontend behaviour.

Top-k LM-head output

This is straightforward and should be implemented early.

For the final sequence position:

last_logits = logits[0, -1]
values, indices = torch.topk(last_logits, k)
probabilities = torch.softmax(last_logits, dim=-1)[indices]

Return token strings, token IDs, logits and probabilities.

Be clear that the highest probability token is not necessarily selected when sampling temperature/top-k/top-p is enabled.

Step and generate behaviour

A diagnostic session maintains:

original prompt tokens
generated token IDs
generation parameters
last diagnostic snapshot

step runs one complete forward pass and generates one new token.

generate continues from the existing state up to max_new_tokens. It may stream:

generation.token

events one token at a time, while only capturing full diagnostic details at configurable points—normally the first step, selected steps, or the final step.

Do not generate and transmit full attention/activation diagnostics for every token in a 100-token generation by default.

Persistence

Persist:

prompt
generated output
generation parameters
top-k output
selected diagnostic summaries
timestamp
associated local run ID

Do not necessarily persist full attention matrices or activation tensors. They can become large and are reproducible while the endpoint and checkpoint remain available.

Object Storage is appropriate for:

checkpoint files
export bundles
large optional diagnostic artifacts

The controller database is appropriate for:

run status
loss history
prompt/output history
small diagnostic summaries
Authentication and isolation

Trainer endpoints must require authentication.

Only the controller should hold the trainer endpoint credential.

Every diagnostic request must verify:

the local user owns the run;
the endpoint is leased to that session;
the mapped remote run is correct;
the run is paused or completed;
the capture request stays within size limits.

Do not expose an arbitrary “module name” or filesystem path supplied by the browser.

Failure and recovery

The frontend must handle:

trainer endpoint starting
endpoint temporarily unavailable
stream disconnected
run completed while disconnected
diagnostic capture unsupported
attention capture too large
checkpoint missing

On reconnect, the controller should return the latest local run state and metrics before continuing the live stream.

A missing attention matrix should not fail the entire diagnostic request. Return:

{
  "attention": {
    "available": false,
    "reason": "Attention implementation does not expose weights in this mode"
  }
}
Implementation order
Architecture manifest and static tensor-shape rules.
Existing training metric stream normalised through the controller.
Pause-safe diagnostic request with prompt tokenisation.
Captured runtime shapes.
LM-head top-k.
One selected attention layer/head.
Activation summaries.
One-step generation.
Continue-generation with token streaming.
Advanced Q/K/V and optional persistence.

This order gives a useful educational feature quickly without committing the project to streaming enormous tensors or rewriting the training loop.

### Addendum: model diagram and additional trainer metrics

Use a **single expanded decoder block** as the main architecture visual, inspired by the attached Bycroft-style diagram:

```text
Tokens → Embedding + Position
       → Transformer Block × N
       → Final LayerNorm → LM Head → Softmax/Top-k
```

Show one representative transformer block containing:

```text
LayerNorm → Causal Self-Attention → Residual
LayerNorm → MLP or MoE           → Residual
```

Do not draw all blocks. Label the block `× 4`, `× 8`, etc. Add a selector such as **Block 1 of 4** so paused diagnostics can display data captured from a chosen block. Highlight the currently selected component and open its Shapes/Math/Config/Runtime details in the Inspector pane.

Split metrics into two classes.

**Static or config-derived—send once when the run loads:**

* total parameter count;
* trainable parameter count;
* nominal active parameter count;
* number of layers, heads and experts;
* embedding/head dimensions;
* model memory estimate.

These should not be streamed every few seconds because they normally do not change.

**Runtime—stream periodically with loss metrics:**

* tokens processed;
* tokens per second, calculated over a rolling interval;
* elapsed time;
* learning rate;
* GPU/CPU device;
* for MoE: tokens routed to each expert, expert utilisation percentages, dropped-token count and load-balancing loss.

Important correction: MoE **nominal active parameter count** is usually determined by model structure and routing top-k, so it is mostly static per token. What varies dynamically is **which experts are active and how many tokens each receives**. Report those usage counts rather than pretending the parameter count constantly changes.

Capture attention matrices, Q/K/V summaries, hidden-state shapes and LM-head top-k only on explicit paused diagnostic requests, not in the normal training stream. This extends the project’s proposed metrics beyond loss without creating an overwhelming data firehose. 
