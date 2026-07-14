const BASE = "/api";

// Check if fixtures mode is enabled via URL query param (?use_fixtures=true)
function useFixtures(): boolean {
  return new URLSearchParams(window.location.search).get("use_fixtures") === "true";
}

// Carries the HTTP status as a real field. Real bug, 2026-07-14: when
// api() started throwing FastAPI's detail message ("Run not found")
// instead of "404 Not Found", App.tsx's disconnect heuristic — which
// classified network-vs-HTTP by whether the MESSAGE started with a 4xx
// status code — misread every 4xx-with-detail as a network failure and
// showed a false "Backend disconnected" banner (e.g. after a backend
// restart with a stale sessionStorage runId). An instanceof/status check
// can't drift with message wording. See docs/DESIGN_DECISIONS.md.
export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    // Previously threw only "{status} {statusText}" (e.g. "400 Bad
    // Request"), discarding FastAPI's actual HTTPException detail message
    // — every rejected call in the app lost its specific reason. Found
    // while wiring the max_new_tokens/block_size validation error through
    // to the UI. See docs/DESIGN_DECISIONS.md.
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // Response body wasn't JSON — keep the status-line fallback.
    }
    throw new ApiError(detail, res.status);
  }
  return res.json();
}

export function fetchPresets() {
  return api<import("../types").Preset[]>("/experiments/presets");
}

export function createFromPreset(presetKey: string) {
  return api<{ experiment_id: number }>(`/experiments/from-preset/${presetKey}`, {
    method: "POST",
  });
}

export function fetchExperiment(id: number) {
  return api<import("../types").Experiment>(`/experiments/${id}`);
}

export function listExperiments() {
  return api<import("../types").Experiment[]>("/experiments");
}

export function startTraining(experimentId: number, device: string = "cpu", backend: string = "local") {
  return api<{ run_id: number }>("/training/start", {
    method: "POST",
    body: JSON.stringify({ experiment_id: experimentId, device, backend }),
  });
}

export function pauseTraining(runId: number) {
  return api<{ run_id: number }>(`/training/${runId}/pause`, { method: "POST" });
}

export function resumeTraining(runId: number) {
  return api<{ run_id: number }>(`/training/${runId}/resume`, { method: "POST" });
}

export function stopTraining(runId: number) {
  return api<{ run_id: number }>(`/training/${runId}/stop`, { method: "POST" });
}

export interface WorkerStatus {
  worker_status: string;
  seconds_idle: number | null;
  idle_timeout_seconds: number | null;
  warning_seconds: number | null;
  backend_mode: string;
  preset: string | null; // actual (live) preset, once an endpoint has run — null before that
  // Optional so existing mocks/tests that predate this field don't need updating.
  // Always present on a real response from GET /api/nebius/workers/{device}.
  actual_platform?: string | null;
  configured_platform?: string;
  configured_preset?: string;
}

export function getWorkerStatus(device: string) {
  return api<WorkerStatus>(`/nebius/workers/${device}`);
}

export function sendWorkerHeartbeat(device: string) {
  return api<{ ok: boolean }>(`/nebius/workers/${device}/heartbeat`, { method: "POST" });
}

export function getWorkerLogs(device: string) {
  return api<{ logs: string }>(`/nebius/workers/${device}/logs`);
}

export interface OpenRun {
  id: number;
  experiment_id: number;
  experiment_name: string;
  status: string;
  device: string;
  execution_backend: string;
  current_step: number;
  total_steps: number;
  started_at: string | null;
}

export function fetchOpenRuns() {
  return api<OpenRun[]>("/training/open");
}

export function fetchRunStatus(runId: number) {
  return api<import("../types").RunStatus>(`/training/${runId}/status`);
}

export function fetchMetrics(runId: number) {
  return api<import("../types").MetricRow[]>(`/training/${runId}/metrics`);
}

export function updateConfig(experimentId: number, config: import("../types").ExperimentConfig) {
  return api<{ ok: boolean }>(`/experiments/${experimentId}/config`, {
    method: "PATCH",
    body: JSON.stringify({ config }),
  });
}

export function updateNotes(experimentId: number, notes_md: string) {
  return api<{ ok: boolean }>(`/experiments/${experimentId}/notes`, {
    method: "PATCH",
    body: JSON.stringify({ notes_md }),
  });
}

export function setChatMessageFeedback(messageId: number, feedback: "up" | "down" | null) {
  return api<{ ok: boolean }>(`/chatbot/messages/${messageId}/feedback`, {
    method: "PATCH",
    body: JSON.stringify({ feedback }),
  });
}


export function fetchCode(experimentId: number) {
  return api<import("../types").CodeFiles>(`/code/${experimentId}`);
}

// === Diagnostic API functions ===

// Fixture data for development/testing without a real backend
const FIXTURE_MANIFEST: import("../types").ArchitectureManifest = {
  schema_version: 1,
  local_run_id: 42,
  template: "transformer",
  param_count: 1782529,
  trainable_param_count: 1782529,
  nodes: [
    {
      id: "embedding",
      kind: "embedding",
      label: "Token + Positional Embedding",
      config: { vocab_size: 65, n_embd: 192, pos_encoding: "learned" },
      static_shapes: [
        { name: "input", dims: ["batch", "sequence"] },
        { name: "output", dims: ["batch", "sequence", "n_embd"] }
      ],
      math_key: "embedding_lookup"
    },
    {
      id: "block",
      kind: "transformer_block_group",
      label: "Transformer Block",
      repeat_count: 4,
      config: { n_head: 6, head_dim: 32, dropout: 0.1, activation: "gelu" },
      children: [
        { id: "block.{i}.ln1", kind: "layernorm", label: "LayerNorm (pre-attention)", config: {} },
        { id: "block.{i}.attention", kind: "attention", label: "Causal Self-Attention", config: {},
          math_key: "scaled_dot_product_attention" },
        { id: "block.{i}.ln2", kind: "layernorm", label: "LayerNorm (pre-MLP)", config: {} },
        { id: "block.{i}.mlp", kind: "mlp", label: "Feed-Forward (dense)", config: {},
          math_key: "mlp_gelu" }
      ]
    },
    {
      id: "final_norm",
      kind: "layernorm",
      label: "Final LayerNorm",
      config: {}
    },
    {
      id: "lm_head",
      kind: "lm_head",
      label: "LM Head",
      config: { vocab_size: 65 },
      static_shapes: [
        { name: "input", dims: ["batch", "sequence", "n_embd"] },
        { name: "output", dims: ["batch", "sequence", "vocab_size"] }
      ]
    }
  ]
};

// Fixture snapshot with attention data populated (Phase 2)
const FIXTURE_SNAPSHOT_WITH_ATTENTION: import("../types").DiagnosticSnapshot = {
  schema_version: 1,
  diagnostic_session_id: "diag-17",
  generation_step: 3,
  input_tokens: [
    { position: 0, id: 51, text: "The" },
    { position: 1, id: 82, text: " king" },
    { position: 2, id: 44, text: " said" }
  ],
  generated_token: { position: 3, id: 91, text: " to" },
  nodes: {
    "embedding": {
      input_shape: [1, 3],
      output_shape: [1, 3, 192],
      summary: { mean: 0.004, std: 0.131, l2_norm: 3.27, min: -0.51, max: 0.62 }
    },
    "block.0.ln1": {
      input_shape: [1, 3, 192],
      output_shape: [1, 3, 192],
      summary: { mean: 0.0, std: 1.0, l2_norm: 24.1, min: -3.1, max: 3.4 }
    },
    "block.0.attention": {
      input_shape: [1, 3, 192],
      output_shape: [1, 3, 192],
      summary: { mean: 0.01, std: 0.44, l2_norm: 10.8, min: -1.9, max: 2.1 }
    },
    "block.0.ln2": {
      input_shape: [1, 3, 192],
      output_shape: [1, 3, 192],
      summary: { mean: 0.0, std: 1.0, l2_norm: 24.1, min: -3.0, max: 3.2 }
    },
    "block.0.mlp": {
      input_shape: [1, 3, 192],
      output_shape: [1, 3, 192],
      summary: { mean: 0.02, std: 0.51, l2_norm: 12.3, min: -2.2, max: 2.4 }
    },
    "final_norm": {
      input_shape: [1, 3, 192],
      output_shape: [1, 3, 192],
      summary: { mean: 0.0, std: 1.0, l2_norm: 24.1, min: -3.2, max: 3.3 }
    }
  },
  attention: {
    available: true,
    layer: 0,
    head: 0,
    weights: [
      [1.0, 0.0, 0.0],
      [0.62, 0.38, 0.0],
      [0.21, 0.35, 0.44]
    ],
    token_labels: ["The", " king", " said"],
    qkv_detail: {
      positions: [0, 1, 2],
      tokens: ["The", " king", " said"],
      q: [
        [0.12, -0.34, 0.08, 0.51, -0.09, 0.22, -0.44, 0.03],
        [0.14, -0.30, 0.10, 0.48, -0.11, 0.20, -0.40, 0.05],
        [0.12, -0.34, 0.08, 0.51, -0.09, 0.22, -0.44, 0.03],
      ],
      k: [
        [0.08, 0.21, -0.17, 0.24, 0.10, -0.38, 0.13, 0.19],
        [0.09, 0.19, -0.15, 0.22, 0.12, -0.35, 0.14, 0.17],
        [0.08, 0.21, -0.17, 0.24, 0.10, -0.38, 0.13, 0.19],
      ],
      v: [
        [-0.15, 0.42, 0.05, -0.29, 0.18, 0.31, -0.22, 0.13],
        [-0.13, 0.39, 0.07, -0.27, 0.16, 0.29, -0.20, 0.11],
        [-0.15, 0.42, 0.05, -0.29, 0.18, 0.31, -0.22, 0.13],
      ],
    }
  },
  activation_summaries: {
    available: true,
    top_abs_components: [
      { index: 42, value: 3.1 },
      { index: 17, value: -2.8 },
      { index: 105, value: 2.4 }
    ],
    value_slice: [0.12, -0.34, 0.08, 0.51, -0.09, 0.22, -0.44, 0.03]
  },
  lm_head: {
    logits_shape: [1, 3, 65],
    selected_position: 2,
    top_k: [
      { rank: 1, token_id: 91, token: " to", logit: 6.21, probability: 0.31 },
      { rank: 2, token_id: 12, token: " have", logit: 5.87, probability: 0.22 },
      { rank: 3, token_id: 33, token: " be", logit: 5.40, probability: 0.14 },
      { rank: 4, token_id: 7,  token: " see", logit: 5.02, probability: 0.09 },
      { rank: 5, token_id: 58, token: " go",  logit: 4.75, probability: 0.07 }
    ],
    top_k_by_position: [
      { position: 0, token: "The", actual_next_token_id: 82, top_k: [
        { rank: 1, token_id: 82, token: " king", logit: 5.10, probability: 0.28 },
        { rank: 2, token_id: 44, token: " said", logit: 4.80, probability: 0.19 },
      ] },
      { position: 1, token: " king", actual_next_token_id: 44, top_k: [
        { rank: 1, token_id: 44, token: " said", logit: 5.50, probability: 0.33 },
        { rank: 2, token_id: 91, token: " to", logit: 4.90, probability: 0.20 },
      ] },
      { position: 2, token: " said", actual_next_token_id: 91, top_k: [
        { rank: 1, token_id: 91, token: " to", logit: 6.21, probability: 0.31 },
        { rank: 2, token_id: 12, token: " have", logit: 5.87, probability: 0.22 },
        { rank: 3, token_id: 33, token: " be", logit: 5.40, probability: 0.14 },
        { rank: 4, token_id: 7,  token: " see", logit: 5.02, probability: 0.09 },
        { rank: 5, token_id: 58, token: " go",  logit: 4.75, probability: 0.07 }
      ] },
    ]
  },
  position_tokens: [],
  complete: true
};

// Fixture snapshot without attention (Phase 1 style, for backward compat)
const FIXTURE_SNAPSHOT: import("../types").DiagnosticSnapshot = {
  schema_version: 1,
  diagnostic_session_id: "diag-17",
  generation_step: 3,
  input_tokens: [
    { position: 0, id: 51, text: "The" },
    { position: 1, id: 82, text: " king" },
    { position: 2, id: 44, text: " said" }
  ],
  generated_token: { position: 3, id: 91, text: " to" },
  nodes: {
    "embedding": {
      input_shape: [1, 3],
      output_shape: [1, 3, 192],
      summary: { mean: 0.004, std: 0.131, l2_norm: 3.27, min: -0.51, max: 0.62 }
    },
    "block.0.ln1": {
      input_shape: [1, 3, 192],
      output_shape: [1, 3, 192],
      summary: { mean: 0.0, std: 1.0, l2_norm: 24.1, min: -3.1, max: 3.4 }
    },
    "block.0.attention": {
      input_shape: [1, 3, 192],
      output_shape: [1, 3, 192],
      summary: { mean: 0.01, std: 0.44, l2_norm: 10.8, min: -1.9, max: 2.1 }
    },
    "block.0.ln2": {
      input_shape: [1, 3, 192],
      output_shape: [1, 3, 192],
      summary: { mean: 0.0, std: 1.0, l2_norm: 24.1, min: -3.0, max: 3.2 }
    },
    "block.0.mlp": {
      input_shape: [1, 3, 192],
      output_shape: [1, 3, 192],
      summary: { mean: 0.02, std: 0.51, l2_norm: 12.3, min: -2.2, max: 2.4 }
    },
    "final_norm": {
      input_shape: [1, 3, 192],
      output_shape: [1, 3, 192],
      summary: { mean: 0.0, std: 1.0, l2_norm: 24.1, min: -3.2, max: 3.3 }
    }
  },
  attention: { available: false, reason: "Not requested" },
  activation_summaries: { available: false, reason: "Not requested" },
  lm_head: {
    logits_shape: [1, 3, 65],
    selected_position: 2,
    top_k: [
      { rank: 1, token_id: 91, token: " to", logit: 6.21, probability: 0.31 },
      { rank: 2, token_id: 12, token: " have", logit: 5.87, probability: 0.22 },
      { rank: 3, token_id: 33, token: " be", logit: 5.40, probability: 0.14 },
      { rank: 4, token_id: 7,  token: " see", logit: 5.02, probability: 0.09 },
      { rank: 5, token_id: 58, token: " go",  logit: 4.75, probability: 0.07 }
    ],
    top_k_by_position: [
      { position: 2, token: " said", actual_next_token_id: 91, top_k: [
        { rank: 1, token_id: 91, token: " to", logit: 6.21, probability: 0.31 },
        { rank: 2, token_id: 12, token: " have", logit: 5.87, probability: 0.22 },
      ] },
    ]
  },
  position_tokens: [],
  complete: true
};

export function fetchArchitecture(runId: number) {
  if (useFixtures()) {
    return Promise.resolve(FIXTURE_MANIFEST);
  }
  return api<import("../types").ArchitectureManifest>(`/training/${runId}/architecture`);
}

export function fetchEmbeddingTable(runId: number) {
  return api<import("../types").EmbeddingTableData>(`/training/${runId}/architecture/embedding-table`);
}

export function startDiagnostic(runId: number, payload: import("../types").DiagnosticStartRequest) {
  if (useFixtures()) {
    return Promise.resolve({
      diagnostic_session_id: "diag-17",
      tokens: [
        { position: 0, id: 51, text: "The" },
        { position: 1, id: 82, text: " king" },
        { position: 2, id: 44, text: " said" }
      ]
    });
  }
  return api<import("../types").DiagnosticSessionResponse>(`/training/${runId}/diagnostics/start`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function stepDiagnostic(runId: number, sessionId: string, params?: import("../types").DiagnosticStepRequest) {
  if (useFixtures()) {
    // Return fixture with attention data if layer/head are requested
    if (params?.attention_layer !== undefined && params.attention_head !== undefined) {
      return Promise.resolve(FIXTURE_SNAPSHOT_WITH_ATTENTION);
    }
    return Promise.resolve(FIXTURE_SNAPSHOT);
  }
  return api<import("../types").DiagnosticSnapshot>(`/training/${runId}/diagnostics/${sessionId}/step`, {
    method: "POST",
    body: JSON.stringify(params || {}),
  });
}

// Recomputes the CURRENT state's snapshot with different attention params —
// no new token sampled, generation_step/token_history untouched. Lets the
// UI refresh attention/Q-K-V immediately when Head changes in Inspector,
// instead of requiring a full step click. See docs/DESIGN_DECISIONS.md.
export function peekDiagnostic(runId: number, sessionId: string, params?: import("../types").DiagnosticStepRequest) {
  if (useFixtures()) {
    if (params?.attention_layer !== undefined && params.attention_head !== undefined) {
      return Promise.resolve(FIXTURE_SNAPSHOT_WITH_ATTENTION);
    }
    return Promise.resolve(FIXTURE_SNAPSHOT);
  }
  return api<import("../types").DiagnosticSnapshot>(`/training/${runId}/diagnostics/${sessionId}/peek`, {
    method: "POST",
    body: JSON.stringify(params || {}),
  });
}

// Persists a diagnostic session reached via manual `>` stepping to the same
// end state `>>` reaches on its own (generation_step >= maxNewTokens) —
// without this, only >> ever wrote a diagnostic_sessions row, so the Lab
// Assistant couldn't see prompts run purely by manual stepping. Direct
// user report, 2026-07-16. See docs/DESIGN_DECISIONS.md.
export function finalizeDiagnosticSession(
  runId: number, sessionId: string, params?: import("../types").DiagnosticStepRequest,
) {
  if (useFixtures()) return Promise.resolve({ success: true });
  return api<{ success: boolean }>(`/training/${runId}/diagnostics/${sessionId}/finalize`, {
    method: "POST",
    body: JSON.stringify(params || {}),
  });
}

export function getDiagnosticSession(runId: number, sessionId: string) {
  if (useFixtures()) {
    return Promise.resolve(FIXTURE_SNAPSHOT_WITH_ATTENTION);
  }
  return api<import("../types").DiagnosticSnapshot>(`/training/${runId}/diagnostics/${sessionId}`);
}

// === Phase 3: Continue-generation via SSE ===

export async function* generateDiagnosticStream(
  runId: number,
  sessionId: string,
  maxNewTokens: number = 50,
  params?: import("../types").DiagnosticStepRequest
): AsyncGenerator<import("../types").GenerateStreamToken | import("../types").GenerateStreamDone, void, unknown> {
  if (useFixtures()) {
    // Fixture: emit a few tokens then a final snapshot
    yield { position: 4, id: 91, text: " to", generation_step: 4 } as import("../types").GenerateStreamToken;
    yield { position: 5, id: 12, text: " have", generation_step: 5 } as import("../types").GenerateStreamToken;
    yield {
      final_snapshot: FIXTURE_SNAPSHOT_WITH_ATTENTION
    } as import("../types").GenerateStreamDone;
    return;
  }

  const res = await fetch(`${BASE}/training/${runId}/diagnostics/${sessionId}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // attention_layer/attention_head/qkv_detail previously never sent here —
    // >> silently ignored whatever was selected in Inspector even though the
    // backend route (DiagnosticsGenerateRequest) always accepted them. See
    // docs/DESIGN_DECISIONS.md.
    body: JSON.stringify({ max_new_tokens: maxNewTokens, ...params }),
  });

  if (!res.ok || !res.body) {
    throw new Error(`${res.status} ${res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";

      for (const frame of frames) {
        if (!frame.trim()) continue;
        let event = "";
        let data = "";
        for (const line of frame.split("\n")) {
          if (line.startsWith("event: ")) event = line.slice(7);
          if (line.startsWith("data: ")) data = line.slice(6);
        }
        if (!data) continue;

        const parsed = JSON.parse(data);
        if (event === "token") {
          yield parsed as import("../types").GenerateStreamToken;
        } else if (event === "done") {
          yield { final_snapshot: parsed.final_snapshot } as import("../types").GenerateStreamDone;
        } else if (event === "error") {
          // Previously silently dropped — a real mid->> failure looked
          // identical to success (loop just ended, nothing updated, no
          // explanation). Direct user report, 2026-07-15. See
          // docs/DESIGN_DECISIONS.md.
          throw new Error(parsed.error || "Generation failed");
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
