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

// Demo-mode fixtures (FIXTURE_MANIFEST / FIXTURE_SNAPSHOT / _WITH_ATTENTION)
// live in ../fixtures/diagnostics.ts and are loaded via DYNAMIC import only
// inside useFixtures() branches — a static import here would put ~230 lines
// of demo data back into the bundle every real user downloads.

export async function fetchArchitecture(runId: number) {
  if (useFixtures()) {
    const { FIXTURE_MANIFEST } = await import("../fixtures/diagnostics");
    return FIXTURE_MANIFEST;
  }
  return api<import("../types").ArchitectureManifest>(`/training/${runId}/architecture`);
}

export function fetchEmbeddingTable(runId: number) {
  return api<import("../types").EmbeddingTableData>(`/training/${runId}/architecture/embedding-table`);
}

export async function startDiagnostic(runId: number, payload: import("../types").DiagnosticStartRequest) {
  if (useFixtures()) {
    const { FIXTURE_START_RESPONSE } = await import("../fixtures/diagnostics");
    return FIXTURE_START_RESPONSE;
  }
  return api<import("../types").DiagnosticSessionResponse>(`/training/${runId}/diagnostics/start`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function stepDiagnostic(runId: number, sessionId: string, params?: import("../types").DiagnosticStepRequest) {
  if (useFixtures()) {
    const { FIXTURE_SNAPSHOT, FIXTURE_SNAPSHOT_WITH_ATTENTION } = await import("../fixtures/diagnostics");
    // Return fixture with attention data if layer/head are requested
    if (params?.attention_layer !== undefined && params.attention_head !== undefined) {
      return FIXTURE_SNAPSHOT_WITH_ATTENTION;
    }
    return FIXTURE_SNAPSHOT;
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
export async function peekDiagnostic(runId: number, sessionId: string, params?: import("../types").DiagnosticStepRequest) {
  if (useFixtures()) {
    const { FIXTURE_SNAPSHOT, FIXTURE_SNAPSHOT_WITH_ATTENTION } = await import("../fixtures/diagnostics");
    if (params?.attention_layer !== undefined && params.attention_head !== undefined) {
      return FIXTURE_SNAPSHOT_WITH_ATTENTION;
    }
    return FIXTURE_SNAPSHOT;
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

export async function getDiagnosticSession(runId: number, sessionId: string) {
  if (useFixtures()) {
    const { FIXTURE_SNAPSHOT_WITH_ATTENTION } = await import("../fixtures/diagnostics");
    return FIXTURE_SNAPSHOT_WITH_ATTENTION;
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
    const { FIXTURE_SNAPSHOT_WITH_ATTENTION } = await import("../fixtures/diagnostics");
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
