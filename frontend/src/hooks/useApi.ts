const BASE = "/api";

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
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

export function startTraining(experimentId: number, device: string = "cpu") {
  return api<{ run_id: number }>("/training/start", {
    method: "POST",
    body: JSON.stringify({ experiment_id: experimentId, device }),
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

export function promptModel(runId: number, prompt: string) {
  return api<{ output: string }>(`/training/${runId}/prompt`, {
    method: "POST",
    body: JSON.stringify({ prompt }),
  });
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

export function fetchCode(experimentId: number) {
  return api<import("../types").CodeFiles>(`/code/${experimentId}`);
}
