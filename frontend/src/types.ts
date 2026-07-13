export interface Preset {
  key: string;
  name: string;
  description: string;
  template: "transformer" | "moe" | "rnn";
  model: Record<string, number | string>;
  training: Record<string, number | string>;
  inference?: Record<string, number | string>;
  dataset: string;
}

export interface ExperimentConfig {
  template: "transformer" | "moe" | "rnn";
  model: Record<string, number | string>;
  training: Record<string, number | string>;
  inference?: Record<string, number | string>;
}

export interface Experiment {
  id: number;
  name: string;
  config: ExperimentConfig;
  notes_md: string;
  preset_key: string | null;
  created_at: string;
  updated_at: string;
}

export interface TrainingRun {
  id: number;
  experiment_id: number;
  status: RunStatusValue;
  device: string;
  train_loss_history: string;
  val_loss_history: string;
  final_train_loss: number | null;
  final_val_loss: number | null;
  total_steps: number;
  current_step: number;
  started_at: string | null;
  completed_at: string | null;
}

/** Must match backend/training/status.py RunStatus enum */
export type RunStatusValue =
  | "queued"
  | "starting"
  | "running"
  | "pause_requested"
  | "checkpointing"
  | "paused"
  | "resuming"
  | "completed"
  | "failed"
  | "cancelled";

export const ACTIVE_RUN_STATUSES = new Set<RunStatusValue>([
  "queued", "starting", "running", "pause_requested", "checkpointing", "resuming",
]);
export const TERMINAL_RUN_STATUSES = new Set<RunStatusValue>([
  "completed", "failed", "cancelled",
]);

export interface RunStatus {
  run_id: number;
  status: RunStatusValue;
  current_step: number;
  total_steps: number;
  metrics_count: number;
  template: string;
  elapsed_seconds: number;
  /** "local" or "nebius_endpoint" — this run's own actual backend, not the
   * app's current global setting. A run started under one setting keeps its
   * own value even if the global setting changes later. */
  execution_backend: string;
}

export interface MetricRow {
  step: number;
  epoch?: number;
  train_loss: number;
  val_loss: number;
  /** MoE-only: percentage of tokens dropped due to expert capacity overflow */
  train_drop_rate?: number;
  val_drop_rate?: number;
  /** Resource utilization, sampled by the worker alongside loss — best-effort,
   * absent if psutil/nvidia-smi sampling failed. GPU fields only for cuda runs. */
  cpu_percent?: number;
  ram_used_mb?: number;
  ram_total_mb?: number;
  gpu_utilization_pct?: number;
  gpu_memory_used_mb?: number;
  gpu_memory_total_mb?: number;
  gpu_temp_c?: number;
}

export interface CodeFiles {
  experiment_id: number;
  template: string;
  files: Record<string, string>;
}

export interface ChatMessage {
  id: number;
  experiment_id: number;
  role: "user" | "assistant";
  content: string;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  latency_ms: number | null;
  feedback?: "up" | "down" | null;
  created_at: string;
}

// === Diagnostic API types ===

export interface TokenInfo {
  position: number;
  id: number;
  text: string;
}

export interface TopKEntry {
  rank: number;
  token_id: number;
  token: string;
  logit: number;
  probability: number;
}

export interface NodeShape {
  name: string;
  dims: string[];
}

export interface ArchitectureNode {
  id: string;
  kind: string;
  label: string;
  repeat_count?: number;
  config: Record<string, unknown>;
  static_shapes?: NodeShape[];
  math_key?: string;
  children?: ArchitectureNode[];
}

export interface ArchitectureManifest {
  schema_version: number;
  local_run_id: number;
  template: string;
  param_count: number;
  trainable_param_count: number;
  nodes: ArchitectureNode[];
}

export interface EmbeddingTableData {
  vocab_size: number;
  n_embd: number;
  tokens: string[];
  // embedding[i] is the trained vector for tokens[i] — full n_embd width,
  // not windowed (transformer/moe only, small char-level vocabs). See
  // docs/DESIGN_DECISIONS.md.
  embedding: number[][];
  // Only present under pos_encoding="learned" — RoPE has no learned
  // position table (computed on the fly), so both are null in that case.
  block_size: number | null;
  position_embedding: number[][] | null;
}

export interface NodePositionVectors {
  // Last DIAGNOSTIC_POSITION_WINDOW (12) positions only — see
  // docs/DESIGN_DECISIONS.md.
  positions: number[];
  vectors: number[][];
}

export interface NodeRuntimeData {
  input_shape?: number[];
  output_shape?: number[];
  summary?: {
    mean: number;
    std: number;
    l2_norm: number;
    min: number;
    max: number;
  };
  position_vectors?: NodePositionVectors | null;
  input_position_vectors?: NodePositionVectors | null;
}

export interface TopKByPositionEntry {
  position: number;
  token: string;
  top_k: TopKEntry[];
}

export interface LMHeadData {
  logits_shape: number[];
  selected_position: number;
  top_k: TopKEntry[];
  // Last DIAGNOSTIC_POSITION_WINDOW (12) positions only — see
  // docs/DESIGN_DECISIONS.md. Feeds the LM Head position stepper.
  top_k_by_position: TopKByPositionEntry[];
}

export interface QKVDetail {
  // One entry per position (last 12 — see docs/DESIGN_DECISIONS.md), not
  // just the final token. positions[i]/tokens[i]/q[i]/k[i]/v[i] all
  // correspond to the same position.
  positions: number[];
  tokens: string[];
  q: number[][];
  k: number[][];
  v: number[][];
}

export interface AttentionData {
  available: boolean;
  layer?: number;
  head?: number;
  weights?: number[][];
  token_labels?: string[];
  reason?: string;
  qkv_detail?: QKVDetail;
  // window_start/total_positions describe the windowed weights/token_labels
  // above (a square DIAGNOSTIC_POSITION_WINDOW block, not the full T x T
  // matrix) — lets the frontend stepper know how far it can shift and what
  // range is currently shown. See docs/DESIGN_DECISIONS.md.
  window_start?: number;
  total_positions?: number;
}

export interface PositionToken {
  position: number;
  id: number;
  token: string;
}

export interface ActivationSummariesData {
  available: boolean;
  top_abs_components?: Array<{ index: number; value: number }>;
  value_slice?: number[];
  reason?: string;
}

export interface DiagnosticSnapshot {
  schema_version: number;
  diagnostic_session_id: string;
  generation_step: number;
  input_tokens: TokenInfo[];
  generated_token: TokenInfo;
  nodes: Record<string, NodeRuntimeData>;
  attention: AttentionData;
  activation_summaries: ActivationSummariesData;
  lm_head: LMHeadData;
  // Windowed token id+text per position, same window as every node's
  // position_vectors — used by the embedding node's one-hot input table.
  position_tokens: PositionToken[];
  complete: boolean;
}

export interface DiagnosticSessionResponse {
  diagnostic_session_id: string;
  tokens: TokenInfo[];
}

export interface DiagnosticStartRequest {
  prompt: string;
  top_k: number;
  max_prompt_tokens: number;
}

export interface DiagnosticStepRequest {
  attention_layer?: number;
  attention_head?: number;
  qkv_detail?: boolean;
  attention_window_offset?: number;
}

export interface GenerateStreamToken {
  position: number;
  id: number;
  text: string;
  generation_step: number;
}

export interface GenerateStreamDone {
  final_snapshot: DiagnosticSnapshot;
}
