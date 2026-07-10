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
}

export interface MetricRow {
  step: number;
  epoch?: number;
  train_loss: number;
  val_loss: number;
  /** MoE-only: percentage of tokens dropped due to expert capacity overflow */
  train_drop_rate?: number;
  val_drop_rate?: number;
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
  created_at: string;
}
