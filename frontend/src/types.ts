export interface Preset {
  key: string;
  name: string;
  description: string;
  template: "transformer" | "rnn";
  model: Record<string, number | string>;
  training: Record<string, number | string>;
  dataset: string;
}

export interface ExperimentConfig {
  template: "transformer" | "rnn";
  model: Record<string, number | string>;
  training: Record<string, number | string>;
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
  status: string;
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

export interface RunStatus {
  run_id: number;
  status: string;
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
}

export interface CodeFiles {
  experiment_id: number;
  template: string;
  files: Record<string, string>;
}
