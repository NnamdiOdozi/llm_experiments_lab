import { useState, useEffect, useRef, useCallback } from "react";
import { ExperimentConfig, MetricRow, RunStatus } from "./types";
import PresetPicker from "./components/PresetPicker";
import ConfigPanel from "./components/ConfigPanel";
import ArchSchematic from "./components/ArchSchematic";
import CodeView from "./components/CodeView";
import LossChart from "./components/LossChart";
import TrainingControls from "./components/TrainingControls";
import PausePrompt from "./components/PausePrompt";
import ExportBar from "./components/ExportBar";
import {
  startTraining,
  pauseTraining,
  resumeTraining,
  stopTraining,
  fetchRunStatus,
  fetchMetrics,
} from "./hooks/useApi";

export default function App() {
  const [experimentId, setExperimentId] = useState<number | null>(null);
  const [config, setConfig] = useState<ExperimentConfig | null>(null);
  const [runId, setRunId] = useState<number | null>(null);
  const [runStatus, setRunStatus] = useState<RunStatus | null>(null);
  const [metrics, setMetrics] = useState<MetricRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [device, setDevice] = useState("cpu");
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  function handlePresetSelect(expId: number, cfg: ExperimentConfig) {
    setExperimentId(expId);
    setConfig(cfg);
    setRunId(null);
    setRunStatus(null);
    setMetrics([]);
  }

  const pollStatus = useCallback(async () => {
    if (runId == null) return;
    try {
      const status = await fetchRunStatus(runId);
      setRunStatus(status);
      const m = await fetchMetrics(runId);
      setMetrics(m);
      if (status.status === "completed" || status.status === "failed") {
        if (pollRef.current) clearInterval(pollRef.current);
      }
    } catch { /* ignore poll errors */ }
  }, [runId]);

  useEffect(() => {
    if (runId == null) return;
    pollStatus();
    pollRef.current = setInterval(pollStatus, 2000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [runId, pollStatus]);

  async function handleStart() {
    if (experimentId == null) return;
    setLoading(true);
    const { run_id } = await startTraining(experimentId, device);
    setRunId(run_id);
    setLoading(false);
  }

  async function handlePause() {
    if (runId == null) return;
    setLoading(true);
    await pauseTraining(runId);
    setLoading(false);
  }

  async function handleResume() {
    if (runId == null) return;
    setLoading(true);
    await resumeTraining(runId);
    setLoading(false);
  }

  async function handleStop() {
    if (runId == null) return;
    setLoading(true);
    await stopTraining(runId);
    setLoading(false);
  }

  // No experiment selected: show preset picker
  if (!experimentId || !config) {
    return (
      <div style={{ maxWidth: 700, margin: "60px auto", padding: "0 20px" }}>
        <h1 style={{ fontSize: 24, marginBottom: 8 }}>LLM Experiments Lab</h1>
        <p style={{ color: "var(--text-dim)", marginBottom: 24, fontSize: 14 }}>
          Pick a preset to create an experiment. Tweak the config, train, and watch loss curves.
        </p>
        <PresetPicker onSelect={handlePresetSelect} />
      </div>
    );
  }

  // Experiment selected: show lab workspace
  return (
    <div style={{ padding: 20 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h1 style={{ fontSize: 20 }}>
          LLM Experiments Lab
          <span style={{ color: "var(--text-dim)", fontSize: 14, marginLeft: 12 }}>
            Experiment #{experimentId}
          </span>
        </h1>
        <button onClick={() => { setExperimentId(null); setConfig(null); setRunId(null); setRunStatus(null); setMetrics([]); }}>
          ← New Experiment
        </button>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "300px 1fr",
          gap: 16,
          alignItems: "start",
        }}
      >
        {/* Left sidebar */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <ConfigPanel
            config={config}
            onChange={setConfig}
            disabled={runStatus?.status === "running"}
          />
          <TrainingControls
            runStatus={runStatus}
            onStart={handleStart}
            onPause={handlePause}
            onResume={handleResume}
            onStop={handleStop}
            loading={loading}
            device={device}
            onDeviceChange={setDevice}
          />
          <ExportBar experimentId={experimentId} />
        </div>

        {/* Main area */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <LossChart metrics={metrics} />
          <ArchSchematic config={config} />
          {runId != null && (
            <PausePrompt runId={runId} paused={runStatus?.status === "paused"} />
          )}
          <CodeView experimentId={experimentId} />
        </div>
      </div>
    </div>
  );
}
