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
import ExperimentNotes from "./components/ExperimentNotes";
import {
  startTraining,
  pauseTraining,
  resumeTraining,
  stopTraining,
  fetchRunStatus,
  fetchMetrics,
  updateConfig,
} from "./hooks/useApi";

const SESSION_KEY = "llm_lab_session";

function saveSession(experimentId: number | null, runId: number | null, config: ExperimentConfig | null) {
  if (experimentId != null && config != null) {
    sessionStorage.setItem(SESSION_KEY, JSON.stringify({ experimentId, runId, config }));
  } else {
    sessionStorage.removeItem(SESSION_KEY);
  }
}

function loadSession(): { experimentId: number; runId: number | null; config: ExperimentConfig } | null {
  try {
    const raw = sessionStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    const s = JSON.parse(raw);
    if (s.experimentId != null && s.config != null) return s;
  } catch { /* ignore corrupt data */ }
  return null;
}

export default function App() {
  const saved = useRef(loadSession());
  const [experimentId, setExperimentId] = useState<number | null>(saved.current?.experimentId ?? null);
  const [config, setConfig] = useState<ExperimentConfig | null>(saved.current?.config ?? null);
  const [runId, setRunId] = useState<number | null>(saved.current?.runId ?? null);
  const [runStatus, setRunStatus] = useState<RunStatus | null>(null);
  const [metrics, setMetrics] = useState<MetricRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [device, setDevice] = useState("cpu");
  const [disconnected, setDisconnected] = useState(false);
  const [lastPollSuccess, setLastPollSuccess] = useState<number | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const [startError, setStartError] = useState<string | null>(null);
  const [controlError, setControlError] = useState<string | null>(null);
  const failCountRef = useRef(0);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const configTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  function handleConfigChange(cfg: ExperimentConfig) {
    setConfig(cfg);
    if (configTimerRef.current) clearTimeout(configTimerRef.current);
    if (experimentId != null) {
      configTimerRef.current = setTimeout(() => {
        updateConfig(experimentId, cfg);
      }, 500);
    }
  }

  function handlePresetSelect(expId: number, cfg: ExperimentConfig) {
    setExperimentId(expId);
    setConfig(cfg);
    setRunId(null);
    setRunStatus(null);
    setMetrics([]);
    saveSession(expId, null, cfg);
  }

  const pollStatus = useCallback(async () => {
    if (runId == null) return;
    try {
      const status = await fetchRunStatus(runId);
      setRunStatus(status);
      const m = await fetchMetrics(runId);
      setMetrics(m);
      failCountRef.current = 0;
      setDisconnected(false);
      setLastPollSuccess(Date.now());
      setPollError(null);
      if (status.status === "completed" || status.status === "failed" || status.status === "cancelled") {
        if (pollRef.current) clearInterval(pollRef.current);
      }
    } catch (err) {
      const isNetworkError = err instanceof TypeError || (err instanceof Error && !err.message.match(/^4\d\d/));
      if (isNetworkError) {
        failCountRef.current += 1;
        if (failCountRef.current >= 3) setDisconnected(true);
      }
      setPollError(err instanceof Error ? err.message : "Poll failed");
    }
  }, [runId]);

  useEffect(() => {
    if (runId == null) return;
    pollStatus();
    pollRef.current = setInterval(pollStatus, 2000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [runId, pollStatus]);

  async function handleStart() {
    if (experimentId == null || !config) return;
    setLoading(true);
    setStartError(null);
    // Flush any pending config debounce before starting
    if (configTimerRef.current) {
      clearTimeout(configTimerRef.current);
      configTimerRef.current = null;
      await updateConfig(experimentId, config);
    }
    try {
      const { run_id } = await startTraining(experimentId, device);
      setRunId(run_id);
      saveSession(experimentId, run_id, config);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.startsWith("429")) {
        setStartError("Max concurrent runs reached. Stop a run first.");
      } else {
        setStartError(msg);
      }
    }
    setLoading(false);
  }

  async function handlePause() {
    if (runId == null) return;
    setLoading(true);
    setControlError(null);
    try {
      await pauseTraining(runId);
    } catch (err) {
      setControlError(err instanceof Error ? err.message : "Pause failed");
    }
    setLoading(false);
  }

  async function handleResume() {
    if (runId == null) return;
    setLoading(true);
    setControlError(null);
    try {
      await resumeTraining(runId);
    } catch (err) {
      setControlError(err instanceof Error ? err.message : "Resume failed");
    }
    setLoading(false);
  }

  async function handleStop() {
    if (runId == null) return;
    setLoading(true);
    setControlError(null);
    try {
      await stopTraining(runId);
    } catch (err) {
      setControlError(err instanceof Error ? err.message : "Stop failed");
    }
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
      {disconnected && (
        <div style={{
          background: "var(--red, #e53e3e)",
          color: "#fff",
          padding: "8px 16px",
          borderRadius: 6,
          marginBottom: 12,
          fontSize: 13,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}>
          <span>⚠ Backend disconnected — restart the server and refresh</span>
          <button
            style={{ background: "rgba(255,255,255,0.2)", border: "none", color: "#fff", padding: "4px 10px", borderRadius: 4, cursor: "pointer" }}
            onClick={() => { setDisconnected(false); failCountRef.current = 0; }}
          >
            Dismiss
          </button>
        </div>
      )}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h1 style={{ fontSize: 20 }}>
          LLM Experiments Lab
          <span style={{ color: "var(--text-dim)", fontSize: 14, marginLeft: 12 }}>
            Experiment #{experimentId}
          </span>
        </h1>
        <button onClick={() => { setExperimentId(null); setConfig(null); setRunId(null); setRunStatus(null); setMetrics([]); saveSession(null, null, null); }}>
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
            onChange={handleConfigChange}
            disabled={runStatus?.status === "running"}
          />
          <TrainingControls
            runId={runId}
            runStatus={runStatus}
            onStart={handleStart}
            onPause={handlePause}
            onResume={handleResume}
            onStop={handleStop}
            loading={loading}
            device={device}
            onDeviceChange={setDevice}
            lastPollSuccess={lastPollSuccess}
            pollError={pollError}
            startError={startError}
            controlError={controlError}
          />
          <ExportBar experimentId={experimentId} />
          <ExperimentNotes experimentId={experimentId} />
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
