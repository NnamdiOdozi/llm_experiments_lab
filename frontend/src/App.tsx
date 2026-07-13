import { useState, useEffect, useRef, useCallback } from "react";
import { ExperimentConfig, MetricRow, RunStatus, ArchitectureNode, DiagnosticSnapshot } from "./types";
import PresetPicker from "./components/PresetPicker";
import ExperimentBrowser from "./components/ExperimentBrowser";
import HardwareSpecs from "./components/HardwareSpecs";
import ConfigPanel from "./components/ConfigPanel";
import ArchSchematic from "./components/ArchSchematic";
import Inspector from "./components/Inspector";
import CodeView from "./components/CodeView";
import LossChart from "./components/LossChart";
import DropRateChart from "./components/DropRateChart";
import TrainingControls from "./components/TrainingControls";
import PausePrompt from "./components/PausePrompt";
import ExportBar from "./components/ExportBar";
import ExperimentNotes from "./components/ExperimentNotes";
import ChatPanel from "./components/ChatPanel";
import WorkerIdleBanner from "./components/WorkerIdleBanner";
import OpenRunsPage from "./components/OpenRunsPage";
import { useActivityHeartbeat } from "./hooks/useActivityHeartbeat";
import {
  startTraining,
  pauseTraining,
  resumeTraining,
  stopTraining,
  fetchRunStatus,
  fetchMetrics,
  updateConfig,
  fetchExperiment,
  fetchPresets,
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

type RightPaneTab = "assistant" | "inspector" | "events";

export default function App() {
  const saved = useRef(loadSession());
  const [experimentId, setExperimentId] = useState<number | null>(saved.current?.experimentId ?? null);
  const [config, setConfig] = useState<ExperimentConfig | null>(saved.current?.config ?? null);
  const [runId, setRunId] = useState<number | null>(saved.current?.runId ?? null);
  const [runStatus, setRunStatus] = useState<RunStatus | null>(null);
  const [metrics, setMetrics] = useState<MetricRow[]>([]);
  const [baselineConfig, setBaselineConfig] = useState<ExperimentConfig | null>(null);
  const [loading, setLoading] = useState(false);
  const [device, setDevice] = useState("cpu");
  const [backend, setBackend] = useState("local");
  const [showOpenRuns, setShowOpenRuns] = useState(false);
  const [disconnected, setDisconnected] = useState(false);
  const [lastPollSuccess, setLastPollSuccess] = useState<number | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const [startError, setStartError] = useState<string | null>(null);
  const [controlError, setControlError] = useState<string | null>(null);

  // Inspector/diagnostic state
  const [rightPaneTab, setRightPaneTab] = useState<RightPaneTab>("assistant");
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<ArchitectureNode | null>(null);
  const [diagnosticSnapshot, setDiagnosticSnapshot] = useState<DiagnosticSnapshot | null>(null);
  const [diagnosticLoading, setDiagnosticLoading] = useState(false);

  const failCountRef = useRef(0);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const configTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Same condition as WorkerIdleBanner's visibility below — a remote
  // worker's idle clock is only relevant while actually using one.
  useActivityHeartbeat(device, runStatus?.execution_backend !== "local");

  function handleConfigChange(cfg: ExperimentConfig) {
    setConfig(cfg);
    if (configTimerRef.current) clearTimeout(configTimerRef.current);
    if (experimentId != null) {
      configTimerRef.current = setTimeout(() => {
        updateConfig(experimentId, cfg);
      }, 500);
    }
  }

  function handlePresetSelect(expId: number, cfg: ExperimentConfig, selectedDevice: string, selectedBackend: string) {
    setExperimentId(expId);
    setConfig(cfg);
    setRunId(null);
    setRunStatus(null);
    setMetrics([]);
    setDevice(selectedDevice);
    setBackend(selectedBackend);
    saveSession(expId, null, cfg);
  }

  // Reopening a past experiment (which may already have runs) to add a new
  // one — mirrors handlePresetSelect but skips creating a new experiment.
  // Device/backend reset to the same defaults a fresh session starts with;
  // TrainingControls lets the user change them before clicking Start.
  function handleLoadExperiment(expId: number, cfg: ExperimentConfig) {
    setExperimentId(expId);
    setConfig(cfg);
    setRunId(null);
    setRunStatus(null);
    setMetrics([]);
    setDevice("cpu");
    setBackend("local");
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

  // Diff-from-baseline: every experiment is created from a preset
  // (PresetPicker is the only creation path), so preset_key is always set.
  // Look it up to show the original values as shadow text in ConfigPanel.
  useEffect(() => {
    if (experimentId == null) {
      setBaselineConfig(null);
      return;
    }
    let cancelled = false;
    (async () => {
      const [exp, presets] = await Promise.all([fetchExperiment(experimentId), fetchPresets()]);
      if (cancelled) return;
      const preset = presets.find((p) => p.key === exp.preset_key);
      setBaselineConfig(
        preset ? { template: preset.template, model: preset.model, training: preset.training, inference: preset.inference } : null,
      );
    })();
    return () => { cancelled = true; };
  }, [experimentId]);

  async function handleStart() {
    if (experimentId == null || !config) return;
    setLoading(true);
    setStartError(null);
    try {
      // Flush any pending config debounce before starting
      if (configTimerRef.current) {
        clearTimeout(configTimerRef.current);
        configTimerRef.current = null;
        await updateConfig(experimentId, config);
      }
      const { run_id } = await startTraining(experimentId, device, backend);
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

  if (showOpenRuns) {
    return <OpenRunsPage onClose={() => setShowOpenRuns(false)} />;
  }

  // No experiment selected: show preset picker
  if (!experimentId || !config) {
    return (
      <div style={{ maxWidth: 700, margin: "60px auto", padding: "0 20px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
          <h1 style={{ fontSize: 24 }}>LLM Experiments Lab</h1>
          <button onClick={() => setShowOpenRuns(true)}>Open Runs</button>
        </div>
        <p style={{ color: "var(--text-dim)", marginBottom: 8, fontSize: 14 }}>
          Pick a preset to create an experiment. Tweak the config, train, and watch loss curves.
        </p>
        <div style={{ marginBottom: 24 }}>
          <HardwareSpecs />
        </div>
        <PresetPicker onSelect={handlePresetSelect} />
        <ExperimentBrowser onSelect={handleLoadExperiment} />
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
      {/* A remote run sits at "queued" for as long as the endpoint takes to
          come up (cold GPU restarts can take several minutes) — this is
          expected, not a failure, but looks identical to a real outage if
          left unexplained. See docs/DESIGN_DECISIONS.md. */}
      {runStatus?.status === "queued" && runStatus?.execution_backend !== "local" && (
        <div style={{
          background: "var(--accent-dim)",
          color: "#fff",
          padding: "8px 16px",
          borderRadius: 6,
          marginBottom: 12,
          fontSize: 13,
        }}>
          Waiting for the serverless {device === "cuda" ? "GPU" : "CPU"} endpoint to start —
          {device === "cuda"
            ? " a cold GPU restart can take up to ~5 minutes."
            : " a cold CPU restart is usually faster, up to ~2 minutes."}{" "}
          Training starts automatically once it's ready, no action needed.
        </div>
      )}
      {/* Hide when the current run is definitively local — a remote worker's
          idle status is irrelevant noise if you're not using it right now.
          See docs/DESIGN_DECISIONS.md §10. */}
      {runStatus?.execution_backend !== "local" && <WorkerIdleBanner device={device} />}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <h1 style={{ fontSize: 20 }}>
          LLM Experiments Lab
          <span style={{ color: "var(--text-dim)", fontSize: 14, marginLeft: 12 }}>
            Experiment #{experimentId}
          </span>
        </h1>
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={() => setShowOpenRuns(true)}>Open Runs</button>
          <button onClick={() => { setExperimentId(null); setConfig(null); setRunId(null); setRunStatus(null); setMetrics([]); saveSession(null, null, null); }}>
            ← New Experiment
          </button>
        </div>
      </div>
      <div style={{ marginBottom: 12 }}>
        {/* runStatus.execution_backend (the active run's real backend) takes
            priority over the device/backend picker state, which is only a
            pending choice for the *next* Start click. */}
        <HardwareSpecs device={device} backend={runStatus?.execution_backend ?? backend} />
      </div>

      <div
        style={{
          display: "grid",
          // 190px right pane ≈ 50mm at 96dpi — dedicated Lab Assistant column
          gridTemplateColumns: "360px 1fr 570px",
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
            baseline={baselineConfig}
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
            backend={backend}
            onBackendChange={setBackend}
            lastPollSuccess={lastPollSuccess}
            pollError={pollError}
            startError={startError}
            controlError={controlError}
          />
          <ExportBar experimentId={experimentId} runId={runId} />
          <ExperimentNotes experimentId={experimentId} />
        </div>

        {/* Main area */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Loss + MoE drop rate charts side-by-side when MoE data present */}
          <div style={{ display: "flex", gap: 16 }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <LossChart metrics={metrics} />
            </div>
            {metrics.some((m) => m.train_drop_rate != null) && (
              <div style={{ flex: 1, minWidth: 0 }}>
                <DropRateChart metrics={metrics} />
              </div>
            )}
          </div>
          <ArchSchematic
            runId={runId}
            onNodeClick={(nodeId, node) => {
              setSelectedNodeId(nodeId);
              setSelectedNode(node);
              setRightPaneTab("inspector");
            }}
            selectedNodeId={selectedNodeId}
          />
          {runId != null && (
            <PausePrompt
              runId={runId}
              canPrompt={runStatus?.status === "paused" || runStatus?.status === "completed"}
              onDiagnosticSnapshot={(snapshot) => {
                setDiagnosticSnapshot(snapshot);
                setDiagnosticLoading(false);
              }}
            />
          )}
          <CodeView experimentId={experimentId} runId={runId} />
        </div>

        {/* Right pane: Lab Assistant / Inspector / Events, sticky, runs top-to-bottom of the viewport */}
        <div style={{ position: "sticky", top: 20, height: "calc(100vh - 100px)" }}>
          {/* Tabs header */}
          <div
            style={{
              display: "flex",
              gap: 24,
              borderBottom: "1px solid var(--border)",
              marginBottom: 12,
              backgroundColor: "var(--surface)",
              borderRadius: "8px 8px 0 0",
              padding: "0 16px",
            }}
          >
            {(["assistant", "inspector", "events"] as RightPaneTab[]).map((tab) => (
              <button
                key={tab}
                onClick={() => setRightPaneTab(tab)}
                style={{
                  background: "none",
                  border: "none",
                  color: rightPaneTab === tab ? "var(--accent)" : "var(--text-dim)",
                  cursor: "pointer",
                  padding: "12px 0",
                  fontSize: 12,
                  fontWeight: rightPaneTab === tab ? 600 : 400,
                  borderBottom: rightPaneTab === tab ? "2px solid var(--accent)" : "none",
                  transition: "all 0.15s",
                  textTransform: "capitalize",
                }}
              >
                {tab}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div style={{ height: "calc(100% - 50px)", overflowY: "auto" }}>
            {rightPaneTab === "assistant" && <ChatPanel experimentId={experimentId} />}
            {rightPaneTab === "inspector" && (
              <Inspector
                selectedNode={selectedNode}
                selectedNodeId={selectedNodeId}
                diagnosticSnapshot={diagnosticSnapshot}
                currentStep={diagnosticSnapshot?.generation_step ?? null}
                isLoading={diagnosticLoading}
              />
            )}
            {rightPaneTab === "events" && (
              <div className="panel">
                <h3>Events</h3>
                <p style={{ fontSize: 12, color: "var(--text-dim)" }}>Event log coming soon in Phase 2.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
