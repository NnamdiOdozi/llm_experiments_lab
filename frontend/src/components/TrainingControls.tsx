import { useEffect, useState } from "react";
import { RunStatus, ACTIVE_RUN_STATUSES, TERMINAL_RUN_STATUSES } from "../types";
import { getWorkerStatus } from "../hooks/useApi";
import GpuFlavorSelect from "./GpuFlavorSelect";
import "./TrainingControls.css";

interface Props {
  runId: number | null;
  runStatus: RunStatus | null;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onStop: () => void;
  loading: boolean;
  device: string;
  onDeviceChange: (d: string) => void;
  backend: string;
  onBackendChange: (b: string) => void;
  gpuFlavor: string;
  onGpuFlavorChange: (flavor: string) => void;
  lastPollSuccess: number | null;
  pollError: string | null;
  startError: string | null;
  controlError: string | null;
  soundMuted: boolean;
  onToggleSoundMuted: () => void;
}

function statusTag(status: string) {
  const cls = `tag tag-${status}`;
  return <span className={cls}>{status}</span>;
}

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function formatPreset(preset: string | null): string {
  if (!preset) return "";
  const match = preset.match(/^(\d+)vcpu-(\d+)gb$/i);
  return match ? `${match[1]}vCPU / ${match[2]}GB` : preset;
}

function progressBar(current: number, total: number) {
  const pct = total > 0 ? Math.min((current / total) * 100, 100) : 0;
  return (
    <div className="training-panel__progress-track">
      <div
        style={{
          background: "var(--accent)",
          borderRadius: 4,
          height: "100%",
          width: `${pct}%`,
          transition: "width 0.3s",
        }}
      />
    </div>
  );
}

export default function TrainingControls({
  runId,
  runStatus,
  onStart,
  onPause,
  onResume,
  onStop,
  loading,
  device,
  onDeviceChange,
  backend,
  onBackendChange,
  gpuFlavor,
  onGpuFlavorChange,
  lastPollSuccess,
  pollError,
  startError,
  controlError,
  soundMuted,
  onToggleSoundMuted,
}: Props) {
  const status = runStatus?.status;
  const isActive = status != null && ACTIVE_RUN_STATUSES.has(status);
  const isDone = status == null || TERMINAL_RUN_STATUSES.has(status);

  // The run's OWN execution_backend, not the app's current global setting —
  // a run started under one setting keeps its own value even if the global
  // setting changes later (or a different run is using different infra).
  // See docs/DESIGN_DECISIONS.md §10.
  const isRemoteRun = runStatus?.execution_backend === "nebius_endpoint";
  const [preset, setPreset] = useState<string | null>(null);
  useEffect(() => {
    if (!isRemoteRun) {
      setPreset(null);
      return;
    }
    getWorkerStatus(device).then((s) => setPreset(s.preset));
  }, [device, isRemoteRun]);
  const workerTag = isRemoteRun
    ? [device.toUpperCase(), formatPreset(preset), "Serverless"].filter(Boolean).join(" · ")
    : `${device.toUpperCase()} · Local`;

  const staleSeconds = lastPollSuccess ? Math.floor((Date.now() - lastPollSuccess) / 1000) : null;
  const isStale = staleSeconds != null && staleSeconds > 10 && isActive;

  return (
    <div className="panel training-panel">
      <h3>Training</h3>

      {startError && (
        <div style={{ background: "var(--red, #e53e3e)", color: "#fff", padding: "6px 12px", borderRadius: 4, fontSize: 15, marginBottom: 8 }}>
          {startError}
        </div>
      )}

      {controlError && (
        <div style={{ background: "var(--red, #e53e3e)", color: "#fff", padding: "6px 12px", borderRadius: 4, fontSize: 15, marginBottom: 8 }}>
          {controlError}
        </div>
      )}

      {isStale && (
        <div style={{ background: "#dd6b20", color: "#fff", padding: "6px 12px", borderRadius: 4, fontSize: 15, marginBottom: 8 }}>
          Data may be stale — last update {staleSeconds}s ago
          {pollError && <span> ({pollError})</span>}
        </div>
      )}

      {runStatus && (
        <div className="training-panel__run-status">
          {runId != null && (
            <span className="training-panel__run-id">
              Run #{runId}
            </span>
          )}
          {statusTag(runStatus.status)}
          <span className="tag training-panel__worker-tag">
            {workerTag}
          </span>
          <span className="training-panel__step">
            Step {runStatus.current_step} / {runStatus.total_steps}
            {runStatus.elapsed_seconds > 0 && ` — ${formatElapsed(runStatus.elapsed_seconds)}`}
          </span>
          {progressBar(runStatus.current_step, runStatus.total_steps)}
        </div>
      )}

      {isDone && (
        <div className="training-panel__selectors">
          <div>
            <label style={{ fontSize: 15, color: "var(--text-dim)", marginRight: 8 }}>Device</label>
            <select
              value={device}
              onChange={(e) => onDeviceChange(e.target.value)}
              style={{ fontSize: 15, padding: "4px 8px" }}
            >
              <option value="cpu">CPU</option>
              <option value="cuda">GPU (CUDA)</option>
            </select>
          </div>
          <div>
            <label style={{ fontSize: 15, color: "var(--text-dim)", marginRight: 8 }}>Backend</label>
            <select
              value={backend}
              onChange={(e) => onBackendChange(e.target.value)}
              style={{ fontSize: 15, padding: "4px 8px" }}
            >
              <option value="local">Local</option>
              <option value="nebius_endpoint">Serverless (Nebius)</option>
            </select>
          </div>
          {device.startsWith("cuda") && backend === "nebius_endpoint" && (
            <GpuFlavorSelect gpuFlavor={gpuFlavor} onFlavorChange={onGpuFlavorChange} />
          )}
        </div>
      )}

      <div className="training-panel__actions">
        {isDone && (
          <button className="btn-primary" onClick={() => { if (!loading) onStart(); }} disabled={loading}>
            {loading ? "Starting…" : "Start Training"}
          </button>
        )}
        {isActive && (
          <button onClick={onPause} disabled={loading || status !== "running"}>
            {status === "pause_requested" || status === "checkpointing" ? "Pausing…" : "Pause"}
          </button>
        )}
        {status === "paused" && (
          <>
            <button className="btn-primary" onClick={onResume} disabled={loading}>
              Resume
            </button>
            <button onClick={onStop} disabled={loading}>
              Stop
            </button>
          </>
        )}
        {isActive && (
          <button style={{ borderColor: "var(--red)" }} onClick={onStop} disabled={loading}>
            Stop
          </button>
        )}
        <button
          onClick={onToggleSoundMuted}
          title={soundMuted ? "Unmute training-start sound" : "Mute training-start sound"}
          style={{ marginLeft: "auto", fontSize: 14 }}
        >
          {soundMuted ? "🔇" : "🔔"}
        </button>
      </div>
    </div>
  );
}
