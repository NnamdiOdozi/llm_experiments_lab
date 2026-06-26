import { RunStatus } from "../types";

interface Props {
  runStatus: RunStatus | null;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onStop: () => void;
  loading: boolean;
  device: string;
  onDeviceChange: (d: string) => void;
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

function progressBar(current: number, total: number) {
  const pct = total > 0 ? Math.min((current / total) * 100, 100) : 0;
  return (
    <div style={{ background: "var(--bg)", borderRadius: 4, height: 8, marginTop: 8 }}>
      <div
        style={{
          background: "var(--accent)",
          borderRadius: 4,
          height: 8,
          width: `${pct}%`,
          transition: "width 0.3s",
        }}
      />
    </div>
  );
}

export default function TrainingControls({
  runStatus,
  onStart,
  onPause,
  onResume,
  onStop,
  loading,
  device,
  onDeviceChange,
}: Props) {
  const status = runStatus?.status;

  return (
    <div className="panel">
      <h3>Training</h3>

      {runStatus && (
        <div style={{ marginBottom: 12 }}>
          {statusTag(runStatus.status)}
          <span style={{ fontSize: 12, color: "var(--text-dim)", marginLeft: 10 }}>
            Step {runStatus.current_step} / {runStatus.total_steps}
            {runStatus.elapsed_seconds > 0 && ` — ${formatElapsed(runStatus.elapsed_seconds)}`}
          </span>
          {progressBar(runStatus.current_step, runStatus.total_steps)}
        </div>
      )}

      {(!status || status === "completed" || status === "failed") && (
        <div style={{ marginBottom: 10 }}>
          <label style={{ fontSize: 12, color: "var(--text-dim)", marginRight: 8 }}>Device</label>
          <select
            value={device}
            onChange={(e) => onDeviceChange(e.target.value)}
            style={{ fontSize: 12, padding: "4px 8px" }}
          >
            <option value="cpu">CPU</option>
            <option value="cuda">GPU (CUDA)</option>
          </select>
        </div>
      )}

      <div style={{ display: "flex", gap: 8 }}>
        {(!status || status === "completed" || status === "failed") && (
          <button className="btn-primary" onClick={onStart} disabled={loading}>
            Start Training
          </button>
        )}
        {status === "running" && (
          <button onClick={onPause} disabled={loading}>
            Pause
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
        {status === "running" && (
          <button style={{ borderColor: "var(--red)" }} onClick={onStop} disabled={loading}>
            Stop
          </button>
        )}
      </div>
    </div>
  );
}
