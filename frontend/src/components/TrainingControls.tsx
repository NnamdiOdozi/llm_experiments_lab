import { RunStatus } from "../types";

interface Props {
  runStatus: RunStatus | null;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onStop: () => void;
  loading: boolean;
}

function statusTag(status: string) {
  const cls = `tag tag-${status}`;
  return <span className={cls}>{status}</span>;
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
          </span>
          {progressBar(runStatus.current_step, runStatus.total_steps)}
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
