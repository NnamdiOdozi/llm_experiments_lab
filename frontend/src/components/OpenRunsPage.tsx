import { useEffect, useState } from "react";
import { fetchOpenRuns, stopTraining, OpenRun } from "../hooks/useApi";

interface Props {
  onClose: () => void;
}

export default function OpenRunsPage({ onClose }: Props) {
  const [runs, setRuns] = useState<OpenRun[] | null>(null);
  const [stoppingId, setStoppingId] = useState<number | null>(null);

  function load() {
    fetchOpenRuns().then(setRuns);
  }

  useEffect(() => {
    load();
  }, []);

  async function handleStop(runId: number) {
    setStoppingId(runId);
    try {
      await stopTraining(runId);
    } finally {
      setStoppingId(null);
      load();
    }
  }

  return (
    <div style={{ maxWidth: 900, margin: "40px auto", padding: "0 20px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h1 style={{ fontSize: 20 }}>Open Runs</h1>
        <button onClick={onClose}>← Back</button>
      </div>

      {runs == null ? (
        <p style={{ color: "var(--text-dim)" }}>Loading...</p>
      ) : runs.length === 0 ? (
        <p style={{ color: "var(--text-dim)" }}>No open runs — nothing currently running or paused.</p>
      ) : (
        <div className="panel">
          {runs.map((r) => (
            <div
              key={r.id}
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "10px 0",
                borderBottom: "1px solid var(--border)",
              }}
            >
              <div>
                <div style={{ fontWeight: 600 }}>{r.experiment_name}</div>
                <div style={{ fontSize: 12, color: "var(--text-dim)" }}>
                  Run #{r.id} &middot; {r.device.toUpperCase()}
                  {r.execution_backend === "nebius_endpoint" ? " · Serverless" : " · Local"}
                  {r.started_at ? ` · started ${r.started_at}` : ""}
                </div>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span className={`tag tag-${r.status}`}>{r.status}</span>
                <span style={{ fontSize: 12, color: "var(--text-dim)" }}>
                  Step {r.current_step} / {r.total_steps}
                </span>
                <button onClick={() => handleStop(r.id)} disabled={stoppingId === r.id}>
                  {stoppingId === r.id ? "Stopping..." : "Stop"}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
