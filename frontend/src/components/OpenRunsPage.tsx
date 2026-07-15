import { useEffect, useState } from "react";
import { fetchOpenRuns, stopTraining, OpenRun } from "../hooks/useApi";

interface Props {
  onClose: () => void;
  // Jumps straight back into a run's workspace — previously this page could
  // only Stop a run, with no way back in at all. Direct user report,
  // 2026-07-15. See docs/DESIGN_DECISIONS.md.
  onReopen: (run: OpenRun) => void;
}

export default function OpenRunsPage({ onClose, onReopen }: Props) {
  const [runs, setRuns] = useState<OpenRun[] | null>(null);
  const [stoppingId, setStoppingId] = useState<number | null>(null);
  const [stopError, setStopError] = useState<string | null>(null);

  function load() {
    fetchOpenRuns().then(setRuns);
  }

  useEffect(() => {
    load();
  }, []);

  async function handleStop(runId: number) {
    setStoppingId(runId);
    setStopError(null);
    try {
      await stopTraining(runId);
    } catch (err) {
      // Real bug found 2026-07-14: this catch didn't exist — a failed
      // stop (e.g. a run pointing at a Nebius endpoint the user had
      // deleted outside the app) threw an unhandled promise rejection
      // visible only in the browser console. The run just silently
      // stayed in the list with no indication anything had gone wrong.
      // See docs/DESIGN_DECISIONS.md.
      setStopError(`Failed to stop run ${runId}: ` + (err instanceof Error ? err.message : String(err)));
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

      {stopError && (
        <div style={{ background: "var(--red, #dc2626)", color: "white", padding: "8px 12px", borderRadius: 6, marginBottom: 12, fontSize: 13 }}>
          {stopError}
        </div>
      )}

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
                <div style={{ fontWeight: 600 }}>{r.experiment_name} <span style={{ color: "var(--text-dim)", fontWeight: "normal" }}>#{r.experiment_id}</span></div>
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
                <button onClick={() => onReopen(r)}>Open</button>
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
