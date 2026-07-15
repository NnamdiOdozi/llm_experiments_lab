import { useEffect, useState } from "react";
import { fetchOpenRuns, stopTraining, OpenRun } from "../hooks/useApi";

interface Props {
  onClose: () => void;
  // Jumps straight back into a run's workspace — previously this page could
  // only Stop a run, with no way back in at all. Direct user report,
  // 2026-07-15. See docs/DESIGN_DECISIONS.md.
  onReopen: (run: OpenRun) => void;
}

// Direct user request, 2026-07-15: once a run finished/failed/stopped,
// there was no way back to it at all — active-only was the entire
// contents of this page. "Active" stays the default so today's behavior
// is unchanged unless asked; the other filters are what's new. See
// docs/DESIGN_DECISIONS.md §79b.
const FILTERS = ["Active", "Completed", "Failed", "Cancelled", "All"] as const;
type Filter = (typeof FILTERS)[number];

function matchesFilter(status: string, filter: Filter): boolean {
  if (filter === "All") return true;
  if (filter === "Active") return status !== "completed" && status !== "failed" && status !== "cancelled";
  return status === filter.toLowerCase();
}

export default function OpenRunsPage({ onClose, onReopen }: Props) {
  const [runs, setRuns] = useState<OpenRun[] | null>(null);
  const [stoppingId, setStoppingId] = useState<number | null>(null);
  const [stopError, setStopError] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>("Active");

  function load() {
    // Only the "Active" filter can be served by the cheap default query —
    // everything else needs terminal runs included, then filtered further
    // client-side (this app has a handful of runs per experiment, not
    // thousands — not worth a separate query per filter).
    fetchOpenRuns(filter !== "Active").then(setRuns);
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filter]);

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

  const visibleRuns = runs?.filter((r) => matchesFilter(r.status, filter)) ?? null;

  return (
    <div style={{ maxWidth: 900, margin: "40px auto", padding: "0 20px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h1 style={{ fontSize: 20 }}>Runs</h1>
        <button onClick={onClose}>← Back</button>
      </div>

      <div style={{ display: "flex", gap: 6, marginBottom: 16 }}>
        {FILTERS.map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            style={{
              fontSize: 12,
              padding: "4px 10px",
              borderRadius: 999,
              border: f === filter ? "1px solid var(--accent)" : "1px solid var(--border)",
              background: f === filter ? "var(--accent)" : "transparent",
              color: f === filter ? "#fff" : "var(--text-dim)",
              cursor: "pointer",
            }}
          >
            {f}
          </button>
        ))}
      </div>

      {stopError && (
        <div style={{ background: "var(--red, #dc2626)", color: "white", padding: "8px 12px", borderRadius: 6, marginBottom: 12, fontSize: 13 }}>
          {stopError}
        </div>
      )}

      {visibleRuns == null ? (
        <p style={{ color: "var(--text-dim)" }}>Loading...</p>
      ) : visibleRuns.length === 0 ? (
        <p style={{ color: "var(--text-dim)" }}>
          {filter === "Active" ? "No active runs — nothing currently running or paused." : `No ${filter.toLowerCase()} runs.`}
        </p>
      ) : (
        <div className="panel">
          {visibleRuns.map((r) => {
            const isTerminal = r.status === "completed" || r.status === "failed" || r.status === "cancelled";
            return (
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
                {/* A terminal run has nothing left to stop — hidden rather
                    than shown-but-a-no-op, direct user request, 2026-07-15.
                    See docs/DESIGN_DECISIONS.md §79b. */}
                {!isTerminal && (
                  <button onClick={() => handleStop(r.id)} disabled={stoppingId === r.id}>
                    {stoppingId === r.id ? "Stopping..." : "Stop"}
                  </button>
                )}
              </div>
            </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
