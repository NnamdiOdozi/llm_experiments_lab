import { useEffect, useState } from "react";
import { fetchMetrics, fetchRunStatus, getWorkerLogs, getWorkerStatus } from "../hooks/useApi";
import { MetricRow, RunStatus } from "../types";

interface Props {
  runId: number | null;
  device: string;
}

type SubTab = "events" | "logs";

const EVENTS_POLL_MS = 3000;
const LOGS_POLL_MS = 10000;

function tabButtonStyle(active: boolean) {
  return {
    fontSize: 12,
    background: active ? "var(--accent-dim)" : undefined,
    color: active ? "#fff" : undefined,
  };
}

export default function WorkerPanel({ runId, device }: Props) {
  const [subTab, setSubTab] = useState<SubTab>("events");
  const [status, setStatus] = useState<RunStatus | null>(null);
  const [metrics, setMetrics] = useState<MetricRow[]>([]);
  const [backendMode, setBackendMode] = useState<string | null>(null);
  const [rawLogs, setRawLogs] = useState("");

  useEffect(() => {
    if (runId == null) {
      setStatus(null);
      setMetrics([]);
      return;
    }
    let cancelled = false;
    async function poll() {
      try {
        const [s, m] = await Promise.all([fetchRunStatus(runId as number), fetchMetrics(runId as number)]);
        if (!cancelled) {
          setStatus(s);
          setMetrics(m);
        }
      } catch {
        // best-effort secondary view — the main poll loop already surfaces errors
      }
    }
    poll();
    const id = setInterval(poll, EVENTS_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [runId]);

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      try {
        const s = await getWorkerStatus(device);
        if (cancelled) return;
        setBackendMode(s.backend_mode);
        if (s.backend_mode === "nebius_endpoint") {
          const { logs } = await getWorkerLogs(device);
          if (!cancelled) setRawLogs(logs);
        }
      } catch {
        // best-effort
      }
    }
    poll();
    const id = setInterval(poll, LOGS_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [device]);

  const eventLines = [
    ...(status ? [`STATUS ${status.status} step=${status.current_step}/${status.total_steps}`] : []),
    ...metrics.map(
      (m) => `STEP step=${m.step} loss=${m.train_loss.toFixed(4)} val_loss=${m.val_loss.toFixed(4)}`,
    ),
  ];

  return (
    <div>
      <div style={{ display: "flex", gap: 6, marginBottom: 10 }}>
        <button style={tabButtonStyle(subTab === "events")} onClick={() => setSubTab("events")}>
          Events
        </button>
        <button style={tabButtonStyle(subTab === "logs")} onClick={() => setSubTab("logs")}>
          Raw Logs
        </button>
      </div>
      {subTab === "events" ? (
        <pre style={{ maxHeight: 300, overflowY: "auto", fontSize: 12 }}>
          {eventLines.length ? eventLines.join("\n") : "No events yet — start a run to see step-by-step progress here."}
        </pre>
      ) : backendMode !== "nebius_endpoint" ? (
        <p style={{ fontSize: 12, color: "var(--text-dim)" }}>
          N/A — running locally, check the server console for logs.
        </p>
      ) : (
        <pre style={{ maxHeight: 300, overflowY: "auto", fontSize: 12, whiteSpace: "pre-wrap" }}>
          {rawLogs || "No logs yet."}
        </pre>
      )}
    </div>
  );
}
