import { useEffect, useState } from "react";
import { fetchMetrics, fetchRunStatus } from "../hooks/useApi";
import { MetricRow, RunStatus } from "../types";

interface Props {
  runId: number | null;
}

const EVENTS_POLL_MS = 3000;

function formatUsage(m: MetricRow): string {
  const parts: string[] = [];
  if (m.cpu_percent != null) parts.push(`cpu=${m.cpu_percent.toFixed(0)}%`);
  if (m.ram_used_mb != null && m.ram_total_mb != null) {
    parts.push(`ram=${Math.round(m.ram_used_mb)}/${Math.round(m.ram_total_mb)}MB`);
  }
  if (m.gpu_utilization_pct != null) parts.push(`gpu=${m.gpu_utilization_pct.toFixed(0)}%`);
  if (m.gpu_memory_used_mb != null && m.gpu_memory_total_mb != null) {
    parts.push(`gpu_mem=${Math.round(m.gpu_memory_used_mb)}/${Math.round(m.gpu_memory_total_mb)}MB`);
  }
  if (m.gpu_temp_c != null) parts.push(`gpu_temp=${m.gpu_temp_c.toFixed(0)}C`);
  return parts.length ? " " + parts.join(" ") : "";
}

// Single flat view — was previously two sub-tabs (Metrics / raw container
// Logs). Collapsed 2026-07-12 per user feedback: the raw-logs sub-tab was
// consistently empty for CPU runs and added nothing the step-by-step
// events below don't already cover (which already include loss + resource
// usage inline). See docs/DESIGN_DECISIONS.md.
export default function WorkerPanel({ runId }: Props) {
  const [status, setStatus] = useState<RunStatus | null>(null);
  const [metrics, setMetrics] = useState<MetricRow[]>([]);

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

  const eventLines = [
    ...(status ? [`STATUS ${status.status} step=${status.current_step}/${status.total_steps}`] : []),
    ...metrics.map(
      (m) => `STEP step=${m.step} loss=${m.train_loss.toFixed(4)} val_loss=${m.val_loss.toFixed(4)}${formatUsage(m)}`,
    ),
  ];

  return (
    <pre style={{ maxHeight: 300, overflowY: "auto", fontSize: 24 }}>
      {eventLines.length ? eventLines.join("\n") : "No metrics yet — start a run to see step-by-step progress here."}
    </pre>
  );
}
