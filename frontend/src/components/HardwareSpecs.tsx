import { useEffect, useState } from "react";
import { getWorkerStatus, WorkerStatus } from "../hooks/useApi";

interface Props {
  // Current run's device/backend. Omit (landing page, no run yet) to show
  // both CPU and GPU serverless specs informationally. When given: a local
  // backend has nothing serverless to show ("Local" only); a serverless
  // backend shows only the device actually in use, not both — showing the
  // GPU spec while running on CPU (or vice versa) is misleading.
  device?: string;
  backend?: string;
}

// Shows what CPU/GPU hardware a serverless run actually uses, so the user
// doesn't have to dig into config/settings.py to find out (e.g. "L40" GPU).
// Prefers the *actual* live spec (worker_sessions.actual_platform/preset,
// captured from the real Nebius response) once an endpoint has run; falls
// back to the *configured* spec (settings.nebius_*_platform/preset) before
// that, labeled distinctly since configured values are marked as unverified
// placeholders in settings.py and can diverge from what actually launches.
// See docs/DESIGN_DECISIONS.md §9, §33, §34.
export default function HardwareSpecs({ device, backend }: Props) {
  const [cpu, setCpu] = useState<WorkerStatus | null>(null);
  const [gpu, setGpu] = useState<WorkerStatus | null>(null);

  const showCpu = backend !== "local" && (backend == null || device === "cpu");
  const showGpu = backend !== "local" && (backend == null || device === "cuda");

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      showCpu ? getWorkerStatus("cpu") : Promise.resolve(null),
      showGpu ? getWorkerStatus("cuda") : Promise.resolve(null),
    ]).then(([c, g]) => {
      if (cancelled) return;
      setCpu(c);
      setGpu(g);
    });
    return () => { cancelled = true; };
  }, [showCpu, showGpu]);

  function line(label: string, status: WorkerStatus | null) {
    if (status == null) return `${label}: …`;
    const live = status.actual_platform != null || status.preset != null;
    const platform = live ? status.actual_platform : status.configured_platform;
    const preset = live ? status.preset : status.configured_preset;
    const tag = live ? "live" : "configured";
    return `${label}: ${platform ?? "?"} · ${preset ?? "?"} (${tag})`;
  }

  if (backend === "local") {
    return <div style={{ fontSize: 12, color: "var(--text-dim)" }}>Local</div>;
  }

  return (
    <div style={{ fontSize: 12, color: "var(--text-dim)", display: "flex", gap: 16 }}>
      {showCpu && <span>Serverless {line("CPU", cpu)}</span>}
      {showGpu && <span>Serverless {line("GPU", gpu)}</span>}
    </div>
  );
}
