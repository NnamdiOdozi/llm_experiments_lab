import { useEffect, useRef, useState } from "react";
import { getWorkerStatus, sendWorkerHeartbeat, WorkerStatus } from "../hooks/useApi";

interface Props {
  device: string;
}

const POLL_INTERVAL_MS = 15000;

function formatRemaining(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}m ${s}s`;
}

export default function WorkerIdleBanner({ device }: Props) {
  const [status, setStatus] = useState<WorkerStatus | null>(null);
  const [stoppedDismissed, setStoppedDismissed] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Only show "stopped due to inactivity" if we actually watched it happen
  // (saw the worker ready, then later stopped) during this page load — not
  // on a cold/first load of the day, where the worker is already stopped
  // before we ever polled it and the message reads as a stale alarm rather
  // than the "you went idle" heads-up it's meant to be. Direct user
  // report, 2026-07-14: "I've been the one who's done stuff... it's the
  // first time this morning." See docs/DESIGN_DECISIONS.md.
  const sawReadyRef = useRef(false);

  async function poll() {
    try {
      const s = await getWorkerStatus(device);
      if (s.worker_status === "ready") sawReadyRef.current = true;
      setStatus(s);
    } catch {
      // Idle-status polling is best-effort — a transient failure just skips this tick
    }
  }

  useEffect(() => {
    setStoppedDismissed(false);
    sawReadyRef.current = false;
    poll();
    pollRef.current = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [device]);

  async function handleContinueSession() {
    await sendWorkerHeartbeat(device);
    poll();
  }

  if (!status || status.worker_status === "none") return null;

  // Part 9: Handle terminal/unavailable states (SHUTTING_DOWN, FAILED, STOPPED)
  if (status.worker_status === "shutting_down") {
    return (
      <div style={{
        background: "var(--orange)", color: "#1a1a1a", padding: "8px 16px",
        borderRadius: 6, marginBottom: 12, fontSize: 13,
      }}>
        This {device.toUpperCase()} worker is shutting down. A new one will be provisioned for the next run.
      </div>
    );
  }

  if (status.worker_status === "failed") {
    return (
      <div style={{
        background: "var(--red)", color: "#fff", padding: "8px 16px",
        borderRadius: 6, marginBottom: 12, fontSize: 13,
      }}>
        This {device.toUpperCase()} worker encountered an error. Starting a run will provision a new one.
      </div>
    );
  }

  if (status.worker_status === "stopped") {
    if (stoppedDismissed || !sawReadyRef.current) return null;
    return (
      <div style={{
        background: "var(--text-dim)", color: "#1a1a1a", padding: "8px 16px",
        borderRadius: 6, marginBottom: 12, fontSize: 13,
        display: "flex", justifyContent: "space-between", alignItems: "center",
      }}>
        <span>
          This {device.toUpperCase()} worker was stopped due to inactivity. Starting a run will provision a new one.
        </span>
        <button
          style={{ background: "rgba(0,0,0,0.15)", border: "none", color: "#1a1a1a", padding: "4px 10px", borderRadius: 4, cursor: "pointer" }}
          onClick={() => setStoppedDismissed(true)}
        >
          Dismiss
        </button>
      </div>
    );
  }

  if (
    status.worker_status !== "ready"
    || status.seconds_idle == null
    || status.idle_timeout_seconds == null
    || status.warning_seconds == null
  ) {
    return null;
  }

  const remaining = status.idle_timeout_seconds - status.seconds_idle;
  if (remaining > status.warning_seconds) return null;

  return (
    <div style={{
      background: "var(--yellow)", color: "#1a1a1a", padding: "8px 16px",
      borderRadius: 6, marginBottom: 12, fontSize: 13,
      display: "flex", justifyContent: "space-between", alignItems: "center",
    }}>
      <span>
        This {device.toUpperCase()} worker will stop in {formatRemaining(Math.max(remaining, 0))} due to inactivity.
      </span>
      <button
        style={{ background: "rgba(0,0,0,0.15)", border: "none", color: "#1a1a1a", padding: "4px 10px", borderRadius: 4, cursor: "pointer" }}
        onClick={handleContinueSession}
      >
        Continue session
      </button>
    </div>
  );
}
