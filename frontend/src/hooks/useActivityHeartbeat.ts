import { useEffect, useRef } from "react";
import { sendWorkerHeartbeat } from "./useApi";

const THROTTLE_MS = 60_000;
const ACTIVITY_EVENTS = ["scroll", "mousemove", "keydown", "click"] as const;

/**
 * Resets a remote worker's idle clock on any real page activity — scroll,
 * mouse movement, typing, clicking. Covers config edits, chatbot typing,
 * notes saves, and prompting as a side effect (all involve a click or
 * keystroke) without wiring each one separately. Throttled to at most one
 * heartbeat call per THROTTLE_MS, not one per event — a user scrolling
 * continuously shouldn't spam the backend.
 *
 * Deliberately generic (any DOM activity counts) per explicit user
 * request — see docs/DESIGN_DECISIONS.md §17 for the tradeoff this
 * accepts: a passively-open tab with occasional incidental scroll may
 * never idle-time-out. Only active for remote (nebius_endpoint) runs;
 * local runs have no worker to keep alive.
 */
export function useActivityHeartbeat(device: string, enabled: boolean) {
  const lastSentRef = useRef(0);

  useEffect(() => {
    if (!enabled) return;

    const onActivity = () => {
      const now = Date.now();
      if (now - lastSentRef.current < THROTTLE_MS) return;
      lastSentRef.current = now;
      sendWorkerHeartbeat(device).catch(() => {
        // best-effort — a missed heartbeat just means the idle clock
        // doesn't reset this time, not worth surfacing to the user
      });
    };

    for (const event of ACTIVITY_EVENTS) {
      window.addEventListener(event, onActivity, { passive: true });
    }
    return () => {
      for (const event of ACTIVITY_EVENTS) {
        window.removeEventListener(event, onActivity);
      }
    };
  }, [device, enabled]);
}
