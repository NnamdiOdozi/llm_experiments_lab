import { TERMINAL_RUN_STATUSES, RunStatusValue } from "../types";

export type ControlAction = "pause" | "resume" | "stop" | null;

/**
 * Should a stale control-failure banner be cleared, given the run's currently
 * observed status?
 *
 * Only when the observation CONTRADICTS the action that failed. Run 287,
 * 2026-09-17: a resume timed out while the trainer was busy, the resume
 * actually took effect, and the red "Remote resume failed:" box then sat over
 * a healthy advancing run — controlError was only ever reset by clicking
 * another control.
 *
 * Clearing on any successful poll would be wrong: a failed pause or stop also
 * leaves the run running, so "running" does not disprove them and the message
 * is still true. See docs/DESIGN_DECISIONS.md.
 */
export function isControlErrorDisproved(
  action: ControlAction,
  status: RunStatusValue | null | undefined,
): boolean {
  if (action == null || !status) return false;
  if (action === "resume") return status === "running";
  if (action === "pause") return status === "paused";
  return TERMINAL_RUN_STATUSES.has(status);
}
