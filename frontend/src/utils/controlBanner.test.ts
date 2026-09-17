import { describe, it, expect } from "vitest";
import { isControlErrorDisproved } from "./controlBanner";
import { RunStatusValue } from "../types";

describe("isControlErrorDisproved", () => {
  it("clears a failed resume once the run is observed running", () => {
    // The run 287 case: resume timed out but actually took effect.
    expect(isControlErrorDisproved("resume", "running")).toBe(true);
  });

  it("keeps a failed pause while the run is still running", () => {
    // "running" does NOT disprove a failed pause — the message is still true.
    expect(isControlErrorDisproved("pause", "running")).toBe(false);
  });

  it("clears a failed pause once the run is observed paused", () => {
    expect(isControlErrorDisproved("pause", "paused")).toBe(true);
  });

  it("keeps a failed stop while the run is still running", () => {
    expect(isControlErrorDisproved("stop", "running")).toBe(false);
  });

  it.each<RunStatusValue>(["completed", "failed", "cancelled"])(
    "clears a failed stop once the run reaches %s",
    (status) => {
      expect(isControlErrorDisproved("stop", status)).toBe(true);
    },
  );

  it("does nothing when no control action has failed", () => {
    expect(isControlErrorDisproved(null, "running")).toBe(false);
  });

  it("does nothing when the status is unknown", () => {
    expect(isControlErrorDisproved("resume", null)).toBe(false);
    expect(isControlErrorDisproved("resume", undefined)).toBe(false);
  });
});
