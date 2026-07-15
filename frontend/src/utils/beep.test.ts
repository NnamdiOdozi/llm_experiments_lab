import { describe, it, expect, beforeEach } from "vitest";
import { shouldPlayTrainingStartBeep, isSoundMuted, setSoundMuted } from "./beep";

// Direct user request, 2026-07-15: a gentle beep when training actually
// starts — easy to miss otherwise while a serverless endpoint provisions.
// Real bug, 2026-07-15: the original design inferred intent from an
// observed queued->running status transition, but a fast local run can
// jump straight to running before the frontend's first poll ever sees
// "queued", making it indistinguishable from a page reload of an
// already-running run (which should NOT beep). Redesigned around an
// explicit "a start was just requested" flag (awaitingStart) instead —
// set by handleStart() itself, not inferred from status history.
describe("shouldPlayTrainingStartBeep", () => {
  it("fires once awaitingStart is true and status reads running", () => {
    expect(shouldPlayTrainingStartBeep(true, "running")).toBe(true);
  });

  it("does not fire if awaitingStart is false, even if running (e.g. page reload)", () => {
    expect(shouldPlayTrainingStartBeep(false, "running")).toBe(false);
  });

  it("does not fire while still queued", () => {
    expect(shouldPlayTrainingStartBeep(true, "queued")).toBe(false);
  });

  it("does not fire on a terminal status", () => {
    expect(shouldPlayTrainingStartBeep(true, "completed")).toBe(false);
  });

  it("does not fire on a null status", () => {
    expect(shouldPlayTrainingStartBeep(true, null)).toBe(false);
  });
});

describe("sound-muted persistence", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("defaults to unmuted", () => {
    expect(isSoundMuted()).toBe(false);
  });

  it("persists a muted preference across reads", () => {
    setSoundMuted(true);
    expect(isSoundMuted()).toBe(true);
    setSoundMuted(false);
    expect(isSoundMuted()).toBe(false);
  });
});
