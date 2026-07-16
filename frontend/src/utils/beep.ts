// Gentle audio cue when training actually starts — direct user request,
// 2026-07-15: serverless endpoints can sit queued for minutes waiting on
// Nebius to provision, and it's easy to step away and miss the moment
// training actually begins. Synthesized via Web Audio API (no asset file to
// ship/host). See docs/DESIGN_DECISIONS.md.
//
// Real bug, 2026-07-15: didn't fire on some starts. Two causes, both fixed
// here:
// 1. The original trigger was "prevStatus===queued && current===running" —
//    a fast local run can go straight to running before the frontend's
//    first poll ever observes a queued reading, so the transition looks
//    like null->running, which the predicate deliberately treats as a no-op
//    (to avoid beeping on page reload of an already-running run). Now
//    App.tsx tracks an explicit "a start was just requested" flag instead
//    of inferring intent from the status sequence — set true in
//    handleStart(), cleared once the run reaches running (fires) or any
//    terminal status (gives up silently).
// 2. Browsers restrict AudioContext creation/resume to inside a direct
//    user-gesture call stack (autoplay policy). The context was previously
//    created lazily the first time a beep actually played — by then we're
//    several polling ticks removed from the original click, so the browser
//    can leave it permanently suspended. unlockAudioContext() must be
//    called synchronously inside the Start-button's own click handler.

let audioCtx: AudioContext | null = null;

function getAudioContext(): AudioContext {
  if (!audioCtx) {
    audioCtx = new AudioContext();
  }
  return audioCtx;
}

// Call synchronously from within the Start-button click handler — creates
// (and resumes, if the browser started it suspended) the AudioContext while
// still inside a real user gesture, so it's already unlocked by the time
// the beep actually needs to play, potentially minutes later.
export function unlockAudioContext() {
  const ctx = getAudioContext();
  if (ctx.state === "suspended") {
    void ctx.resume();
  }
}

export function playTrainingStartBeep() {
  const ctx = getAudioContext();
  if (ctx.state === "suspended") {
    void ctx.resume();
  }
  const oscillator = ctx.createOscillator();
  const gain = ctx.createGain();
  oscillator.type = "sine";
  oscillator.frequency.value = 880; // A5 — gentle, not alarming
  oscillator.connect(gain);
  gain.connect(ctx.destination);

  // Short attack/decay envelope so it's a soft "blip", not a harsh click.
  // Peak 0.9 = original 0.15 raised 500% — user request 2026-07-16.
  const now = ctx.currentTime;
  gain.gain.setValueAtTime(0, now);
  gain.gain.linearRampToValueAtTime(0.9, now + 0.10);
  gain.gain.linearRampToValueAtTime(0, now + 0.35);

  oscillator.start(now);
  oscillator.stop(now + 0.35);
}

// Extracted as a pure predicate so the exact trigger condition is unit-
// testable without mounting App.tsx. awaitingStart is true only between a
// user's own handleStart() call and that run reaching running/terminal —
// not inferred from the status sequence, so it's immune to a fast local run
// skipping past an observable "queued" reading.
export function shouldPlayTrainingStartBeep(awaitingStart: boolean, currentStatus: string | null): boolean {
  return awaitingStart && currentStatus === "running";
}

const SOUND_MUTED_KEY = "llm_lab_sound_muted";

export function isSoundMuted(): boolean {
  return localStorage.getItem(SOUND_MUTED_KEY) === "true";
}

export function setSoundMuted(muted: boolean) {
  localStorage.setItem(SOUND_MUTED_KEY, muted ? "true" : "false");
}
