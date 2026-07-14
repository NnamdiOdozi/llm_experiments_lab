import { useState, useEffect } from "react";
import { startDiagnostic, stepDiagnostic, generateDiagnosticStream } from "../hooks/useApi";
import { DiagnosticSnapshot, DiagnosticSessionResponse } from "../types";

interface Props {
  runId: number;
  // A checkpoint exists for a paused run just as much as a completed one —
  // prompt_paused_model() (backend/training/runner.py) works for either.
  canPrompt: boolean;
  // Block/head/QKV-detail selection lives in App.tsx and is configured from
  // the Inspector pane (contextual to whichever attention node is selected
  // in the diagram) — this component only consumes it when running a step.
  // See docs/DESIGN_DECISIONS.md.
  attentionBlock: number | null;
  attentionHead: number | null;
  showQKVDetail: boolean;
  // Shifts the attention heatmap/qkv_detail window earlier in the sequence
  // (0 = most recent) — set from the Inspector's heatmap stepper, applied
  // here so > also captures the currently-viewed window, not always the
  // tail. See docs/DESIGN_DECISIONS.md.
  attentionWindowOffset: number;
  // Same idea, for every other node's position_vectors window — set from
  // the Inspector's per-node stepper. Direct user request, 2026-07-15. See
  // docs/DESIGN_DECISIONS.md.
  nodeWindowOffset: number;
  maxNewTokens: number;
  // Same config.inference source maxNewTokens uses — read live on every
  // render, same as maxNewTokens already was. No separate UI here:
  // ConfigPanel's existing Inference section (ConfigPanel.tsx, left
  // sidebar) is the only place to edit these, not disabled while paused —
  // editing there already takes effect on the next >/>> immediately.
  // Originally added a second, duplicate set of controls here; direct user
  // correction, 2026-07-15 ("I already see a selector... in the config on
  // the left-hand side... makes it redundant"). See
  // docs/DESIGN_DECISIONS.md.
  temperature: number;
  decodingMode: string;
  onDiagnosticSnapshot?: (snapshot: DiagnosticSnapshot) => void;
  // Surfaces the active session id (or null once finished/not started) so
  // App.tsx can call peekDiagnostic() when Head/Block changes — that needs
  // the session id, which otherwise only lives in this component's local
  // state. See docs/DESIGN_DECISIONS.md.
  onSessionIdChange?: (sessionId: string | null) => void;
}

// The separate "Generate" button was removed entirely, 2026-07-14 —
// direct user request: it hit a completely different backend route
// (/prompt) that never created a diagnostic session, so the Inspector
// never had anything to show after clicking it ("the runtime needs to
// match what's happening with the prompt model... sometimes I've pressed
// Generate and the runtime doesn't pick it up"). Now there are only two
// controls, > and >>, both driven through the same diagnostic-session
// machinery Inspector already reads from — the disconnect is gone because
// there's no longer a second, separate path.
//
// max_new_tokens is now a single TOTAL budget shared across > and >>
// within one session (previously >> always requested a fresh
// maxNewTokens on top of whatever > had already generated, so stepping
// twice then hitting >> could overshoot past the configured cap). Once
// the session's total generation_step reaches maxNewTokens, the session
// auto-closes (unlocking the prompt box for a new one) exactly like >>
// already did on its own natural end — > cannot be clicked past that
// point either, matching direct user spec: "once we've gone to the end,
// obviously you can't single-step again." See docs/DESIGN_DECISIONS.md.
export default function PausePrompt({
  runId, canPrompt, attentionBlock, attentionHead, showQKVDetail, attentionWindowOffset, nodeWindowOffset, maxNewTokens,
  temperature, decodingMode, onDiagnosticSnapshot, onSessionIdChange,
}: Props) {
  const [prompt, setPrompt] = useState("");

  // Diagnostic step-through state
  const [diagnosticSession, setDiagnosticSession] = useState<DiagnosticSessionResponse | null>(null);
  const [diagnosticSnapshot, setDiagnosticSnapshot] = useState<DiagnosticSnapshot | null>(null);
  const [diagnosticLoading, setDiagnosticLoading] = useState(false);
  const [diagnosticStep, setDiagnosticStep] = useState(0);

  // Generated text so far, shown as the one and only output line — no
  // separate history of past prompts kept on the dashboard, direct user
  // request 2026-07-14 ("don't try and keep a history of previous
  // prompts... there should only be one output line").
  const [generatedTokens, setGeneratedTokens] = useState<string>("");

  // Real deadlock bug found 2026-07-14: atCap must require an OPEN
  // session, not just "diagnosticStep happens to equal maxNewTokens".
  // diagnosticStep is deliberately left showing its final value after a
  // session closes (see closeSession() below) — the only code path that
  // ever resets it back to 0 is ensureSession(), called from inside the
  // very > / >> button handlers that atCap disables. Without the
  // `diagnosticSession !== null` check, finishing a generation at the cap
  // permanently froze both buttons: nothing could ever call
  // ensureSession() again to clear the stale step count, even after
  // typing a brand new prompt. See docs/DESIGN_DECISIONS.md.
  const atCap = diagnosticSession !== null && diagnosticStep >= maxNewTokens;

  // Real bug report, 2026-07-13: prompt a paused model, resume training,
  // pause again — the old, partially-stepped-through prompt/session was
  // still sitting here, and stepping it errored (resume loads a fresh
  // model checkpoint server-side, so the old in-memory diagnostic
  // session's model reference is stale). This component never unmounts
  // on a canPrompt flip (App.tsx keeps it mounted the whole time the run
  // exists), so state survived the round trip. Clear everything the
  // moment the run leaves paused/completed — resuming or retraining
  // should always start the next prompt from a clean slate. See
  // docs/DESIGN_DECISIONS.md.
  useEffect(() => {
    if (!canPrompt) {
      setPrompt("");
      setDiagnosticSession(null);
      setDiagnosticSnapshot(null);
      setDiagnosticStep(0);
      setGeneratedTokens("");
      onSessionIdChange?.(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canPrompt]);

  if (!canPrompt) {
    return (
      <div className="panel" style={{ opacity: 0.5 }}>
        <h3>Prompt Model</h3>
        <p style={{ fontSize: 12, color: "var(--text-dim)" }}>
          Pause or finish training to prompt the model and see its output quality.
        </p>
      </div>
    );
  }

  // Returns the active session, starting a fresh one first if none exists
  // yet — shared by both > and >>, so either control can be the first
  // thing clicked on a new prompt (direct user spec: ">> takes you to the
  // end" should work even with nothing single-stepped yet, not require a
  // warm-up > click first). startDiagnostic only tokenizes the prompt; it
  // doesn't sample anything itself, so there's no redundant extra step
  // needed before continuing into either stepDiagnostic or the >> stream.
  async function ensureSession(): Promise<DiagnosticSessionResponse> {
    if (diagnosticSession) return diagnosticSession;
    if (!prompt.trim()) throw new Error("Enter a prompt first.");
    setGeneratedTokens("");
    setDiagnosticStep(0);
    const session = await startDiagnostic(runId, {
      prompt: prompt.trim(),
      top_k: 5,
      max_prompt_tokens: 32,
    });
    setDiagnosticSession(session);
    onSessionIdChange?.(session.diagnostic_session_id);
    return session;
  }

  // Ends the current session and unlocks the prompt box for a new one.
  // Deliberately does NOT reset diagnosticStep/diagnosticSnapshot/
  // generatedTokens — those keep showing the final reached state until a
  // new prompt's first real step overwrites them in ensureSession(). Real
  // bug found 2026-07-14 via a full API trace: resetting the step count
  // to 0 immediately after finishing was what made the Inspector's step
  // counter look wrong/stale right after a completed generation. See
  // docs/DESIGN_DECISIONS.md.
  //
  // Also deliberately does NOT call onSessionIdChange?.(null) — real bug
  // found 2026-07-14: App.tsx's session-id-driven peek effect (used to
  // refresh attention when a different node/head is selected in
  // Inspector, without re-stepping) needs a live session id to target.
  // Backend diagnostic sessions have no expiry/cleanup at all (confirmed
  // in backend/training/diagnostics.py — the session registry is a plain
  // dict, nothing ever removes an entry until process restart), so the
  // backend session this id points at is still fully alive and peekable
  // long after the UI considers it "finished." Nulling the id here made
  // Inspector's attention pane permanently stuck on "click > to capture"
  // for any node selected *after* a session closed — which, since >> now
  // always auto-closes its session, was most of the time. Keeping the id
  // around lets you select the Causal Self-Attention node (or any node)
  // at any point after generation and still see real captured data for
  // the final reached state. A genuinely new prompt's ensureSession()
  // still correctly overwrites this with the new session's id, so nothing
  // stale leaks across prompts. See docs/DESIGN_DECISIONS.md.
  function closeSession() {
    setDiagnosticSession(null);
  }

  async function handleStepDiagnostic() {
    if (diagnosticLoading || atCap) return;
    setDiagnosticLoading(true);
    try {
      const session = await ensureSession();
      const snapshot = await stepDiagnostic(runId, session.diagnostic_session_id, {
        attention_layer: attentionBlock ?? undefined,
        attention_head: attentionHead ?? undefined,
        qkv_detail: showQKVDetail || undefined,
        attention_window_offset: attentionWindowOffset,
        node_window_offset: nodeWindowOffset,
        temperature,
        decoding_mode: decodingMode,
      });
      // Keep the previous snapshot visible while loading, then swap atomically
      setDiagnosticSnapshot(snapshot);
      setDiagnosticStep(snapshot.generation_step);
      setGeneratedTokens((prev) => prev + snapshot.generated_token.text);
      onDiagnosticSnapshot?.(snapshot);
      if (snapshot.generation_step >= maxNewTokens) closeSession();
    } catch (err) {
      console.error("Diagnostic step error:", err);
      alert("Error stepping diagnostic: " + (err instanceof Error ? err.message : String(err)));
    }
    setDiagnosticLoading(false);
  }

  async function handleContinueGeneration() {
    if (diagnosticLoading || atCap) return;
    setDiagnosticLoading(true);
    try {
      const session = await ensureSession();
      // Total budget minus whatever's already been generated in this
      // session (via prior > clicks) — not always the full maxNewTokens,
      // otherwise stepping twice then hitting >> would overshoot past the
      // configured cap. Direct user spec, 2026-07-14. See
      // docs/DESIGN_DECISIONS.md.
      const remaining = maxNewTokens - diagnosticStep;
      const generator = generateDiagnosticStream(runId, session.diagnostic_session_id, remaining, {
        attention_layer: attentionBlock ?? undefined,
        attention_head: attentionHead ?? undefined,
        qkv_detail: showQKVDetail || undefined,
        node_window_offset: nodeWindowOffset,
        temperature,
        decoding_mode: decodingMode,
      });
      for await (const event of generator) {
        if ("text" in event) {
          setGeneratedTokens((prev) => prev + event.text);
        } else if ("final_snapshot" in event) {
          const finalSnapshot = event.final_snapshot;
          setDiagnosticSnapshot(finalSnapshot);
          setDiagnosticStep(finalSnapshot.generation_step);
          onDiagnosticSnapshot?.(finalSnapshot);
        }
      }
      // >> always runs to the total budget — a defined end — so unlock
      // the prompt box automatically. See docs/DESIGN_DECISIONS.md.
      closeSession();
    } catch (err) {
      console.error("Generate stream error:", err);
      alert("Error during generation: " + (err instanceof Error ? err.message : String(err)));
    }
    setDiagnosticLoading(false);
  }

  // Manual early exit — lets the user abandon a >-only session before
  // reaching maxNewTokens and start a fresh prompt without going all the
  // way via >>.
  function handleFinishSession() {
    closeSession();
  }

  return (
    <div className="panel">
      <h3>Prompt Model</h3>
      <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter a prompt..."
          disabled={diagnosticSession !== null}
        />
      </div>

      <div style={{ marginBottom: 8, fontSize: 12 }}>
        {diagnosticSnapshot && (
          <span style={{ color: "var(--text-dim)" }}>
            {/* generated_token.position + 1 is the REAL current sequence
                length (prompt + everything generated so far). input_tokens
                is only ever the original prompt — fixed, never grows — so
                using it here previously made this label silently stale
                after the first step. See docs/DESIGN_DECISIONS.md. */}
            Step {diagnosticStep} of {maxNewTokens}, {diagnosticSnapshot.generated_token.position + 1} tokens
            {atCap && " — done, enter a new prompt"}
          </span>
        )}
      </div>

      {/* Block/Head/Q-K-V-detail selection lives in the Inspector pane
          (contextual to whichever attention node is selected in the
          diagram) — this panel just runs the step using whatever's
          currently configured there. The status note that used to live
          here ("No attention node selected...") was removed entirely,
          direct user request 2026-07-14 — wrong place for it, didn't do
          anything useful, and duplicated Inspector's own contextual
          messaging. See docs/DESIGN_DECISIONS.md. */}

      <div style={{ display: "flex", gap: 8 }}>
        <button
          className="btn-primary"
          onClick={handleStepDiagnostic}
          disabled={diagnosticLoading || atCap || (diagnosticSession == null && !prompt.trim())}
          title={atCap ? "Already at max_new_tokens — enter a new prompt" : !prompt.trim() && !diagnosticSession ? "Enter a prompt first" : "Run one autoregressive forward pass"}
        >
          {diagnosticLoading ? "..." : ">"}
        </button>
        <button
          className="btn-primary"
          onClick={handleContinueGeneration}
          disabled={diagnosticLoading || atCap || (diagnosticSession == null && !prompt.trim())}
          title={atCap ? "Already at max_new_tokens — enter a new prompt" : !prompt.trim() && !diagnosticSession ? "Enter a prompt first" : "Continue generation to max_new_tokens"}
        >
          {diagnosticLoading ? "..." : ">>"}
        </button>
        {diagnosticSession && !atCap && (
          <button onClick={handleFinishSession} disabled={diagnosticLoading} title="End this session and edit a new prompt">
            Finish (new prompt)
          </button>
        )}
      </div>

      {generatedTokens && (
        <div style={{ marginTop: 8, fontSize: 12 }}>
          <div style={{ color: "var(--text-dim)", marginBottom: 4 }}>Output:</div>
          <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-word", margin: 0 }}>{generatedTokens}</pre>
        </div>
      )}
    </div>
  );
}
