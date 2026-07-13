import { useState, useEffect } from "react";
import { promptModel, startDiagnostic, stepDiagnostic, generateDiagnosticStream } from "../hooks/useApi";
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
  maxNewTokens: number;
  onDiagnosticSnapshot?: (snapshot: DiagnosticSnapshot) => void;
  // Surfaces the active session id (or null once finished/not started) so
  // App.tsx can call peekDiagnostic() when Head/Block changes — that needs
  // the session id, which otherwise only lives in this component's local
  // state. See docs/DESIGN_DECISIONS.md.
  onSessionIdChange?: (sessionId: string | null) => void;
}

export default function PausePrompt({
  runId, canPrompt, attentionBlock, attentionHead, showQKVDetail, attentionWindowOffset, maxNewTokens, onDiagnosticSnapshot, onSessionIdChange,
}: Props) {
  const [prompt, setPrompt] = useState("");
  const [output, setOutput] = useState("");
  const [loading, setLoading] = useState(false);

  // Diagnostic step-through state
  const [diagnosticSession, setDiagnosticSession] = useState<DiagnosticSessionResponse | null>(null);
  const [diagnosticSnapshot, setDiagnosticSnapshot] = useState<DiagnosticSnapshot | null>(null);
  const [diagnosticLoading, setDiagnosticLoading] = useState(false);
  const [diagnosticStep, setDiagnosticStep] = useState(0);

  // Phase 3: Generated tokens displayed progressively
  const [generatedTokens, setGeneratedTokens] = useState<string>("");

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
      setOutput("");
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

  async function handleSubmit() {
    if (!prompt.trim()) return;
    setLoading(true);
    try {
      const res = await promptModel(runId, prompt);
      setOutput(res.output);
    } catch {
      setOutput("Error: could not generate text.");
    }
    setLoading(false);
  }

  async function handleStepDiagnostic() {
    if (diagnosticLoading) return;

    // Phase 1 fix: set diagnosticLoading true when request starts, not after
    setDiagnosticLoading(true);
    try {
      if (!diagnosticSession) {
        // Start a new diagnostic session
        if (!prompt.trim()) {
          alert("Enter a prompt first.");
          setDiagnosticLoading(false);
          return;
        }
        // Fresh session — clear any leftover output from a previous
        // finished cycle so it doesn't read as part of this one.
        setGeneratedTokens("");
        const session = await startDiagnostic(runId, {
          prompt: prompt.trim(),
          top_k: 5,
          max_prompt_tokens: 32,
        });
        setDiagnosticSession(session);
        onSessionIdChange?.(session.diagnostic_session_id);
        // Immediately step to get the first snapshot
        const snapshot = await stepDiagnostic(runId, session.diagnostic_session_id, {
          attention_layer: attentionBlock ?? undefined,
          attention_head: attentionHead ?? undefined,
          qkv_detail: showQKVDetail || undefined,
          attention_window_offset: attentionWindowOffset,
        });
        setDiagnosticSnapshot(snapshot);
        setDiagnosticStep(snapshot.generation_step);
        setGeneratedTokens((prev) => prev + snapshot.generated_token.text);
        onDiagnosticSnapshot?.(snapshot);
      } else {
        // Step the existing diagnostic session
        const snapshot = await stepDiagnostic(runId, diagnosticSession.diagnostic_session_id, {
          attention_layer: attentionBlock ?? undefined,
          attention_head: attentionHead ?? undefined,
          qkv_detail: showQKVDetail || undefined,
          attention_window_offset: attentionWindowOffset,
        });
        // Keep the previous snapshot visible while loading, then swap atomically
        setDiagnosticSnapshot(snapshot);
        setDiagnosticStep(snapshot.generation_step);
        setGeneratedTokens((prev) => prev + snapshot.generated_token.text);
        onDiagnosticSnapshot?.(snapshot);
      }
    } catch (err) {
      console.error("Diagnostic step error:", err);
      alert("Error stepping diagnostic: " + (err instanceof Error ? err.message : String(err)));
    }
    setDiagnosticLoading(false);
  }

  async function handleContinueGeneration() {
    if (diagnosticLoading || !diagnosticSession) return;

    setDiagnosticLoading(true);
    try {
      const generator = generateDiagnosticStream(runId, diagnosticSession.diagnostic_session_id, maxNewTokens, {
        attention_layer: attentionBlock ?? undefined,
        attention_head: attentionHead ?? undefined,
        qkv_detail: showQKVDetail || undefined,
      });
      for await (const event of generator) {
        // Check if this is a token event or a done event
        if ("text" in event) {
          // Token event: append to generated text progressively
          setGeneratedTokens((prev) => prev + event.text);
        } else if ("final_snapshot" in event) {
          // Done event: swap in final snapshot atomically
          const finalSnapshot = event.final_snapshot;
          setDiagnosticSnapshot(finalSnapshot);
          setDiagnosticStep(finalSnapshot.generation_step);
          onDiagnosticSnapshot?.(finalSnapshot);
        }
      }
      // >> always runs to a defined end (max_new_tokens) — that's a natural
      // "finished" point, so unlock the prompt automatically. The step-through
      // (>) path has no such natural end (no EOS token in this char-level
      // model), so it needs the explicit Finish button instead — see
      // handleFinishSession. See docs/DESIGN_DECISIONS.md.
      setDiagnosticSession(null);
      onSessionIdChange?.(null);
      setDiagnosticStep(0);
    } catch (err) {
      console.error("Generate stream error:", err);
      alert("Error during generation: " + (err instanceof Error ? err.message : String(err)));
    }
    setDiagnosticLoading(false);
  }

  // Ends the current step-through session and unlocks the prompt box for a
  // fresh one. Without this, once you click > there was previously no way
  // back to editing the prompt short of reloading the page — and repeated
  // >/>> clicks on the same never-ending session silently kept growing the
  // model's input sequence (no KV-cache, so every step reprocesses prompt +
  // everything generated so far) far past what the token-count label showed,
  // which is what produced a surprising [1, 82, 192] shape after what looked
  // like only a few steps. See docs/DESIGN_DECISIONS.md.
  function handleFinishSession() {
    setDiagnosticSession(null);
    onSessionIdChange?.(null);
    setDiagnosticStep(0);
  }

  return (
    <div className="panel">
      <h3>Prompt Model</h3>
      <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter a prompt..."
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          disabled={diagnosticSession !== null}
        />
        <button
          className="btn-primary"
          onClick={handleSubmit}
          disabled={loading || diagnosticSession !== null}
        >
          {loading ? "..." : "Generate"}
        </button>
      </div>

      {/* Step-through diagnostics */}
      <div style={{ marginTop: 16, paddingTop: 16, borderTop: "1px solid var(--border)" }}>
        <div style={{ marginBottom: 8, fontSize: 12 }}>
          <strong>Step-through Diagnostics</strong>
          {diagnosticSnapshot && (
            <span style={{ color: "var(--text-dim)", marginLeft: 8 }}>
              {/* generated_token.position + 1 is the REAL current sequence
                  length (prompt + everything generated so far). input_tokens
                  is only ever the original prompt — fixed, never grows — so
                  using it here previously made this label silently stale
                  after the first step. See docs/DESIGN_DECISIONS.md. */}
              (Step {diagnosticStep}, {diagnosticSnapshot.generated_token.position + 1} tokens)
            </span>
          )}
        </div>

        {/* Block/Head/Q-K-V-detail selection now lives in the Inspector pane
            (contextual to whichever attention node is selected in the
            diagram) — this panel just runs the step using whatever's
            currently configured there. See docs/DESIGN_DECISIONS.md. */}
        <div style={{ marginBottom: 12, fontSize: 11, color: "var(--text-dim)" }}>
          {/* Internal state stays 0-indexed (matches node ids, request
              payloads, Python range(n_head)) — only the display converts to
              1-indexed, matching the diagram's "Block 4 of 4" labeling.
              See docs/DESIGN_DECISIONS.md. */}
          {attentionBlock !== null ? (
            <>
              Capturing block {attentionBlock + 1}
              {attentionHead !== null ? `, head ${attentionHead + 1}` : " (pick a head in Inspector)"}
              {showQKVDetail && attentionHead !== null ? " + Q/K/V detail" : ""} —
              results appear in the <strong>Inspector</strong> tab.
            </>
          ) : (
            <>
              No attention node selected — click a block's "Causal Self-Attention" node in the
              diagram above and pick a head in the <strong>Inspector</strong> tab to capture attention/Q-K-V.
            </>
          )}
        </div>

        <div style={{ display: "flex", gap: 8 }}>
          <button
            className="btn-primary"
            onClick={handleStepDiagnostic}
            disabled={diagnosticLoading || !prompt.trim()}
            title={!prompt.trim() ? "Enter a prompt first" : "Run one autoregressive forward pass"}
          >
            {diagnosticLoading ? "..." : ">"}
          </button>
          {/* Phase 3: Enable >> button */}
          <button
            className="btn-primary"
            onClick={handleContinueGeneration}
            disabled={diagnosticLoading || !diagnosticSession}
            title={!diagnosticSession ? "Start diagnostic with > first" : "Continue generation"}
          >
            {diagnosticLoading ? "..." : ">>"}
          </button>
          {/* >> auto-finishes (it has a defined end, max_new_tokens); > alone
              has no natural end point, so this is the only way to end a
              step-through session and unlock the prompt for a new one. */}
          {diagnosticSession && (
            <button onClick={handleFinishSession} disabled={diagnosticLoading} title="End this session and edit a new prompt">
              Finish (new prompt)
            </button>
          )}
        </div>
        {generatedTokens && (
          <div style={{ marginTop: 8, fontSize: 12 }}>
            <div style={{ color: "var(--text-dim)", marginBottom: 4 }}>Output so far:</div>
            <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-word", margin: 0 }}>{generatedTokens}</pre>
          </div>
        )}
      </div>

      {output && !diagnosticSession && <pre>{output}</pre>}
    </div>
  );
}
