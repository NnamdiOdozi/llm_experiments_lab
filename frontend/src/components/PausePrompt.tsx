import { useState } from "react";
import { promptModel, startDiagnostic, stepDiagnostic, generateDiagnosticStream } from "../hooks/useApi";
import { DiagnosticSnapshot, DiagnosticSessionResponse } from "../types";

interface Props {
  runId: number;
  // A checkpoint exists for a paused run just as much as a completed one —
  // prompt_paused_model() (backend/training/runner.py) works for either.
  canPrompt: boolean;
  onDiagnosticSnapshot?: (snapshot: DiagnosticSnapshot) => void;
}

export default function PausePrompt({ runId, canPrompt, onDiagnosticSnapshot }: Props) {
  const [prompt, setPrompt] = useState("");
  const [output, setOutput] = useState("");
  const [loading, setLoading] = useState(false);

  // Diagnostic step-through state
  const [diagnosticSession, setDiagnosticSession] = useState<DiagnosticSessionResponse | null>(null);
  const [diagnosticSnapshot, setDiagnosticSnapshot] = useState<DiagnosticSnapshot | null>(null);
  const [diagnosticLoading, setDiagnosticLoading] = useState(false);
  const [diagnosticStep, setDiagnosticStep] = useState(0);

  // Phase 2: Attention layer/head selectors
  const [attentionLayer, setAttentionLayer] = useState<number | null>(null);
  const [attentionHead, setAttentionHead] = useState<number | null>(null);

  // Phase 4: Q/K/V detail toggle
  const [showQKVDetail, setShowQKVDetail] = useState(false);

  // Phase 3: Generated tokens displayed progressively
  const [generatedTokens, setGeneratedTokens] = useState<string>("");

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
        // Immediately step to get the first snapshot
        const snapshot = await stepDiagnostic(runId, session.diagnostic_session_id, {
          attention_layer: attentionLayer ?? undefined,
          attention_head: attentionHead ?? undefined,
          qkv_detail: showQKVDetail || undefined,
        });
        setDiagnosticSnapshot(snapshot);
        setDiagnosticStep(snapshot.generation_step);
        setGeneratedTokens((prev) => prev + snapshot.generated_token.text);
        onDiagnosticSnapshot?.(snapshot);
      } else {
        // Step the existing diagnostic session
        const snapshot = await stepDiagnostic(runId, diagnosticSession.diagnostic_session_id, {
          attention_layer: attentionLayer ?? undefined,
          attention_head: attentionHead ?? undefined,
          qkv_detail: showQKVDetail || undefined,
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
      const generator = generateDiagnosticStream(runId, diagnosticSession.diagnostic_session_id, 50);
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

        {/* Phase 2: Attention layer/head selectors */}
        <div style={{ marginBottom: 12, display: "flex", gap: 8, fontSize: 12 }}>
          <label>
            Layer:
            <input
              type="number"
              min={0}
              value={attentionLayer ?? ""}
              onChange={(e) => setAttentionLayer(e.target.value === "" ? null : parseInt(e.target.value, 10))}
              placeholder="0-3"
              style={{ width: 50, marginLeft: 4 }}
            />
          </label>
          <label>
            Head:
            <input
              type="number"
              min={0}
              value={attentionHead ?? ""}
              onChange={(e) => setAttentionHead(e.target.value === "" ? null : parseInt(e.target.value, 10))}
              placeholder="0-5"
              style={{ width: 50, marginLeft: 4 }}
            />
          </label>
          <span style={{ color: "var(--text-dim)", fontSize: 11, marginTop: 4 }}>
            (Leave blank to skip attention capture)
          </span>
        </div>

        {/* Phase 4: Q/K/V detail checkbox */}
        <div style={{ marginBottom: 4, display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
          <input
            type="checkbox"
            id="show-qkv"
            checked={showQKVDetail}
            onChange={(e) => setShowQKVDetail(e.target.checked)}
            disabled={attentionLayer === null || attentionHead === null}
            // Unstyled checkboxes render as the browser's native grey/white
            // widget, which clashes badly against the dark theme (looked
            // like a mystery empty square with no context) — accentColor is
            // the simplest cross-browser way to theme a native checkbox.
            style={{ accentColor: "var(--accent)" }}
          />
          <label htmlFor="show-qkv" style={{ cursor: "pointer" }}>
            Show Q/K/V detail
          </label>
          <span style={{ color: "var(--text-dim)", fontSize: 11 }}>
            (requires layer & head)
          </span>
        </div>
        {/* Setting Layer/Head only controls WHAT gets captured — it doesn't
            display anything here. The actual heatmap/Q-K-V values render in
            the Inspector tab, only after clicking that block's "Causal
            Self-Attention" node in the Architecture diagram above. This was
            genuinely not discoverable before — nothing here pointed there. */}
        <div style={{ marginBottom: 12, fontSize: 11, color: "var(--text-dim)" }}>
          Captured attention/Q-K-V appears in the <strong>Inspector</strong> tab (right pane) —
          click the block's "Causal Self-Attention" node in the diagram above to view it.
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
        {/* > and >> always pick the single highest-probability token
            (greedy decoding) — deliberately, so this always matches the
            Top-k panel's #1 entry exactly. Generate above instead samples
            with temperature (config.inference.temperature). Greedy decoding
            on an early-training model can fall into a repetition loop
            (e.g. "the the the the") — that's a real, well-known decoding
            behavior, not a bug, but nothing else here explains why >/>>
            output looks different from Generate's. See docs/DESIGN_DECISIONS.md. */}
        <div style={{ marginTop: 4, fontSize: 11, color: "var(--text-dim)" }}>
          &gt;/&gt;&gt; always pick the single most-likely token (greedy) — matches the Top-k panel exactly,
          but can loop on an early-training model. Generate above samples with temperature instead.
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
