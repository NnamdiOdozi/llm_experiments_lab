import { useState } from "react";
import { promptModel } from "../hooks/useApi";

interface Props {
  runId: number;
  // A checkpoint exists for a paused run just as much as a completed one —
  // prompt_paused_model() (backend/training/runner.py) works for either.
  canPrompt: boolean;
}

export default function PausePrompt({ runId, canPrompt }: Props) {
  const [prompt, setPrompt] = useState("");
  const [output, setOutput] = useState("");
  const [loading, setLoading] = useState(false);

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

  return (
    <div className="panel">
      <h3>Prompt Model</h3>
      <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter a prompt..."
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
        />
        <button className="btn-primary" onClick={handleSubmit} disabled={loading}>
          {loading ? "..." : "Generate"}
        </button>
      </div>
      {output && <pre>{output}</pre>}
    </div>
  );
}
