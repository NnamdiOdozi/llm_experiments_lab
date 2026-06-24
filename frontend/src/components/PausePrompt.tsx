import { useState } from "react";
import { promptModel } from "../hooks/useApi";

interface Props {
  runId: number;
  paused: boolean;
}

export default function PausePrompt({ runId, paused }: Props) {
  const [prompt, setPrompt] = useState("");
  const [output, setOutput] = useState("");
  const [loading, setLoading] = useState(false);

  if (!paused) return null;

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
      <h3>Prompt Paused Model</h3>
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
