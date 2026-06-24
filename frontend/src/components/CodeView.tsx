import { useEffect, useState } from "react";
import { fetchCode } from "../hooks/useApi";
import { CodeFiles } from "../types";

interface Props {
  experimentId: number;
}

export default function CodeView({ experimentId }: Props) {
  const [code, setCode] = useState<CodeFiles | null>(null);
  const [activeFile, setActiveFile] = useState("model.py");

  useEffect(() => {
    fetchCode(experimentId).then(setCode);
  }, [experimentId]);

  if (!code) return null;

  const fileNames = Object.keys(code.files);
  const content = code.files[activeFile] || "// file not found";

  return (
    <div className="panel">
      <h3>Template Code ({code.template})</h3>
      <div style={{ display: "flex", gap: 6, marginBottom: 10 }}>
        {fileNames.map((f) => (
          <button
            key={f}
            onClick={() => setActiveFile(f)}
            style={{
              fontSize: 12,
              background: f === activeFile ? "var(--accent-dim)" : undefined,
              color: f === activeFile ? "white" : undefined,
            }}
          >
            {f}
          </button>
        ))}
      </div>
      <pre>{content}</pre>
    </div>
  );
}
