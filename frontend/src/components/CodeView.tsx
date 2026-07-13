import { useEffect, useState } from "react";
import { fetchCode } from "../hooks/useApi";
import { CodeFiles } from "../types";
import WorkerPanel from "./WorkerPanel";

interface Props {
  experimentId: number;
  runId: number | null;
}

const WORKER_TAB = "__worker__";

export default function CodeView({ experimentId, runId }: Props) {
  const [code, setCode] = useState<CodeFiles | null>(null);
  const [activeFile, setActiveFile] = useState("model.py");

  useEffect(() => {
    fetchCode(experimentId).then(setCode);
  }, [experimentId]);

  if (!code) return null;

  const fileNames = Object.keys(code.files);
  const content = code.files[activeFile] || "// file not found";
  const tabStyle = (active: boolean) => ({
    fontSize: 12,
    background: active ? "var(--accent-dim)" : undefined,
    color: active ? "white" : undefined,
  });

  return (
    <div className="panel">
      <h3>Template Code ({code.template})</h3>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <div style={{ display: "flex", gap: 6 }}>
          {fileNames.map((f) => (
            <button key={f} onClick={() => setActiveFile(f)} style={tabStyle(f === activeFile)}>
              {f}
            </button>
          ))}
        </div>
        <button onClick={() => setActiveFile(WORKER_TAB)} style={tabStyle(activeFile === WORKER_TAB)}>
          Serverless Metrics
        </button>
      </div>
      {activeFile === WORKER_TAB ? (
        <WorkerPanel runId={runId} />
      ) : (
        <pre>{content}</pre>
      )}
    </div>
  );
}
