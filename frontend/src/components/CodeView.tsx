import { useEffect, useState } from "react";
import { fetchCode } from "../hooks/useApi";
import { CodeFiles } from "../types";
import WorkerPanel from "./WorkerPanel";

interface Props {
  experimentId: number;
  runId: number | null;
  device: string;
}

const WORKER_TAB = "__worker__";

export default function CodeView({ experimentId, runId, device }: Props) {
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
      <div style={{ display: "flex", gap: 6, marginBottom: 10 }}>
        {fileNames.map((f) => (
          <button key={f} onClick={() => setActiveFile(f)} style={tabStyle(f === activeFile)}>
            {f}
          </button>
        ))}
        <button
          onClick={() => setActiveFile(WORKER_TAB)}
          style={{ ...tabStyle(activeFile === WORKER_TAB), marginLeft: 16, paddingLeft: 12, borderLeft: "1px solid var(--border)" }}
        >
          Serverless
        </button>
      </div>
      {activeFile === WORKER_TAB ? (
        <WorkerPanel runId={runId} device={device} />
      ) : (
        <pre>{content}</pre>
      )}
    </div>
  );
}
