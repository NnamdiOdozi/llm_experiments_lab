import { useEffect, useState } from "react";
import { fetchCode } from "../hooks/useApi";
import { CodeFiles } from "../types";
import WorkerPanel from "./WorkerPanel";

interface Props {
  experimentId: number;
  runId: number | null;
}

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
      {/* Side-by-side, not a tab toggle — Template Code and Metrics used to
          be mutually exclusive views of this panel, but the panel has
          plenty of unused width to show both at once. Direct user request,
          2026-07-16. See docs/DESIGN_DECISIONS.md. */}
      <div style={{ display: "flex", gap: 16 }}>
        <div style={{ flex: "1 1 0", minWidth: 0 }}>
          <div style={{ display: "flex", gap: 6, marginBottom: 10 }}>
            {fileNames.map((f) => (
              <button key={f} onClick={() => setActiveFile(f)} style={tabStyle(f === activeFile)}>
                {f}
              </button>
            ))}
          </div>
          {/* max-height 400 matches the global `pre` rule in index.css —
              set explicitly here (rather than relying on that shared
              default) so it can't silently drift out of sync with
              WorkerPanel's own max-height again. Direct user report
              (bottom edges misaligned), 2026-07-16. See
              docs/DESIGN_DECISIONS.md. */}
          <pre style={{ maxHeight: 400, overflow: "auto" }}>{content}</pre>
        </div>
        <div style={{ flex: "1 1 0", minWidth: 0 }}>
          <div style={{ fontSize: 12, color: "var(--text-dim)", marginBottom: 10, padding: "6px 0" }}>Metrics</div>
          <WorkerPanel runId={runId} />
        </div>
      </div>
    </div>
  );
}
