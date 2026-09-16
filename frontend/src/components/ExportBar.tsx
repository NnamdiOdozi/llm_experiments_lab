interface Props {
  experimentId: number;
  runId: number | null;
}

export default function ExportBar({ experimentId, runId }: Props) {
  // Pass the run through on ALL three, not just the bundle: the backend exports
  // the run's config snapshot when it has one (what actually trained) rather
  // than the experiment's current config, which may have been edited since.
  const runQuery = runId != null ? `?run_id=${runId}` : "";
  const bundleHref = `/api/code/${experimentId}/export.zip${runQuery}`;
  return (
    <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
      <a href={`/api/code/${experimentId}/export.py${runQuery}`} download>
        <button>Export .py</button>
      </a>
      <a href={`/api/code/${experimentId}/export.ipynb${runQuery}`} download>
        <button>Export .ipynb</button>
      </a>
      <a href={bundleHref} download>
        <button>Export bundle (.zip)</button>
      </a>
    </div>
  );
}
