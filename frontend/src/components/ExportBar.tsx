interface Props {
  experimentId: number;
  runId: number | null;
}

export default function ExportBar({ experimentId, runId }: Props) {
  const bundleHref = `/api/code/${experimentId}/export.zip${runId != null ? `?run_id=${runId}` : ""}`;
  return (
    <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
      <a href={`/api/code/${experimentId}/export.py`} download>
        <button>Export .py</button>
      </a>
      <a href={`/api/code/${experimentId}/export.ipynb`} download>
        <button>Export .ipynb</button>
      </a>
      <a href={bundleHref} download>
        <button>Export bundle (.zip)</button>
      </a>
    </div>
  );
}
