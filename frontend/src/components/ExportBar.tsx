interface Props {
  experimentId: number;
}

export default function ExportBar({ experimentId }: Props) {
  return (
    <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
      <a href={`/api/code/${experimentId}/export.py`} download>
        <button>Export .py</button>
      </a>
      <a href={`/api/code/${experimentId}/export.ipynb`} download>
        <button>Export .ipynb</button>
      </a>
    </div>
  );
}
