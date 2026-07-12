import { useState, useEffect, useRef } from "react";
import { updateRunNotes, fetchRunNotes } from "../hooks/useApi";

interface Props {
  runId: number | null;
}

export default function ExperimentNotes({ runId }: Props) {
  const [notes, setNotes] = useState("");
  const [saved, setSaved] = useState(true);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (runId == null) {
      setNotes("");
      setSaved(true);
      return;
    }
    fetchRunNotes(runId).then((res) => {
      setNotes(res.notes_md || "");
      setSaved(true);
    });
  }, [runId]);

  function handleChange(value: string) {
    if (runId == null) return;
    setNotes(value);
    setSaved(false);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(async () => {
      await updateRunNotes(runId, value);
      setSaved(true);
    }, 1000);
  }

  return (
    <div className="panel">
      <h3>
        Notes
        <span style={{ fontSize: 11, color: saved ? "var(--green)" : "var(--text-dim)", marginLeft: 8 }}>
          {saved ? "saved" : "saving..."}
        </span>
      </h3>
      <textarea
        value={notes}
        onChange={(e) => handleChange(e.target.value)}
        disabled={runId == null}
        placeholder={runId == null ? "Start a run to add notes..." : "Add notes for this run..."}
        style={{
          width: "100%",
          minHeight: 100,
          resize: "vertical",
          fontSize: 13,
          fontFamily: "inherit",
          background: "var(--bg)",
          color: "var(--text)",
          border: "1px solid var(--border)",
          borderRadius: 4,
          padding: 8,
        }}
      />
    </div>
  );
}
