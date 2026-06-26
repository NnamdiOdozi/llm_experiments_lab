import { useState, useEffect, useRef } from "react";
import { updateNotes, fetchExperiment } from "../hooks/useApi";

interface Props {
  experimentId: number;
}

export default function ExperimentNotes({ experimentId }: Props) {
  const [notes, setNotes] = useState("");
  const [saved, setSaved] = useState(true);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    fetchExperiment(experimentId).then((exp) => {
      setNotes(exp.notes_md || "");
    });
  }, [experimentId]);

  function handleChange(value: string) {
    setNotes(value);
    setSaved(false);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(async () => {
      await updateNotes(experimentId, value);
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
        placeholder="Add experiment notes here..."
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
