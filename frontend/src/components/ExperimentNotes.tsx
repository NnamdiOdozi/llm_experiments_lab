import { useState, useEffect, useRef } from "react";
import { updateNotes, fetchExperiment } from "../hooks/useApi";

interface Props {
  experimentId: number | null;
}

// Notes accumulate at the experiment level across all its runs — e.g.
// "run 1: loss too noisy, run 2: lowered LR, much better" (see
// docs/LLM_Experiments_Lab_Project_Discussion(2).md §6.1.1). A prior
// version of this component was run-scoped; reverted 2026-07-13.
export default function ExperimentNotes({ experimentId }: Props) {
  const [notes, setNotes] = useState("");
  const [saved, setSaved] = useState(true);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (experimentId == null) {
      setNotes("");
      setSaved(true);
      return;
    }
    fetchExperiment(experimentId).then((res) => {
      setNotes(res.notes_md || "");
      setSaved(true);
    });
  }, [experimentId]);

  function handleChange(value: string) {
    if (experimentId == null) return;
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
        disabled={experimentId == null}
        placeholder={experimentId == null ? "Select an experiment to add notes..." : "Add notes for this experiment..."}
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
