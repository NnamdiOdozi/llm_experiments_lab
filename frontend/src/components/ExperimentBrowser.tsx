import { useEffect, useState } from "react";
import { Experiment, ExperimentConfig } from "../types";
import { listExperiments } from "../hooks/useApi";

interface Props {
  onSelect: (experimentId: number, config: ExperimentConfig) => void;
}

// Lets a user reopen a past experiment (which may already have one or more
// runs) and add a new run to it, rather than only ever starting fresh from
// a preset. Existing runs are untouched — this just loads the experiment's
// current config into the workspace with no active run selected, the same
// state PresetPicker produces for a brand-new experiment. New-run
// concurrency is still governed by the normal per-device/backend limits
// (config/settings.py) once the user clicks Start. See docs/DESIGN_DECISIONS.md.
export default function ExperimentBrowser({ onSelect }: Props) {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listExperiments()
      .then(setExperiments)
      .catch(() => setExperiments([]))
      .finally(() => setLoading(false));
  }, []);

  if (loading || experiments.length === 0) return null;

  const sorted = [...experiments].sort((a, b) => (a.updated_at < b.updated_at ? 1 : -1));

  return (
    <div className="panel" style={{ marginTop: 16 }}>
      <h3>Or Load an Existing Experiment</h3>
      <p style={{ fontSize: 12, color: "var(--text-dim)", marginBottom: 12 }}>
        Reopen a past experiment to add a new run to it. Existing runs stay untouched.
      </p>
      <div style={{ display: "flex", flexDirection: "column", gap: 6, maxHeight: 260, overflowY: "auto" }}>
        {sorted.map((exp) => (
          <button
            key={exp.id}
            onClick={() => onSelect(exp.id, exp.config)}
            style={{
              textAlign: "left",
              padding: "8px 12px",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              gap: 12,
            }}
          >
            <span>
              <span style={{ fontWeight: 600 }}>{exp.name}</span>
              <span style={{ fontSize: 12, color: "var(--text-dim)", marginLeft: 8 }}>
                #{exp.id} · {exp.config.template}
              </span>
            </span>
            <span style={{ fontSize: 11, color: "var(--text-dim)", whiteSpace: "nowrap" }}>
              Updated {new Date(exp.updated_at).toLocaleString()}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
