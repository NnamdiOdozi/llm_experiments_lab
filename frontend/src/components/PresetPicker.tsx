import { useEffect, useState } from "react";
import { Preset, ExperimentConfig } from "../types";
import { fetchPresets, createFromPreset } from "../hooks/useApi";

interface Props {
  onSelect: (experimentId: number, config: ExperimentConfig, device: string, backend: string) => void;
}

export default function PresetPicker({ onSelect }: Props) {
  const [presets, setPresets] = useState<Preset[]>([]);
  const [device, setDevice] = useState("cpu");
  const [backend, setBackend] = useState("local");

  useEffect(() => {
    fetchPresets().then(setPresets);
  }, []);

  async function handlePick(p: Preset) {
    const { experiment_id } = await createFromPreset(p.key);
    const config: ExperimentConfig = {
      template: p.template,
      model: p.model,
      training: p.training,
    };
    onSelect(experiment_id, config, device, backend);
  }

  return (
    <div className="panel">
      <h3>Choose a Preset</h3>
      <div style={{ marginBottom: 12, display: "flex", gap: 16 }}>
        <div>
          <label style={{ fontSize: 12, color: "var(--text-dim)", marginRight: 8 }}>Device</label>
          <select
            value={device}
            onChange={(e) => setDevice(e.target.value)}
            style={{ fontSize: 12, padding: "4px 8px" }}
          >
            <option value="cpu">CPU</option>
            <option value="cuda">GPU (CUDA)</option>
          </select>
        </div>
        <div>
          <label style={{ fontSize: 12, color: "var(--text-dim)", marginRight: 8 }}>Backend</label>
          <select
            value={backend}
            onChange={(e) => setBackend(e.target.value)}
            style={{ fontSize: 12, padding: "4px 8px" }}
          >
            <option value="local">Local</option>
            <option value="nebius_endpoint">Serverless (Nebius)</option>
          </select>
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        {presets.map((p) => (
          <button
            key={p.key}
            onClick={() => handlePick(p)}
            style={{ textAlign: "left", padding: 12 }}
          >
            <div style={{ fontWeight: 600, marginBottom: 4 }}>{p.name}</div>
            <div style={{ fontSize: 12, color: "var(--text-dim)" }}>
              {p.description}
            </div>
            <div
              className={`tag ${p.template === "rnn" ? "tag-paused" : "tag-running"}`}
              style={{ marginTop: 6 }}
            >
              {p.template}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
