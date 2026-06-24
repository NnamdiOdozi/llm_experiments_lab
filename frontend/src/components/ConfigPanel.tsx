import { ExperimentConfig } from "../types";

interface Props {
  config: ExperimentConfig;
  onChange: (config: ExperimentConfig) => void;
  disabled?: boolean;
}

function renderSection(
  title: string,
  section: Record<string, number | string>,
  sectionKey: "model" | "training",
  config: ExperimentConfig,
  onChange: Props["onChange"],
  disabled: boolean
) {
  return (
    <div style={{ marginBottom: 16 }}>
      <h4 style={{ fontSize: 12, color: "var(--accent)", marginBottom: 8 }}>
        {title}
      </h4>
      {Object.entries(section).map(([key, val]) => (
        <div
          key={key}
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 6,
          }}
        >
          <label style={{ fontSize: 12, color: "var(--text-dim)" }}>{key}</label>
          <input
            style={{ width: 100, textAlign: "right" }}
            value={val}
            disabled={disabled}
            onChange={(e) => {
              const v = e.target.value;
              const parsed = isNaN(Number(v)) ? v : Number(v);
              onChange({
                ...config,
                [sectionKey]: { ...config[sectionKey], [key]: parsed },
              });
            }}
          />
        </div>
      ))}
    </div>
  );
}

export default function ConfigPanel({ config, onChange, disabled = false }: Props) {
  return (
    <div className="panel">
      <h3>Configuration</h3>
      <div className={`tag ${config.template === "rnn" ? "tag-paused" : "tag-running"}`}
        style={{ marginBottom: 12 }}>
        {config.template}
      </div>
      {renderSection("Model", config.model, "model", config, onChange, disabled)}
      {renderSection("Training", config.training, "training", config, onChange, disabled)}
    </div>
  );
}
