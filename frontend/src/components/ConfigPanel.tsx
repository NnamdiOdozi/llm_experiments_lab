import { ExperimentConfig } from "../types";

interface Props {
  config: ExperimentConfig;
  onChange: (config: ExperimentConfig) => void;
  disabled?: boolean;
}

const DROPDOWN_FIELDS: Record<string, string[]> = {
  pos_encoding: ["learned", "rope"],
  optimizer: ["adam", "adamw", "sgd"],
  activation: ["gelu", "relu", "silu"],
};

function renderSection(
  title: string,
  section: Record<string, number | string>,
  sectionKey: "model" | "training" | "inference",
  config: ExperimentConfig,
  onChange: Props["onChange"],
  disabled: boolean
) {
  return (
    <div style={{ marginBottom: 16 }}>
      <h4 style={{ fontSize: 12, color: "var(--accent)", marginBottom: 8 }}>
        {title}
      </h4>
      {Object.entries(section).map(([key, val]) => {
        const options = DROPDOWN_FIELDS[key];
        const handleChange = (newVal: string | number) => {
          onChange({
            ...config,
            [sectionKey]: { ...config[sectionKey], [key]: newVal },
          });
        };
        return (
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
            {options ? (
              <select
                style={{ width: 100, textAlign: "right", fontSize: 12, padding: "4px 8px" }}
                value={String(val)}
                disabled={disabled}
                onChange={(e) => handleChange(e.target.value)}
              >
                {options.map((opt) => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
            ) : (
              <input
                style={{ width: 100, textAlign: "right" }}
                value={val}
                disabled={disabled}
                onChange={(e) => {
                  const v = e.target.value;
                  handleChange(isNaN(Number(v)) ? v : Number(v));
                }}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

const INFERENCE_DEFAULTS: Record<string, number> = { max_new_tokens: 100, temperature: 0.8 };

export default function ConfigPanel({ config, onChange, disabled = false }: Props) {
  // Normalize inference section so onChange always has all fields,
  // even for experiments created before inference config existed.
  const normalizedConfig: ExperimentConfig = {
    ...config,
    inference: { ...INFERENCE_DEFAULTS, ...config.inference },
  };

  return (
    <div className="panel">
      <h3>Configuration</h3>
      <div className={`tag ${normalizedConfig.template === "rnn" ? "tag-paused" : "tag-running"}`}
        style={{ marginBottom: 12 }}>
        {normalizedConfig.template}
      </div>
      {renderSection("Model", normalizedConfig.model, "model", normalizedConfig, onChange, disabled)}
      {renderSection("Training", normalizedConfig.training, "training", normalizedConfig, onChange, disabled)}
      {/* Inference section controls generation params (temperature, max_new_tokens)
          used when prompting a paused model from the dashboard. */}
      {renderSection("Inference", normalizedConfig.inference!, "inference", normalizedConfig, onChange, disabled)}
    </div>
  );
}
