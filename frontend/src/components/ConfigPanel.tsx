import { useState, useEffect, CSSProperties } from "react";
import { ExperimentConfig } from "../types";

interface Props {
  config: ExperimentConfig;
  onChange: (config: ExperimentConfig) => void;
  disabled?: boolean;
  // Preset this experiment was created from — used to show "baseline: X"
  // shadow text under any field the user has changed. Null if unresolved yet.
  baseline?: ExperimentConfig | null;
  // Surfaces a rejected config PATCH (e.g. max_new_tokens > block_size) —
  // previously failed completely silently. See docs/DESIGN_DECISIONS.md.
  error?: string | null;
}

const DROPDOWN_FIELDS: Record<string, string[]> = {
  pos_encoding: ["learned", "rope"],
  optimizer: ["adam", "adamw", "sgd"],
  activation: ["gelu", "relu", "silu"],
  decoding_mode: ["sample", "greedy"],
};

// Determined by the dataset (character-level vocab), not a real training
// choice — editing it doesn't do anything useful and just invites
// confusion. Shown for information only. Direct user request, 2026-07-13.
// See docs/DESIGN_DECISIONS.md.
const READ_ONLY_FIELDS = new Set(["vocab_size"]);

// Real bug, 2026-07-15: a plain controlled <input> feeding straight
// Number(e.target.value) back into `value` snapped decimals back to an
// integer mid-typing — Number("0.") is 0, so typing "0" then "." erased
// the "." the instant it appeared, making it impossible to type any
// decimal at all (temperature, dropout, learning_rate, capacity_factor —
// every numeric field here, not just temperature). Fix: buffer the raw
// text locally, only re-sync from the external numeric value when it
// changes for a reason OTHER than this field's own typing (guarded by
// comparing Number(text) against value, so our own onChange round-trip
// doesn't clobber what's still being typed). See docs/DESIGN_DECISIONS.md.
function NumericField({
  value, onChange, disabled, title, style,
}: {
  value: number | string;
  onChange: (v: number) => void;
  disabled: boolean;
  title?: string;
  style: CSSProperties;
}) {
  const [text, setText] = useState(String(value));
  useEffect(() => {
    if (Number(text) !== value) setText(String(value));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);
  return (
    <input
      style={style}
      value={text}
      disabled={disabled}
      title={title}
      onChange={(e) => {
        const v = e.target.value;
        setText(v);
        if (v !== "" && v !== "-" && !isNaN(Number(v))) onChange(Number(v));
      }}
    />
  );
}

function renderSection(
  title: string,
  section: Record<string, number | string>,
  sectionKey: "model" | "training" | "inference",
  config: ExperimentConfig,
  onChange: Props["onChange"],
  disabled: boolean,
  baselineSection?: Record<string, number | string>
) {
  return (
    <div style={{ marginBottom: 16 }}>
      <h4 style={{ fontSize: 12, color: "var(--accent)", marginBottom: 8 }}>
        {title}
      </h4>
      {/* Temperature has no effect under greedy decoding — argmax(logits/T)
          is the same index as argmax(logits) for any positive T, since
          scaling preserves order. Greying it out here says so directly
          instead of leaving it live-but-inert. See docs/DESIGN_DECISIONS.md. */}
      {(() => {
        const isGreedy = sectionKey === "inference" && section.decoding_mode === "greedy";
        return Object.entries(section).map(([key, val]) => {
        const options = DROPDOWN_FIELDS[key];
        const baselineVal = baselineSection?.[key];
        const changedFromBaseline = baselineVal != null && String(baselineVal) !== String(val);
        const isReadOnly = READ_ONLY_FIELDS.has(key);
        const fieldDisabled = disabled || (key === "temperature" && isGreedy) || isReadOnly;
        const handleChange = (newVal: string | number) => {
          onChange({
            ...config,
            [sectionKey]: { ...config[sectionKey], [key]: newVal },
          });
        };
        return (
          <div key={key} style={{ marginBottom: 6 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
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
              <NumericField
                style={{ width: 100, textAlign: "right" }}
                value={val}
                disabled={fieldDisabled}
                title={
                  isReadOnly
                    ? "Determined by the dataset — not editable"
                    : key === "temperature" && isGreedy
                      ? "No effect under greedy decoding"
                      : undefined
                }
                onChange={handleChange}
              />
            )}
          </div>
          {changedFromBaseline && (
            <div style={{ textAlign: "right", fontSize: 10, color: "var(--text-dim)", opacity: 0.6 }}>
              baseline: {String(baselineVal)}
            </div>
          )}
          {key === "temperature" && isGreedy && (
            <div style={{ textAlign: "right", fontSize: 10, color: "var(--text-dim)", opacity: 0.6 }}>
              no effect under greedy
            </div>
          )}
          {isReadOnly && (
            <div style={{ textAlign: "right", fontSize: 10, color: "var(--text-dim)", opacity: 0.6 }}>
              fixed by dataset
            </div>
          )}
          </div>
        );
        });
      })()}
    </div>
  );
}

const INFERENCE_DEFAULTS: Record<string, number | string> = {
  max_new_tokens: 100,
  temperature: 0.8,
  // Same setting used everywhere decoding happens — Generate button and
  // step-through > / >>. See docs/DESIGN_DECISIONS.md.
  decoding_mode: "sample",
};

export default function ConfigPanel({ config, onChange, disabled = false, baseline = null, error = null }: Props) {
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
      {error && (
        <div style={{ background: "var(--red, #e53e3e)", color: "#fff", padding: "6px 12px", borderRadius: 4, fontSize: 12, marginBottom: 12 }}>
          {error}
        </div>
      )}
      {renderSection("Model", normalizedConfig.model, "model", normalizedConfig, onChange, disabled, baseline?.model)}
      {renderSection("Training", normalizedConfig.training, "training", normalizedConfig, onChange, disabled, baseline?.training)}
      {/* Inference section controls generation params (temperature, max_new_tokens)
          used when prompting a paused model from the dashboard. */}
      {renderSection("Inference", normalizedConfig.inference!, "inference", normalizedConfig, onChange, disabled)}
    </div>
  );
}
