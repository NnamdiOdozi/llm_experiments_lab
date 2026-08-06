import { useState, useEffect } from "react";
import { ExperimentConfig } from "../types";
import "./ConfigPanel.css";

// Static vocab_size mappings for each tokenizer
const TOKENIZER_VOCAB_SIZES: Record<string, number> = {
  char: 65,
  bpe_1k: 1024,
  bpe_4k: 4096,
};

// Tokenizer artifact mappings
const TOKENIZER_ARTIFACTS: Record<string, string | null> = {
  char: null,
  bpe_1k: "tiny-shakespeare-bpe-1k-v1.json",
  bpe_4k: "tiny-shakespeare-bpe-4k-v1.json",
};

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
  value, onChange, disabled, title,
}: {
  value: number | string;
  onChange: (v: number) => void;
  disabled: boolean;
  title?: string;
}) {
  const [text, setText] = useState(String(value));
  useEffect(() => {
    if (Number(text) !== value) setText(String(value));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);
  return (
    <input
      className="config-field__control"
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
    <div className="config-section">
      <h4 className="config-section__title">
        {title}
      </h4>
      {/* Temperature has no effect under greedy decoding — argmax(logits/T)
          is the same index as argmax(logits) for any positive T, since
          scaling preserves order. Greying it out here says so directly
          instead of leaving it live-but-inert. See docs/DESIGN_DECISIONS.md. */}
      {(() => {
        const isGreedy = sectionKey === "inference" && section.decoding_mode === "greedy";
        return Object.entries(section).filter(([key]) => key !== "vocab_size").map(([key, val]) => {
        const options = DROPDOWN_FIELDS[key];
        const baselineVal = baselineSection?.[key];
        const changedFromBaseline = baselineVal != null && String(baselineVal) !== String(val);
        const fieldDisabled = disabled || (key === "temperature" && isGreedy);
        const handleChange = (newVal: string | number) => {
          onChange({
            ...config,
            [sectionKey]: { ...config[sectionKey], [key]: newVal },
          });
        };
        return (
          <div key={key} className="config-field">
          <div
            className="config-field__row"
          >
            <label className="config-field__label">{key}</label>
            {options ? (
              <select
                className="config-field__control"
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
                value={val}
                disabled={fieldDisabled}
                title={
                  key === "temperature" && isGreedy
                      ? "No effect under greedy decoding"
                      : undefined
                }
                onChange={handleChange}
              />
            )}
          </div>
          {changedFromBaseline && (
            <div className="config-field__hint">
              baseline: {String(baselineVal)}
            </div>
          )}
          {key === "temperature" && isGreedy && (
            <div className="config-field__hint">
              no effect under greedy
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

// Default data section for configs that don't have it yet
const DATA_DEFAULTS = {
  dataset: "tiny_shakespeare",
  tokenizer: "char",
  tokenizer_artifact: null,
  vocab_size: 65,
};

export default function ConfigPanel({ config, onChange, disabled = false, baseline = null, error = null }: Props) {
  // Normalize inference section so onChange always has all fields,
  // even for experiments created before inference config existed.
  const normalizedConfig: ExperimentConfig = {
    ...config,
    inference: { ...INFERENCE_DEFAULTS, ...config.inference },
    data: { ...DATA_DEFAULTS, ...config.data },
  };

  const handleTokenizerChange = (tokenizer: string) => {
    const newVocabSize = TOKENIZER_VOCAB_SIZES[tokenizer];
    const newArtifact = TOKENIZER_ARTIFACTS[tokenizer];
    const updatedData = {
      ...normalizedConfig.data!,
      tokenizer,
      vocab_size: newVocabSize,
      tokenizer_artifact: newArtifact,
    };
    const updatedConfig = {
      ...normalizedConfig,
      data: updatedData,
      model: { ...normalizedConfig.model, vocab_size: newVocabSize },
    };
    onChange(updatedConfig);
  };

  return (
    <div className="panel config-panel">
      <h3>Configuration</h3>
      <div className={`tag ${normalizedConfig.template === "rnn" ? "tag-paused" : "tag-running"}`}
        style={{ marginBottom: 8 }}>
        {normalizedConfig.template}
      </div>
      {error && (
        <div className="config-panel__error">
          {error}
        </div>
      )}
      {/* Data section: tokenizer selection and read-only vocab_size */}
      <div className="config-section">
        <h4 className="config-section__title">Data</h4>
        {normalizedConfig.template !== "rnn" && (
          <div className="config-field">
            <div className="config-field__row">
              <label className="config-field__label">Tokenizer</label>
              <select
                className="config-field__control"
                value={normalizedConfig.data!.tokenizer}
                disabled={disabled}
                onChange={(e) => handleTokenizerChange(e.target.value)}
              >
                <option value="char">Character — 65 tokens</option>
                <option value="bpe_1k">BPE Small — 1,024 tokens</option>
                <option value="bpe_4k">BPE Medium — 4,096 tokens</option>
              </select>
            </div>
          </div>
        )}
        <div className="config-field">
          <div className="config-field__row">
            <label className="config-field__label">vocab_size</label>
            <div className="config-field__readonly">
              {normalizedConfig.data!.vocab_size}
            </div>
          </div>
          <div className="config-field__hint">
            fixed by tokenizer
          </div>
        </div>
      </div>
      {renderSection("Model", normalizedConfig.model, "model", normalizedConfig, onChange, disabled, baseline?.model)}
      {renderSection("Training", normalizedConfig.training, "training", normalizedConfig, onChange, disabled, baseline?.training)}
      {/* Inference section controls generation params (temperature, max_new_tokens,
          decoding_mode) used when prompting a paused model. These don't affect the
          model's weights/shape, so they stay editable even during an active run —
          the user may tweak them to prompt a paused model. Pass false for the
          run-lock; the intrinsic greedy-temperature rule still applies internally.
          Direct user request, 2026-08-06. */}
      {renderSection("Inference", normalizedConfig.inference!, "inference", normalizedConfig, onChange, false)}
    </div>
  );
}
