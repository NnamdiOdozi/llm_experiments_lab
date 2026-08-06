import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ConfigPanel from "./ConfigPanel";
import { ExperimentConfig } from "../types";

const config: ExperimentConfig = {
  template: "transformer",
  model: { vocab_size: 65, block_size: 128, n_embd: 192 },
  training: { learning_rate: 0.0003 },
  inference: { max_new_tokens: 100, temperature: 0.8, decoding_mode: "sample" },
};

describe("ConfigPanel numeric fields", () => {
  it("lets the user type a decimal point without it being erased", () => {
    // Real bug, 2026-07-15: onChange fed Number(e.target.value) straight
    // back into the controlled `value` — Number("0.") is 0, so typing "0"
    // then "." snapped the displayed value back to "0" immediately,
    // making it impossible to ever type a decimal. See
    // docs/DESIGN_DECISIONS.md.
    const onChange = vi.fn();
    render(<ConfigPanel config={config} onChange={onChange} />);

    const temperature = screen.getByDisplayValue("0.8") as HTMLInputElement;
    fireEvent.change(temperature, { target: { value: "0." } });
    expect(temperature.value).toBe("0.");

    fireEvent.change(temperature, { target: { value: "0.5" } });
    expect(temperature.value).toBe("0.5");
    expect(onChange).toHaveBeenLastCalledWith({
      ...config,
      inference: { ...config.inference, temperature: 0.5 },
    });
  });

  it("shows a config error when given one", () => {
    render(<ConfigPanel config={config} onChange={() => {}} error="max_new_tokens (150) cannot exceed block_size (128)" />);
    expect(screen.getByText(/cannot exceed block_size/)).toBeInTheDocument();
  });

  it("uses the compact configuration control treatment", () => {
    render(<ConfigPanel config={config} onChange={() => {}} />);
    expect(screen.getByText("Configuration").closest(".config-panel")).toBeInTheDocument();
    expect(screen.getByDisplayValue("128")).toHaveClass("config-field__control");
    expect(screen.getByText("block_size")).toHaveClass("config-field__label");
  });

  it("keeps tokenizer-derived vocab_size out of the user configuration form", () => {
    render(<ConfigPanel config={config} onChange={() => {}} />);
    expect(screen.queryByText("vocab_size")).not.toBeInTheDocument();
    expect(screen.queryByDisplayValue("65")).not.toBeInTheDocument();
  });
});
