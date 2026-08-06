import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ConfigPanel from "./ConfigPanel";
import { ExperimentConfig } from "../types";

const config: ExperimentConfig = {
  template: "transformer",
  model: { vocab_size: 65, block_size: 128, n_embd: 192 },
  training: { learning_rate: 0.0003 },
  inference: { max_new_tokens: 100, temperature: 0.8, decoding_mode: "sample" },
  data: {
    dataset: "tiny_shakespeare",
    tokenizer: "char",
    tokenizer_artifact: null,
    vocab_size: 65,
  },
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

  it("uses the denser treatment only for the longer MoE configuration", () => {
    const { rerender } = render(<ConfigPanel config={{ ...config, template: "moe" }} onChange={() => {}} />);
    expect(screen.getByText("Configuration").closest(".config-panel")).toHaveClass("config-panel--moe");

    rerender(<ConfigPanel config={config} onChange={() => {}} />);
    expect(screen.getByText("Configuration").closest(".config-panel")).not.toHaveClass("config-panel--moe");
  });

  it("keeps tokenizer-derived vocab_size out of the model editable fields", () => {
    render(<ConfigPanel config={config} onChange={() => {}} />);
    // vocab_size should not appear as an editable input in the Model section
    const vocabInputs = screen.queryAllByDisplayValue("65");
    // The read-only vocab_size in Data section is displayed as text, not an input
    vocabInputs.forEach(input => {
      expect((input as HTMLInputElement).classList.contains("config-field__control")).toBe(false);
    });
  });

  it("displays vocab_size as read-only in the Data section", () => {
    render(<ConfigPanel config={config} onChange={() => {}} />);
    expect(screen.getByText("65", { selector: ".config-field__readonly" })).toBeInTheDocument();
    expect(screen.queryByText("fixed by tokenizer")).not.toBeInTheDocument();
  });

  it("shows the Tokenizer dropdown for transformer template", () => {
    render(<ConfigPanel config={config} onChange={() => {}} />);
    const tokenizer = screen.getByDisplayValue("Character — 65 tokens") as HTMLSelectElement;
    expect(tokenizer).toBeInTheDocument();
  });

  it("hides the Tokenizer dropdown for RNN template", () => {
    const rnnConfig: ExperimentConfig = {
      ...config,
      template: "rnn",
    };
    render(<ConfigPanel config={rnnConfig} onChange={() => {}} />);
    expect(screen.queryByDisplayValue("Character — 65 tokens")).not.toBeInTheDocument();
    // But vocab_size should still be displayed (read-only)
    expect(screen.getByText("65", { selector: ".config-field__readonly" })).toBeInTheDocument();
  });

  it("updates vocab_size when tokenizer changes to BPE Small", () => {
    const onChange = vi.fn();
    render(<ConfigPanel config={config} onChange={onChange} />);

    const tokenizer = screen.getByDisplayValue("Character — 65 tokens") as HTMLSelectElement;
    fireEvent.change(tokenizer, { target: { value: "bpe_1k" } });

    expect(onChange).toHaveBeenCalledWith({
      ...config,
      data: {
        dataset: "tiny_shakespeare",
        tokenizer: "bpe_1k",
        tokenizer_artifact: "tiny-shakespeare-bpe-1k-v1.json",
        vocab_size: 1024,
      },
      model: { ...config.model, vocab_size: 1024 },
    });
  });

  it("updates vocab_size when tokenizer changes to BPE Medium", () => {
    const onChange = vi.fn();
    render(<ConfigPanel config={config} onChange={onChange} />);

    const tokenizer = screen.getByDisplayValue("Character — 65 tokens") as HTMLSelectElement;
    fireEvent.change(tokenizer, { target: { value: "bpe_4k" } });

    expect(onChange).toHaveBeenCalledWith({
      ...config,
      data: {
        dataset: "tiny_shakespeare",
        tokenizer: "bpe_4k",
        tokenizer_artifact: "tiny-shakespeare-bpe-4k-v1.json",
        vocab_size: 4096,
      },
      model: { ...config.model, vocab_size: 4096 },
    });
  });

  it("displays the correct vocab_size label when tokenizer is selected", () => {
    const onChange = vi.fn();
    const { rerender } = render(<ConfigPanel config={config} onChange={onChange} />);

    // Initially char (65)
    expect(screen.getByText("65", { selector: ".config-field__readonly" })).toBeInTheDocument();

    // Change to bpe_1k (1024)
    const tokenizer = screen.getByDisplayValue("Character — 65 tokens") as HTMLSelectElement;
    fireEvent.change(tokenizer, { target: { value: "bpe_1k" } });

    const updatedConfig = onChange.mock.calls[0][0];
    rerender(<ConfigPanel config={updatedConfig} onChange={onChange} />);
    expect(screen.getByText("1024", { selector: ".config-field__readonly" })).toBeInTheDocument();
  });

  it("defaults to char tokenizer when config.data is undefined", () => {
    const configWithoutData: ExperimentConfig = {
      template: "transformer",
      model: { vocab_size: 65, block_size: 128, n_embd: 192 },
      training: { learning_rate: 0.0003 },
    };
    render(<ConfigPanel config={configWithoutData} onChange={() => {}} />);
    expect(screen.getByDisplayValue("Character — 65 tokens")).toBeInTheDocument();
    expect(screen.getByText("65", { selector: ".config-field__readonly" })).toBeInTheDocument();
  });

  it("disables the tokenizer dropdown when panel is disabled", () => {
    render(<ConfigPanel config={config} onChange={() => {}} disabled={true} />);
    const tokenizer = screen.getByDisplayValue("Character — 65 tokens") as HTMLSelectElement;
    expect(tokenizer.disabled).toBe(true);
  });
});
