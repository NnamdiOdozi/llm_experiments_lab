import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import ArchSchematic from "./ArchSchematic";

// Real bug report, 2026-07-14: the numbered block buttons only ever set
// ArchSchematic's local selectedBlockIdx — App's selectedNodeId (which
// drives the Inspector, the peek effect, and the staleness warning) never
// changed, so switching block silently did nothing in the Runtime
// inspector. These tests pin the fix: with a block-child node selected,
// picking a different block must remap the selection to that block.
// See docs/DESIGN_DECISIONS.md.

const manifest = {
  schema_version: 1,
  local_run_id: 1,
  template: "transformer",
  param_count: 1000,
  trainable_param_count: 1000,
  nodes: [
    { id: "embedding", kind: "embedding", label: "Token + Positional Embedding", config: {} },
    {
      id: "block",
      kind: "transformer_block_group",
      label: "Transformer Block",
      repeat_count: 4,
      config: {},
      children: [
        { id: "block.{i}.ln1", kind: "layernorm", label: "LayerNorm (pre-attention)", config: {} },
        { id: "block.{i}.attention", kind: "attention", label: "Causal Self-Attention", config: {} },
        { id: "block.{i}.ln2", kind: "layernorm", label: "LayerNorm (pre-MLP)", config: {} },
        { id: "block.{i}.mlp", kind: "mlp", label: "Feed-Forward (dense)", config: {} },
      ],
    },
    { id: "lm_head", kind: "lm_head", label: "LM Head", config: {} },
  ],
};

vi.mock("../hooks/useApi", () => ({
  fetchArchitecture: vi.fn(() => Promise.resolve(manifest)),
}));

describe("ArchSchematic block selector", () => {
  it("remaps a selected block-child node to the newly picked block", async () => {
    const onNodeClick = vi.fn();
    render(
      <ArchSchematic runId={1} onNodeClick={onNodeClick} selectedNodeId="block.0.attention" />,
    );
    await waitFor(() => expect(screen.getByText("Block 1 of 4")).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: "3" }));

    expect(onNodeClick).toHaveBeenCalledWith(
      "block.2.attention",
      expect.objectContaining({ id: "block.{i}.attention", kind: "attention" }),
    );
  });

  it("does not fire onNodeClick when no block-child node is selected", async () => {
    const onNodeClick = vi.fn();
    render(<ArchSchematic runId={1} onNodeClick={onNodeClick} selectedNodeId="embedding" />);
    await waitFor(() => expect(screen.getByText("Block 1 of 4")).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: "2" }));

    expect(onNodeClick).not.toHaveBeenCalled();
  });
});
