import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import ArchSchematic from "./ArchSchematic";
import { fetchArchitecture, previewArchitecture } from "../hooks/useApi";

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
    { id: "final_norm", kind: "layernorm", label: "Final LayerNorm", config: {} },
    { id: "lm_head", kind: "lm_head", label: "LM Head", config: {} },
  ],
};

vi.mock("../hooks/useApi", () => ({
  fetchArchitecture: vi.fn(() => Promise.resolve(manifest)),
  previewArchitecture: vi.fn(() => Promise.resolve(manifest)),
}));

describe("ArchSchematic block selector", () => {
  it("remaps a selected block-child node to the newly picked block", async () => {
    const onNodeClick = vi.fn();
    render(
      <ArchSchematic runId={1} onNodeClick={onNodeClick} selectedNodeId="block.0.attention" />,
    );
    await waitFor(() => expect(screen.getByText(/Transformer\s+Block 1 of 4/)).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: "3" }));

    expect(onNodeClick).toHaveBeenCalledWith(
      "block.2.attention",
      expect.objectContaining({ id: "block.{i}.attention", kind: "attention" }),
    );
  });

  it("does not fire onNodeClick when no block-child node is selected", async () => {
    const onNodeClick = vi.fn();
    render(<ArchSchematic runId={1} onNodeClick={onNodeClick} selectedNodeId="embedding" />);
    await waitFor(() => expect(screen.getByText(/Transformer\s+Block 1 of 4/)).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: "2" }));

    expect(onNodeClick).not.toHaveBeenCalled();
  });

  it("uses concise diagram labels without changing the node passed to the inspector", async () => {
    const onNodeClick = vi.fn();
    render(<ArchSchematic runId={1} onNodeClick={onNodeClick} />);

    await waitFor(() => expect(screen.getByText("T + P Embedding")).toBeInTheDocument());
    expect(screen.getByText("Causal S. Attention")).toBeInTheDocument();
    expect(screen.getByText("Feed Forward")).toBeInTheDocument();
    expect(screen.getByText(/Final\s+LayerNorm/)).toBeInTheDocument();
    expect(screen.getAllByText("LayerNorm")).toHaveLength(2);

    fireEvent.click(screen.getByText("Causal S. Attention"));
    expect(onNodeClick).toHaveBeenCalledWith(
      "block.0.attention",
      expect.objectContaining({ label: "Causal Self-Attention" }),
    );
  });

  it("renders the selected block internals as a visually grouped detail region", async () => {
    const { container } = render(<ArchSchematic runId={1} />);

    await waitFor(() => expect(screen.getByText("Inside selected transformer block")).toBeInTheDocument());
    expect(container.querySelector(".arch-node--expanded")).toBeInTheDocument();
    expect(container.querySelector(".arch-block-detail")).toBeInTheDocument();
  });

  it("uses the same hierarchical top-row labels for an MoE transformer", async () => {
    const block = manifest.nodes[1];
    vi.mocked(fetchArchitecture).mockResolvedValueOnce({
      ...manifest,
      template: "moe",
      nodes: [
        manifest.nodes[0],
        {
          ...block,
          // Simulates an older remote MoE manifest whose presentation text
          // and kind differ from the current local producer. Stable node ids
          // must still receive the shared Transformer/MoE headings.
          kind: "moe_block_group",
          label: "MoE Blocks",
          children: [
            ...block.children!.slice(0, 3),
            { id: "block.{i}.moe", kind: "moe", label: "Mixture of Experts", config: {} },
          ],
        },
        { ...manifest.nodes[2], label: "Output normalization" },
        manifest.nodes[3],
      ],
    });

    render(<ArchSchematic runId={2} />);

    await waitFor(() => expect(screen.getByText(/Transformer\s+Block 1 of 4/)).toBeInTheDocument());
    expect(screen.getByText(/Final\s+LayerNorm/)).toBeInTheDocument();
    expect(screen.getByText("Experts")).toBeInTheDocument();
  });
});

// Direct user request, 2026-07-15: the diagram should render as soon as a
// preset is picked, before Start is clicked, and update as the user tweaks
// structural fields — not stay blank until a run exists. See
// docs/DESIGN_DECISIONS.md.
describe("ArchSchematic preview mode (no run yet)", () => {
  const previewConfig = { template: "transformer", model: { n_layer: 4 } } as any;

  beforeEach(() => {
    vi.useFakeTimers();
    vi.mocked(previewArchitecture).mockClear();
    vi.mocked(fetchArchitecture).mockClear();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders from config alone when there is no runId, via previewArchitecture not fetchArchitecture", async () => {
    render(<ArchSchematic config={previewConfig} />);

    await act(async () => {
      vi.advanceTimersByTime(400);
    });
    await act(async () => {
      await Promise.resolve();
    });

    expect(previewArchitecture).toHaveBeenCalledWith(previewConfig);
    expect(fetchArchitecture).not.toHaveBeenCalled();
    expect(screen.getByText(/Transformer\s+Block 1 of 4/)).toBeInTheDocument();
  });

  it("debounces rapid config changes into a single call", async () => {
    const { rerender } = render(
      <ArchSchematic config={{ template: "transformer", model: { n_layer: 4 } } as any} />,
    );
    rerender(<ArchSchematic config={{ template: "transformer", model: { n_layer: 5 } } as any} />);
    rerender(<ArchSchematic config={{ template: "transformer", model: { n_layer: 6 } } as any} />);

    await act(async () => {
      vi.advanceTimersByTime(399);
    });
    expect(previewArchitecture).not.toHaveBeenCalled();

    await act(async () => {
      vi.advanceTimersByTime(1);
    });
    await act(async () => {
      await Promise.resolve();
    });

    expect(previewArchitecture).toHaveBeenCalledTimes(1);
    expect(previewArchitecture).toHaveBeenCalledWith({ template: "transformer", model: { n_layer: 6 } });
  });

  it("does not call previewArchitecture once a runId exists", async () => {
    render(<ArchSchematic runId={1} config={previewConfig} />);

    await act(async () => {
      vi.advanceTimersByTime(400);
    });
    await act(async () => {
      await Promise.resolve();
    });

    expect(fetchArchitecture).toHaveBeenCalledWith(1);
    expect(previewArchitecture).not.toHaveBeenCalled();
  });
});

describe("ArchSchematic residual connection visualization", () => {
  it("renders residual arc overlay with two arcs and two plus markers for transformer blocks", async () => {
    const { container } = render(<ArchSchematic runId={1} />);

    await waitFor(() => expect(screen.getByText("Inside selected transformer block")).toBeInTheDocument());

    // Arcs overlay should exist.
    const overlay = container.querySelector(".arch-residual-overlay");
    expect(overlay).toBeInTheDocument();

    // Should have exactly 2 arc paths.
    const arcs = container.querySelectorAll('[data-testid="residual-arc"]');
    expect(arcs).toHaveLength(2);

    // Should have exactly 2 "+" markers (one at each rejoin point).
    const pluses = container.querySelectorAll('[data-testid="residual-plus"]');
    expect(pluses).toHaveLength(2);

    // Should have 2 branch dots at split points.
    const branchDots = container.querySelectorAll('[data-testid="residual-branch-dot"]');
    expect(branchDots).toHaveLength(2);

    // Addition points must remain readable at normal dashboard scale.
    const operatorCircles = container.querySelectorAll(".arch-residual-overlay__operator");
    expect(operatorCircles).toHaveLength(2);
    operatorCircles.forEach((operator) => expect(operator).toHaveAttribute("r", "11"));

    // The bypass is explicitly identified instead of relying on line shape alone.
    expect(container.querySelector(".arch-residual-overlay__label")).toHaveTextContent("RESIDUAL STREAM");
  });

  it("renders residual overlay with pointer-events:none so it does not block box clicks", async () => {
    const onNodeClick = vi.fn();
    const { container } = render(
      <ArchSchematic runId={1} onNodeClick={onNodeClick} />,
    );

    await waitFor(() => expect(screen.getByText("Inside selected transformer block")).toBeInTheDocument());

    const overlay = container.querySelector(".arch-residual-overlay");
    expect(overlay).toHaveStyle("pointerEvents: none");

    // Verify that clicking a box still works (existing functionality).
    fireEvent.click(screen.getByText("Causal S. Attention"));
    expect(onNodeClick).toHaveBeenCalledWith(
      "block.0.attention",
      expect.objectContaining({ kind: "attention" }),
    );
  });

  it("centers the first addition between the attention and second LayerNorm boxes", async () => {
    const rect = (left: number, width: number, top = 58, height = 50) => ({
      x: left,
      y: top,
      left,
      right: left + width,
      top,
      bottom: top + height,
      width,
      height,
      toJSON: () => ({}),
    } as DOMRect);

    const geometrySpy = vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
      if (this.classList.contains("arch-block-detail__flow")) return rect(0, 1000, 0, 120);
      if (this.classList.contains("arch-arrow--entry")) return rect(20, 60);
      if (this.classList.contains("arch-arrow--exit")) return rect(880, 60);
      if (this.classList.contains("arch-stage") && this.closest(".arch-flow-item--detail")) {
        const label = this.textContent?.trim();
        if (label === "Causal S. Attention") return rect(300, 150);
        if (label === "Feed Forward") return rect(700, 150);
        const detailStages = Array.from(document.querySelectorAll(".arch-flow-item--detail .arch-stage"));
        return rect(detailStages.indexOf(this) === 0 ? 100 : 500, 150);
      }
      return rect(0, 0, 0, 0);
    });

    try {
      const { container } = render(<ArchSchematic runId={1} />);
      await waitFor(() => expect(container.querySelector(".arch-residual-overlay__operator")).toHaveAttribute("cx", "475"));
    } finally {
      geometrySpy.mockRestore();
    }
  });

  it("preserves the connector's layout width when the residual overlay paints the visible arrow", async () => {
    render(<ArchSchematic runId={1} />);

    await waitFor(() => expect(screen.getByText("Causal S. Attention")).toBeInTheDocument());

    const attentionItem = screen.getByText("Causal S. Attention").closest(".arch-flow-item--detail");
    const reservedConnector = attentionItem?.querySelector(".arch-arrow");
    expect(reservedConnector).toHaveClass("arch-arrow--muted");
  });

  it("does not render residual arcs for non-transformer templates without attention/mlp", async () => {
    const rnnManifest = {
      schema_version: 1,
      local_run_id: 1,
      template: "rnn",
      param_count: 1000,
      trainable_param_count: 1000,
      nodes: [
        { id: "embedding", kind: "embedding", label: "Token + Positional Embedding", config: {} },
        {
          id: "block",
          kind: "transformer_block_group",
          label: "RNN Block",
          repeat_count: 2,
          config: {},
          // No children, or missing attention/mlp — graceful degrade.
          children: [
            { id: "block.{i}.rnn", kind: "rnn", label: "RNN", config: {} },
          ],
        },
        { id: "lm_head", kind: "lm_head", label: "LM Head", config: {} },
      ],
    };

    vi.mocked(fetchArchitecture).mockResolvedValueOnce(rnnManifest);
    const { container } = render(<ArchSchematic runId={2} />);

    await waitFor(() => expect(screen.getByText("Inside selected transformer block")).toBeInTheDocument());

    // No residual arcs should be present (missing attention/mlp boxes).
    const arcs = container.querySelectorAll('[data-testid="residual-arc"]');
    expect(arcs).toHaveLength(0);
  });
});
