import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import Inspector from "./Inspector";
import type { ArchitectureNode, DiagnosticSnapshot } from "../types";

// Eager attention_maps (DESIGN_DECISIONS §76): the heatmap must re-render for
// a different block/head purely from the snapshot already in memory — no
// network. These tests drive Inspector directly with prop changes and a
// fetch spy. Known limitation: the peek-SKIP condition itself lives in
// App.tsx's effect and is not exercised here (App is too heavy to render in
// isolation); what IS pinned down is that Inspector needs no fetch to show
// any (block, head) pair from maps, which is what makes App's skip safe.

const attentionNode = (block: number): ArchitectureNode => ({
  id: `block.${block}.attention`,
  kind: "attention",
  label: "Causal Self-Attention",
  config: {},
});

// Marker weights: cell text is weight.toFixed(2); row1col0 is unmasked and
// unique per (layer, head) so assertions are unambiguous.
const marker = (m: number) => [
  [1.0, 0.0],
  [m, 1 - m],
];

const SNAPSHOT_WITH_MAPS: DiagnosticSnapshot = {
  schema_version: 1,
  diagnostic_session_id: "diag-test",
  generation_step: 1,
  input_tokens: [{ position: 0, id: 1, text: "a" }],
  generated_token: { position: 1, id: 2, text: "b" },
  nodes: {},
  // Deliberately stale single-pair data (0.99 marker): with maps present the
  // heatmap must IGNORE this and read maps, else block switching goes stale.
  attention: {
    available: true,
    layer: 0,
    head: 0,
    weights: marker(0.99),
    token_labels: ["a", "b"],
    window_start: 0,
    total_positions: 2,
  },
  activation_summaries: { available: false, reason: "Not requested" },
  lm_head: { logits_shape: [1, 2, 65], selected_position: 1, top_k: [], top_k_by_position: [] },
  position_tokens: [],
  attention_maps: {
    available: true,
    window_start: 0,
    total_positions: 2,
    token_labels: ["a", "b"],
    positions: [0, 1],
    n_layer: 2,
    n_head: 2,
    weights: [
      [marker(0.11), marker(0.33)], // layer 0, heads 0/1
      [marker(0.77), marker(0.88)], // layer 1, heads 0/1
    ],
    qkv: [
      // Layer 0
      [
        // Head 0: q starts with 0.41
        { q: [[0.41, 0.42], [0.43, 0.44]], k: [[0.45, 0.46], [0.47, 0.48]], v: [[0.49, 0.50], [0.51, 0.52]] },
        // Head 1: q starts with 0.43
        { q: [[0.43, 0.44], [0.45, 0.46]], k: [[0.47, 0.48], [0.49, 0.50]], v: [[0.51, 0.52], [0.53, 0.54]] },
      ],
      // Layer 1
      [
        // Head 0: q starts with 0.61
        { q: [[0.61, 0.62], [0.63, 0.64]], k: [[0.65, 0.66], [0.67, 0.68]], v: [[0.69, 0.70], [0.71, 0.72]] },
        // Head 1: q starts with 0.63
        { q: [[0.63, 0.64], [0.65, 0.66]], k: [[0.67, 0.68], [0.69, 0.70]], v: [[0.71, 0.72], [0.73, 0.74]] },
      ],
    ],
  },
  // Full-context matrix for the canvas heatmap (one selected layer/head). The
  // canvas reads THIS, not attention_maps — see AttentionHeatmapCanvas.
  attention_full: {
    available: true,
    layer: 0,
    head: 0,
    weights: marker(0.11),
    token_labels: ["a", "b"],
    total_positions: 2,
    block_size: 128,
  },
  complete: true,
};

// No full matrix (e.g. an older backend that never populates attention_full) —
// the canvas heatmap should show "Not captured". attention_maps/attention are
// still present for the (unrelated) Q/K/V path.
const SNAPSHOT_NO_MAPS: DiagnosticSnapshot = {
  ...SNAPSHOT_WITH_MAPS,
  attention_maps: undefined,
  attention_full: undefined,
};

const SNAPSHOT_NOTHING_CAPTURED: DiagnosticSnapshot = {
  ...SNAPSHOT_WITH_MAPS,
  attention_maps: undefined,
  attention_full: undefined,
  attention: { available: false, reason: "Not requested" },
};

// Snapshot with maps (weights only, no qkv) and fallback single-pair qkv_detail.
// This pins the behavior: when maps lack qkv, the single-pair qkv_detail must render.
const SNAPSHOT_MAPS_NO_QKV: DiagnosticSnapshot = {
  ...SNAPSHOT_WITH_MAPS,
  attention_maps: {
    ...SNAPSHOT_WITH_MAPS.attention_maps!,
    qkv: undefined, // maps have weights but NOT qkv
  },
  attention: {
    available: true,
    layer: 0,
    head: 0,
    weights: marker(0.99),
    token_labels: ["a", "b"],
    window_start: 0,
    total_positions: 2,
    qkv_detail: {
      // Distinct marker for fallback: q starts with 0.91
      positions: [0, 1],
      tokens: ["a", "b"],
      q: [[0.91, 0.92], [0.93, 0.94]],
      k: [[0.95, 0.96], [0.97, 0.98]],
      v: [[0.99, 0.00], [0.01, 0.02]],
    },
  },
};

function renderInspector(block: number, head: number, snapshot: DiagnosticSnapshot, showQKVDetail: boolean = false) {
  return render(
    <Inspector
      runId={1}
      selectedNode={attentionNode(block)}
      selectedNodeId={`block.${block}.attention`}
      diagnosticSnapshot={snapshot}
      currentStep={snapshot.generation_step}
      isLoading={false}
      attentionHead={head}
      onAttentionHeadChange={() => {}}
      showQKVDetail={showQKVDetail}
      onShowQKVDetailChange={() => {}}
      qkvWindowOffset={0}
      onQkvWindowOffsetChange={() => {}}
      nodeWindowOffset={0}
      onNodeWindowOffsetChange={() => {}}
      numHeads={2}
      onOpenDataTab={() => {}}
      activeTab="runtime"
      onActiveTabChange={() => {}}
    />
  );
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("Inspector heatmap with eager attention_maps", () => {
  // The heatmap is now an HTML <canvas> fed by snapshot.attention_full (the
  // full T×T matrix for one selected layer/head), not the old numeric table
  // read from attention_maps. So we assert on the canvas + header, and that no
  // per-cell numbers are printed. Fetching to obtain attention_full lives in
  // App.tsx's peek effect (App is too heavy to render here); Inspector itself
  // just displays whatever attention_full the snapshot carries.
  it("renders the canvas heatmap (no per-cell numbers) when attention_full is available", () => {
    const { container } = renderInspector(0, 0, SNAPSHOT_WITH_MAPS);
    expect(container.querySelector("canvas")).toBeTruthy();
    expect(screen.getByText(/Layer 1, Head 1/)).toBeInTheDocument();
    // Canvas paints cells; it must NOT print the numeric weight as DOM text.
    expect(screen.queryByText("0.11")).not.toBeInTheDocument();
  });

  it("heatmap header reflects the selected block/head", () => {
    renderInspector(1, 1, SNAPSHOT_WITH_MAPS);
    expect(screen.getByText(/Layer 2, Head 2/)).toBeInTheDocument();
  });

  it("heatmap shows 'Not captured' when attention_full is absent (e.g. old backend)", () => {
    const { container } = renderInspector(0, 0, SNAPSHOT_NO_MAPS);
    expect(container.querySelector("canvas")).toBeNull();
    expect(screen.getByText(/Not captured/)).toBeInTheDocument();
  });

  it("shows 'Not captured' without crashing when nothing is captured", () => {
    // Regression: this branch used to read `att.reason` off an unassigned
    // variable and throw a TypeError before first capture.
    renderInspector(0, 0, SNAPSHOT_NOTHING_CAPTURED);
    expect(screen.getByText(/Not captured/)).toBeInTheDocument();
  });

  it("Q/K/V table renders from attention_maps.qkv with showQKVDetail=true, no fetches", () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    renderInspector(0, 0, SNAPSHOT_WITH_MAPS, true);
    // Layer 0, Head 0: q starts with 0.41 → renders as "[0.41, ..."
    expect(screen.getByText(/0\.41/)).toBeInTheDocument();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("Q/K/V block change with showQKVDetail=true uses maps, zero fetches", () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const { unmount } = renderInspector(0, 0, SNAPSHOT_WITH_MAPS, true);
    expect(screen.getByText(/0\.41/)).toBeInTheDocument();
    unmount();
    // Change block to 1, head 0: q starts with 0.61
    renderInspector(1, 0, SNAPSHOT_WITH_MAPS, true);
    expect(screen.getByText(/0\.61/)).toBeInTheDocument();
    expect(screen.queryByText(/0\.41/)).not.toBeInTheDocument();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("Q/K/V fallback: when maps.qkv absent, renders snapshot.attention.qkv_detail", () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    // SNAPSHOT_MAPS_NO_QKV has maps but no qkv; its attention has qkv_detail (0.91 marker)
    renderInspector(0, 0, SNAPSHOT_MAPS_NO_QKV, true);
    // Should render the fallback qkv_detail with the 0.91 marker, not the heatmap
    expect(screen.getByText(/0\.91/)).toBeInTheDocument();
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});
