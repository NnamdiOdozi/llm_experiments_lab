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
    n_layer: 2,
    n_head: 2,
    weights: [
      [marker(0.11), marker(0.33)], // layer 0, heads 0/1
      [marker(0.77), marker(0.88)], // layer 1, heads 0/1
    ],
  },
  complete: true,
};

const SNAPSHOT_NO_MAPS: DiagnosticSnapshot = {
  ...SNAPSHOT_WITH_MAPS,
  attention_maps: undefined,
};

const SNAPSHOT_NOTHING_CAPTURED: DiagnosticSnapshot = {
  ...SNAPSHOT_WITH_MAPS,
  attention_maps: undefined,
  attention: { available: false, reason: "Not requested" },
};

function renderInspector(block: number, head: number, snapshot: DiagnosticSnapshot) {
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
      showQKVDetail={false}
      onShowQKVDetailChange={() => {}}
      attentionWindowOffset={0}
      nodeWindowOffset={0}
      onAttentionWindowOffsetChange={() => {}}
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
  it("renders the selected pair from attention_maps, ignoring stale single-pair data", () => {
    renderInspector(0, 0, SNAPSHOT_WITH_MAPS);
    expect(screen.getByText("0.11")).toBeInTheDocument();
    // The stale snapshot.attention marker must not be shown.
    expect(screen.queryByText("0.99")).not.toBeInTheDocument();
  });

  it("block change re-renders from maps with the SAME snapshot and zero fetches", () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const { unmount } = renderInspector(0, 0, SNAPSHOT_WITH_MAPS);
    expect(screen.getByText("0.11")).toBeInTheDocument();
    unmount();
    // Same snapshot object, only the selected block prop changes — the new
    // pair's weights must appear without any network round-trip.
    renderInspector(1, 0, SNAPSHOT_WITH_MAPS);
    expect(screen.getByText("0.77")).toBeInTheDocument();
    expect(screen.queryByText("0.11")).not.toBeInTheDocument();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("head change re-renders from maps with zero fetches", () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    renderInspector(0, 1, SNAPSHOT_WITH_MAPS);
    expect(screen.getByText("0.33")).toBeInTheDocument();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("falls back to single-pair snapshot.attention when maps are absent (old backend)", () => {
    renderInspector(0, 0, SNAPSHOT_NO_MAPS);
    expect(screen.getByText("0.99")).toBeInTheDocument();
  });

  it("shows 'Not captured' without crashing when neither maps nor single-pair exist", () => {
    // Regression: this branch used to read `att.reason` off an unassigned
    // variable and throw a TypeError before first capture.
    renderInspector(0, 0, SNAPSHOT_NOTHING_CAPTURED);
    expect(screen.getByText(/Not captured/)).toBeInTheDocument();
  });
});
