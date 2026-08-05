import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { useState } from "react";
import PausePrompt from "../components/PausePrompt";
import Inspector, { SubTab } from "../components/Inspector";
import * as api from "../hooks/useApi";
import type { ArchitectureNode, DiagnosticSnapshot, DiagnosticSessionResponse } from "../types";

// Real bug report, 2026-07-14: "I press Step, output updates, but the
// Inspector pane does not snap — feels like I need to click again or
// refresh." Individual component tests wouldn't catch a wiring bug
// between PausePrompt and Inspector (they only exist together in
// App.tsx), so this wires them together exactly the way App.tsx does —
// same shared diagnosticSnapshot state, same onDiagnosticSnapshot
// callback pattern — with only the network layer mocked, to check for a
// real integration bug rather than guessing further. See
// docs/DESIGN_DECISIONS.md.

const EMBEDDING_NODE: ArchitectureNode = {
  id: "embedding",
  kind: "embedding",
  label: "Token + Positional Embedding",
  config: {},
};

function Harness() {
  const [selectedNodeId] = useState<string | null>("embedding");
  const [diagnosticSnapshot, setDiagnosticSnapshot] = useState<DiagnosticSnapshot | null>(null);
  const [diagnosticLoading, setDiagnosticLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<SubTab>("overview");

  return (
    <div>
      <PausePrompt
        runId={1}
        canPrompt={true}
        attentionBlock={null}
        attentionHead={null}
        showQKVDetail={false}
        attentionWindowOffset={0}
        nodeWindowOffset={0}
        maxNewTokens={100}
        temperature={0.8}
        decodingMode="sample"
        onDiagnosticSnapshot={(snapshot) => {
          setDiagnosticSnapshot(snapshot);
          setDiagnosticLoading(false);
        }}
        onSessionIdChange={() => {}}
      />
      <Inspector
        runId={1}
        selectedNode={EMBEDDING_NODE}
        selectedNodeId={selectedNodeId}
        diagnosticSnapshot={diagnosticSnapshot}
        currentStep={diagnosticSnapshot?.generation_step ?? null}
        isLoading={diagnosticLoading}
        attentionHead={null}
        onAttentionHeadChange={() => {}}
        showQKVDetail={false}
        onShowQKVDetailChange={() => {}}
        qkvWindowOffset={0}
        nodeWindowOffset={0}
        onQkvWindowOffsetChange={() => {}}
        onNodeWindowOffsetChange={() => {}}
        numHeads={6}
        onOpenDataTab={() => {}}
        activeTab={activeTab}
        onActiveTabChange={setActiveTab}
      />
    </div>
  );
}

const ATTENTION_NODE: ArchitectureNode = {
  id: "block.0.attention",
  kind: "attention",
  label: "Causal Self-Attention",
  config: {},
};

// Full App.tsx-equivalent wiring for the attention case specifically —
// attentionBlock derived from selectedNodeId the same way App.tsx does,
// attentionHead as real state (Runtime auto-defaults it via
// onAttentionHeadChange, matching production). Given how much recent
// churn the attention/heatmap code has had, worth ruling out separately
// from the generic-node case above.
function AttentionHarness() {
  const selectedNodeId = "block.0.attention";
  const [diagnosticSnapshot, setDiagnosticSnapshot] = useState<DiagnosticSnapshot | null>(null);
  const [diagnosticLoading, setDiagnosticLoading] = useState(false);
  const [attentionHead, setAttentionHead] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<SubTab>("overview");
  const attentionBlockMatch = selectedNodeId.match(/^block\.(\d+)\.attention$/);
  const attentionBlock = attentionBlockMatch ? parseInt(attentionBlockMatch[1], 10) : null;

  return (
    <div>
      <PausePrompt
        runId={1}
        canPrompt={true}
        attentionBlock={attentionBlock}
        attentionHead={attentionHead}
        showQKVDetail={false}
        attentionWindowOffset={0}
        nodeWindowOffset={0}
        maxNewTokens={100}
        temperature={0.8}
        decodingMode="sample"
        onDiagnosticSnapshot={(snapshot) => {
          setDiagnosticSnapshot(snapshot);
          setDiagnosticLoading(false);
        }}
        onSessionIdChange={() => {}}
      />
      <Inspector
        runId={1}
        selectedNode={ATTENTION_NODE}
        selectedNodeId={selectedNodeId}
        diagnosticSnapshot={diagnosticSnapshot}
        currentStep={diagnosticSnapshot?.generation_step ?? null}
        isLoading={diagnosticLoading}
        attentionHead={attentionHead}
        onAttentionHeadChange={setAttentionHead}
        showQKVDetail={false}
        onShowQKVDetailChange={() => {}}
        qkvWindowOffset={0}
        nodeWindowOffset={0}
        onQkvWindowOffsetChange={() => {}}
        onNodeWindowOffsetChange={() => {}}
        numHeads={6}
        onOpenDataTab={() => {}}
        activeTab={activeTab}
        onActiveTabChange={setActiveTab}
      />
    </div>
  );
}

const FAKE_ATTENTION_SNAPSHOT: DiagnosticSnapshot = {
  schema_version: 1,
  diagnostic_session_id: "diag-test",
  generation_step: 1,
  input_tokens: [{ position: 0, id: 46, text: "h" }],
  generated_token: { position: 1, id: 43, text: "e" },
  nodes: {},
  attention: {
    available: true,
    layer: 0,
    head: 0,
    weights: [[1.0]],
    token_labels: ["h"],
    window_start: 0,
    total_positions: 1,
  },
  // The heatmap is now a canvas fed by attention_full; the "Layer 1, Head 1"
  // header renders once this is available.
  attention_full: {
    available: true,
    layer: 0,
    head: 0,
    weights: [[1.0]],
    token_labels: ["h"],
    total_positions: 1,
    block_size: 128,
  },
  activation_summaries: { available: false, reason: "Not requested" },
  lm_head: { logits_shape: [1, 1, 65], selected_position: 0, top_k: [], top_k_by_position: [] },
  position_tokens: [{ position: 0, id: 46, token: "h" }],
  complete: true,
};

const FAKE_SNAPSHOT: DiagnosticSnapshot = {
  schema_version: 1,
  diagnostic_session_id: "diag-test",
  generation_step: 1,
  input_tokens: [{ position: 0, id: 46, text: "h" }],
  generated_token: { position: 1, id: 43, text: "e" },
  nodes: {
    embedding: {
      input_shape: [1, 1],
      output_shape: [1, 1, 192],
      summary: { mean: 0.1234, std: 0.5, l2_norm: 1.2, min: -1, max: 1 },
      position_vectors: { positions: [0], vectors: [[0.1, 0.2, 0.3]] },
      input_position_vectors: null,
    },
  },
  attention: { available: false, reason: "Not requested" },
  activation_summaries: { available: false, reason: "Not requested" },
  lm_head: { logits_shape: [1, 1, 65], selected_position: 0, top_k: [], top_k_by_position: [] },
  position_tokens: [{ position: 0, id: 46, token: "h" }],
  complete: true,
};

const FAKE_SNAPSHOT_2: DiagnosticSnapshot = {
  ...FAKE_SNAPSHOT,
  generation_step: 2,
  generated_token: { position: 2, id: 50, text: "l" },
  nodes: {
    embedding: {
      ...FAKE_SNAPSHOT.nodes.embedding,
      summary: { mean: 0.9999, std: 0.5, l2_norm: 1.2, min: -1, max: 1 },
    },
  },
};

describe("PausePrompt + Inspector integration", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("Inspector reflects the new snapshot immediately after one Step click, no second interaction needed", async () => {
    vi.spyOn(api, "startDiagnostic").mockResolvedValue(
      { diagnostic_session_id: "diag-test", tokens: [{ position: 0, id: 46, text: "h" }] } as DiagnosticSessionResponse,
    );
    vi.spyOn(api, "stepDiagnostic").mockResolvedValue(FAKE_SNAPSHOT);

    render(<Harness />);

    fireEvent.change(screen.getByPlaceholderText("Enter a prompt..."), { target: { value: "hello" } });
    fireEvent.click(screen.getByRole("button", { name: ">" }));

    // Switch to the Runtime sub-tab once (a real, expected navigation —
    // not the "click Step again" the user reported).
    fireEvent.click(screen.getByRole("button", { name: /runtime/i }));

    // Should reflect the fresh snapshot's summary stats without any
    // further interaction — mean: 0.1234 only exists in FAKE_SNAPSHOT.
    await waitFor(() => expect(screen.getByText(/0\.1234/)).toBeInTheDocument());
  });

  it("still reflects the fresh snapshot when Runtime is already open BEFORE clicking Step", async () => {
    vi.spyOn(api, "startDiagnostic").mockResolvedValue(
      { diagnostic_session_id: "diag-test", tokens: [{ position: 0, id: 46, text: "h" }] } as DiagnosticSessionResponse,
    );
    vi.spyOn(api, "stepDiagnostic").mockResolvedValue(FAKE_SNAPSHOT);

    render(<Harness />);

    // Open Runtime FIRST — it'll say "no diagnostic step has been run yet".
    fireEvent.click(screen.getByRole("button", { name: /runtime/i }));
    expect(screen.getByText(/no diagnostic step has been run yet/i)).toBeInTheDocument();

    fireEvent.change(screen.getByPlaceholderText("Enter a prompt..."), { target: { value: "hello" } });
    fireEvent.click(screen.getByRole("button", { name: ">" }));

    await waitFor(() => expect(screen.getByText(/0\.1234/)).toBeInTheDocument());
  });

  it("second Step click shows the SECOND snapshot's data, not a lagging first one", async () => {
    vi.spyOn(api, "startDiagnostic").mockResolvedValue(
      { diagnostic_session_id: "diag-test", tokens: [{ position: 0, id: 46, text: "h" }] } as DiagnosticSessionResponse,
    );
    const step = vi.spyOn(api, "stepDiagnostic")
      .mockResolvedValueOnce(FAKE_SNAPSHOT)
      .mockResolvedValueOnce(FAKE_SNAPSHOT_2);

    render(<Harness />);
    fireEvent.click(screen.getByRole("button", { name: /runtime/i }));

    fireEvent.change(screen.getByPlaceholderText("Enter a prompt..."), { target: { value: "hello" } });
    fireEvent.click(screen.getByRole("button", { name: ">" }));
    await waitFor(() => expect(screen.getByText(/0\.1234/)).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: ">" }));
    await waitFor(() => expect(step).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.getByText(/0\.9999/)).toBeInTheDocument());
    expect(screen.queryByText(/0\.1234/)).not.toBeInTheDocument();
  });

  it("attention node: Inspector's heatmap reflects the fresh snapshot immediately after one Step click", async () => {
    vi.spyOn(api, "startDiagnostic").mockResolvedValue(
      { diagnostic_session_id: "diag-test", tokens: [{ position: 0, id: 46, text: "h" }] } as DiagnosticSessionResponse,
    );
    vi.spyOn(api, "stepDiagnostic").mockResolvedValue(FAKE_ATTENTION_SNAPSHOT);

    render(<AttentionHarness />);
    fireEvent.click(screen.getByRole("button", { name: /runtime/i }));

    // Runtime's own effect auto-defaults Head to 0 the moment an attention
    // node is selected — matches production.
    fireEvent.change(screen.getByPlaceholderText("Enter a prompt..."), { target: { value: "hello" } });
    fireEvent.click(screen.getByRole("button", { name: ">" }));

    // "Layer 1, Head 1" (1-indexed display) only renders once
    // snapshot.attention.available is true — confirms the heatmap picked
    // up the fresh snapshot without a second click.
    await waitFor(() => expect(screen.getByText(/Layer 1, Head 1/)).toBeInTheDocument());
  });
});
