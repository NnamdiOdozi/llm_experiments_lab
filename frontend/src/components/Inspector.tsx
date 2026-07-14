import { useState, useEffect, CSSProperties } from "react";
import { ArchitectureNode, DiagnosticSnapshot } from "../types";
import { fetchEmbeddingTable } from "../hooks/useApi";

interface Props {
  runId: number | null;
  selectedNode: ArchitectureNode | null;
  selectedNodeId: string | null;
  diagnosticSnapshot: DiagnosticSnapshot | null;
  currentStep: number | null;
  isLoading: boolean;
  // Head/Q-K-V-detail selection — block is implied by selectedNodeId
  // (block.{i}.attention), so only head needs picking. Lives here rather
  // than in PausePrompt since it's only ever meaningful once you've
  // selected an attention node, which happens here. See
  // docs/DESIGN_DECISIONS.md.
  attentionHead: number | null;
  onAttentionHeadChange: (head: number | null) => void;
  showQKVDetail: boolean;
  onShowQKVDetailChange: (show: boolean) => void;
  // Shifts the attention heatmap/qkv_detail window earlier in the sequence
  // — 0 = most recent. See docs/DESIGN_DECISIONS.md.
  attentionWindowOffset: number;
  onAttentionWindowOffsetChange: (offset: number) => void;
  // Same idea, for every other node's position_vectors window (LayerNorm,
  // MLP, embedding, final_norm) — direct user request, 2026-07-15. See
  // docs/DESIGN_DECISIONS.md.
  nodeWindowOffset: number;
  onNodeWindowOffsetChange: (offset: number) => void;
  // config.model.n_head — bounds the Head dropdown so it only ever lists
  // real options, never a value the model doesn't have.
  numHeads: number | null;
  // Double-clicking a vector cell opens a full, static, copyable view in a
  // new closeable tab at the App level (Colab/VS Code variable-inspector
  // pattern, per direct user reference 2026-07-15). title identifies node +
  // position + block/head; content is the full-precision vector text.
  onOpenDataTab: (title: string, content: number[]) => void;
  // Lifted to App.tsx — Inspector unmounts whenever a data tab opens (App.tsx
  // conditionally renders it only when rightPaneTab === "inspector"), so a
  // local useState here reset back to "overview" every time a data tab was
  // closed, even if the user had been on Runtime (where data tabs are always
  // opened from). Direct user report, 2026-07-15. See docs/DESIGN_DECISIONS.md.
  activeTab: SubTab;
  onActiveTabChange: (tab: SubTab) => void;
}

// Hardcoded descriptions per node kind
const NODE_DESCRIPTIONS: Record<string, string> = {
  embedding: "Converts token IDs to dense vectors and adds positional information.",
  layernorm: "Normalizes layer activations to mean=0, std=1 for stable training.",
  attention: "Allows each token to attend to all previous tokens and blend their information.",
  mlp: "Dense feed-forward network that applies non-linear transformations.",
  moe: "Mixture of Experts: routes each token to a subset of expert networks via a learned gating mechanism. Structurally distinct from standard dense MLP.",
  lm_head: "Projects final hidden states to vocabulary-size logits for next-token prediction.",
  transformer_block_group: "A stack of transformer blocks, each containing attention and feed-forward layers.",
};

const NODE_MATH: Record<string, { formula: string; explanation: string }> = {
  embedding: {
    formula: "X = E[token_ids]",
    explanation: "Each token ID is replaced by a learned vector from the embedding table.",
  },
  layernorm: {
    formula: "y = γ((x - μ) / √(σ² + ε)) + β",
    explanation: "Rescales activations to have mean 0 and standard deviation 1.",
  },
  attention: {
    formula: "Attention(Q,K,V) = softmax(QKᵀ / √d_head)V",
    explanation: "Each token looks at other tokens and blends their information, weighted by relevance.",
  },
  mlp: {
    formula: "MLP(x) = σ(xW₁ + b₁)W₂ + b₂",
    explanation: "Two dense layers with non-linear activation between them.",
  },
  moe: {
    formula: "output = Σ (G[i](x) * Expert[i](x)) for i in top_k experts",
    explanation: "A gating network G assigns weights to experts; tokens route to top-k experts selected by gate output.",
  },
  lm_head: {
    formula: "logits = xW + b",
    explanation: "Linear projection of the final hidden state to vocabulary-size logits.",
  },
};

export type SubTab = "overview" | "shapes" | "math" | "config" | "runtime";

const tabStyle: CSSProperties = {
  display: "flex",
  gap: 4,
  marginBottom: 12,
  borderBottom: "1px solid var(--border)",
  paddingBottom: 8,
};

const tabButtonStyle = (active: boolean): CSSProperties => ({
  background: "none",
  border: "none",
  color: active ? "var(--accent)" : "var(--text-dim)",
  cursor: "pointer",
  fontSize: 12,
  fontWeight: active ? 600 : 400,
  paddingBottom: 4,
  borderBottom: active ? "2px solid var(--accent)" : "none",
  transition: "all 0.15s",
});

function formatShape(dims: (string | number)[]): string {
  return `[${dims.join(", ")}]`;
}

function Overview({ node }: { node: ArchitectureNode }) {
  const desc = NODE_DESCRIPTIONS[node.kind] || "No description available.";
  return <p style={{ fontSize: 12, lineHeight: 1.6, color: "var(--text)" }}>{desc}</p>;
}

function Shapes({ node, snapshot, selectedNodeId }: { node: ArchitectureNode; snapshot: DiagnosticSnapshot | null; selectedNodeId: string | null }) {
  const staticShapes = node.static_shapes || [];
  const runtimeData = selectedNodeId ? snapshot?.nodes[selectedNodeId] : null;

  return (
    <div style={{ fontSize: 12 }}>
      {staticShapes.length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <h4 style={{ fontSize: 11, color: "var(--accent)", marginBottom: 6 }}>Static Shapes</h4>
          {staticShapes.map((shape, idx) => (
            <div key={idx} style={{ marginBottom: 4, color: "var(--text-dim)" }}>
              <strong>{shape.name}:</strong> {formatShape(shape.dims)}
            </div>
          ))}
        </div>
      )}

      {runtimeData ? (
        <div>
          <h4 style={{ fontSize: 11, color: "var(--accent)", marginBottom: 6 }}>Runtime Shapes (Step {snapshot?.generation_step})</h4>
          {runtimeData.input_shape && (
            <div style={{ marginBottom: 4, color: "var(--text-dim)" }}>
              <strong>input:</strong> {formatShape(runtimeData.input_shape)}
            </div>
          )}
          {runtimeData.output_shape && (
            <div style={{ marginBottom: 4, color: "var(--text-dim)" }}>
              <strong>output:</strong> {formatShape(runtimeData.output_shape)}
            </div>
          )}
        </div>
      ) : (
        <p style={{ color: "var(--text-dim)" }}>Not captured</p>
      )}
    </div>
  );
}

function MathTab({ node }: { node: ArchitectureNode }) {
  const math = NODE_MATH[node.kind];
  if (!math) {
    return <p style={{ fontSize: 12, color: "var(--text-dim)" }}>No mathematical formula available.</p>;
  }
  return (
    <div style={{ fontSize: 12 }}>
      <div
        style={{
          fontFamily: "var(--font-mono)",
          background: "var(--bg)",
          padding: 8,
          borderRadius: 4,
          marginBottom: 8,
          border: "1px solid var(--border)",
          color: "var(--text)",
        }}
      >
        {math.formula}
      </div>
      <p style={{ fontSize: 12, lineHeight: 1.6, color: "var(--text)" }}>{math.explanation}</p>
    </div>
  );
}

function Config({ node }: { node: ArchitectureNode }) {
  const config = node.config || {};
  if (Object.keys(config).length === 0) {
    return <p style={{ fontSize: 12, color: "var(--text-dim)" }}>No configuration parameters.</p>;
  }
  return (
    <div style={{ fontSize: 12 }}>
      {Object.entries(config).map(([key, val]) => (
        <div key={key} style={{ marginBottom: 6, display: "flex", justifyContent: "space-between" }}>
          <span style={{ color: "var(--text-dim)" }}>{key}:</span>
          <strong style={{ color: "var(--text)" }}>{String(val)}</strong>
        </div>
      ))}
    </div>
  );
}

const positionTableCellStyle: CSSProperties = {
  border: "1px solid var(--border)",
  padding: "4px 6px",
  textAlign: "left",
  whiteSpace: "nowrap",
};

// LM Head: a stepper, not a table (unlike QKV) — each position carries a
// full ranked top-5 list with bars, too much to compress into a table
// row/tooltip usefully. Defaults to the most recent position (index
// length-1), matching what a single fixed "Top-k Tokens" view showed
// before this was steppable — numerically identical, since
// top_k_by_position's last entry and lm_head.top_k both come from the same
// final logits row. See docs/DESIGN_DECISIONS.md.
function LmHeadStepper({ snapshot }: { snapshot: DiagnosticSnapshot }) {
  // Defensive: top_k_by_position is a new field (2026-07-14) — a snapshot
  // from a trainer container/session that predates this change (e.g. a
  // serverless run against an un-rebuilt trainer image, or a diagnostic
  // session started before a local backend restart picked up the change)
  // won't have it. Fall back to empty rather than crash on `.length` of
  // undefined. See docs/DESIGN_DECISIONS.md.
  const entries = snapshot.lm_head.top_k_by_position ?? [];
  const [index, setIndex] = useState(entries.length - 1);
  // Track the newest position automatically as the user steps through the
  // model (new generation_step -> new snapshot) rather than staying
  // wherever the stepper was last left — stepping the model is the more
  // common action than manually browsing backward, so "jump to latest"
  // should be the default each time, not something you re-navigate to
  // after every click. Manual ◀/▶ still works freely between steps.
  useEffect(() => {
    setIndex(entries.length - 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [snapshot.generation_step]);
  const clampedIndex = Math.min(index, entries.length - 1);
  const entry = entries[clampedIndex];

  if (!entry) {
    return <div style={{ fontSize: 12, color: "var(--text-dim)" }}>No per-position data captured yet.</div>;
  }

  // Direct user request, 2026-07-15: highlight "what actually got
  // selected" at ANY browsable position, not just the most recent one —
  // actual_next_token_id (backend-computed ground truth, see
  // docs/DESIGN_DECISIONS.md) makes this a plain id comparison regardless
  // of position. Text for the note below comes from the NEXT entry in
  // this same list (its .token IS the decoded text of this position's
  // actual-next id) — except at the very last entry, where the actual
  // next token isn't itself in this windowed list, so fall back to
  // snapshot.generated_token (the two are guaranteed to be the same
  // token either way — see the backend comment on full_next_tokens).
  const actualNextText =
    clampedIndex < entries.length - 1 ? entries[clampedIndex + 1]?.token : snapshot.generated_token?.text;

  return (
    <div style={{ fontSize: 11 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <button onClick={() => setIndex(Math.max(0, clampedIndex - 1))} disabled={clampedIndex === 0}>
          ◀
        </button>
        <span>
          {/* Display 1-indexed — internal entry.position stays 0-indexed.
              Just the position, per direct user feedback 2026-07-15 — the
              "X of Y shown (last N of M total)" window-size explanation
              was unnecessary noise once you can already see the ◀/▶
              controls. See docs/DESIGN_DECISIONS.md. */}
          Position {entry.position + 1} ("{entry.token}")
        </span>
        <button onClick={() => setIndex(Math.min(entries.length - 1, clampedIndex + 1))} disabled={clampedIndex === entries.length - 1}>
          ▶
        </button>
      </div>
      {entry.top_k.map((tk) => {
        const isSelected = tk.token_id === entry.actual_next_token_id;
        return (
          <div
            key={tk.rank}
            style={{
              marginBottom: 8,
              padding: 4,
              background: isSelected ? "rgba(76, 175, 80, 0.1)" : "transparent",
              borderRadius: 4,
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
              <span>
                {/* Quoted — direct user request 2026-07-14: an
                    unquoted space character rendered as visibly nothing,
                    making it look like the row was empty/broken rather
                    than a real, meaningful prediction. See
                    docs/DESIGN_DECISIONS.md. */}
                <strong>#{tk.rank}</strong> "{tk.token}" (id={tk.token_id})
                {isSelected && <span style={{ color: "#4caf50", marginLeft: 6 }}>← selected</span>}
              </span>
              <span style={{ color: "var(--text-dim)" }}>
                logit {tk.logit.toFixed(2)} · {(tk.probability * 100).toFixed(1)}%
              </span>
            </div>
            <div style={{ background: "var(--bg)", borderRadius: 3, height: 8, overflow: "hidden" }}>
              <div
                style={{
                  width: `${Math.max(tk.probability * 100, 1)}%`,
                  height: "100%",
                  background: isSelected ? "#4caf50" : "var(--accent)",
                  borderRadius: 3,
                }}
              />
            </div>
          </div>
        );
      })}
      {entry.actual_next_token_id != null && !entry.top_k.some((tk) => tk.token_id === entry.actual_next_token_id) && (
        <div style={{ marginTop: 6, color: "var(--text-dim)" }}>
          Selected token "{actualNextText}" (id={entry.actual_next_token_id}) fell outside the top 5 — temperature sampling can pick a lower-probability token.
        </div>
      )}
    </div>
  );
}

// Colab-variable-inspector style: one row per position, values truncated
// with the full vector available via the native title tooltip on hover —
// same lightweight mechanism the heatmap cells already use, shows the
// whole window (up to DIAGNOSTIC_POSITION_WINDOW positions) at once instead
// of stepping through one at a time. See docs/DESIGN_DECISIONS.md.
// Consistent everywhere a vector is truncated for preview (real feedback,
// 2026-07-15: was arbitrary — first-4-only in one place — and needed to be
// standard). First 6, "…", last 6 — the same convention used everywhere.
// Vectors shorter than 2*edgeLen just show in full, no ellipsis. See
// docs/DESIGN_DECISIONS.md.
function truncatedVector(v: number[], edgeLen = 6): { preview: string; full: string } {
  const preview =
    v.length <= edgeLen * 2
      ? `[${v.map((x) => x.toFixed(2)).join(", ")}]`
      : `[${v.slice(0, edgeLen).map((x) => x.toFixed(2)).join(", ")}, …, ${v.slice(-edgeLen).map((x) => x.toFixed(2)).join(", ")}]`;
  return {
    preview,
    full: `[${v.map((x) => x.toFixed(4)).join(", ")}]`,
  };
}

// Icon button, not spelled-out text — direct user request, 2026-07-15
// ("I think it looks better"). Same copy/checkmark glyph pair already
// used in ChatPanel.tsx, reused here for icon consistency across the app
// rather than inventing a new one. Brief checkmark confirmation on click,
// same pattern as ChatPanel's own copy button. See docs/DESIGN_DECISIONS.md.
function CopyIconButton({ getText, title }: { getText: () => string; title: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={() => {
        navigator.clipboard.writeText(getText());
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
      title={title}
      style={{ background: "none", border: "none", color: "var(--text-dim)", cursor: "pointer", padding: 0, lineHeight: 0 }}
    >
      {copied ? (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      ) : (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="9" y="9" width="11" height="11" rx="2" />
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
        </svg>
      )}
    </button>
  );
}

// Shared by QKVTable/NodeVectorTable — one vector-kind (Q, K, V, Input,
// Output) rendered as its own Position(+Token)/Value table. Previously Q/K/V
// and Input/Output were side-by-side columns in one table; direct user
// request 2026-07-15: stack them vertically instead (Input above Output;
// Q above K above V) so each is read top-to-bottom on its own rather than
// scanned across a wide row. See docs/DESIGN_DECISIONS.md.
function VectorPreviewTable({
  label, positions, tokens, vectors, onOpenCell,
}: {
  label: string;
  positions: number[];
  tokens?: string[];
  vectors: number[][];
  onOpenCell: (pos: number, vector: number[]) => void;
}) {
  // One button copies every row in this table at once — full-precision
  // values (vectors[i] itself, not the truncated display string), tab-
  // separated so it pastes into a spreadsheet as a real grid (one row per
  // position). Deliberately not a per-row copy icon — double-click already
  // opens any single vector in its own copyable tab, so a per-row icon
  // here would just duplicate that path and clutter a table that can have
  // many rows. Direct user request, 2026-07-15. Blank line between rows
  // added 2026-07-16 — long rows word-wrap in Word/Docs and adjacent
  // vectors blurred together with no visual gap; still a real tab-grid
  // for spreadsheet paste, just with a separator line Excel/Sheets ignore
  // as an empty row. See docs/DESIGN_DECISIONS.md.
  const tableText = () => positions.map((pos, i) => [pos + 1, ...vectors[i]].join("\t")).join("\n\n");
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
        <div style={{ fontSize: 10, color: "var(--text-dim)" }}>{label}</div>
        <CopyIconButton getText={tableText} title={`Copy all ${label} vectors (full precision, tab-separated)`} />
      </div>
      <table style={{ borderCollapse: "collapse", fontSize: 10, fontFamily: "var(--font-mono)", width: "100%" }}>
        <thead>
          <tr>
            <th style={positionTableCellStyle}>Position</th>
            {tokens && <th style={positionTableCellStyle}>Token</th>}
            <th style={positionTableCellStyle}>{label}</th>
          </tr>
        </thead>
        <tbody>
          {positions.map((pos, i) => {
            const t = truncatedVector(vectors[i]);
            return (
              <tr key={pos}>
                <td style={{ ...positionTableCellStyle, color: "var(--text-dim)" }}>{pos + 1}</td>
                {tokens && <td style={{ ...positionTableCellStyle, color: "var(--text)" }}>"{tokens[i]}"</td>}
                <td
                  style={{ ...positionTableCellStyle, color: "var(--text)", cursor: "pointer" }}
                  title={t.full}
                  onDoubleClick={() => onOpenCell(pos, vectors[i])}
                >
                  {t.preview}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function QKVTable({
  qkv, blockNum, head, onOpenDataTab,
}: {
  qkv: import("../types").QKVDetail;
  blockNum: number | null;
  head: number | null;
  onOpenDataTab: (title: string, content: number[]) => void;
}) {
  const openCell = (kind: "Q" | "K" | "V", pos: number, vector: number[]) => {
    const title = `Block ${blockNum != null ? blockNum + 1 : "?"} Head ${head != null ? head + 1 : "?"} — ${kind} — pos ${pos + 1}`;
    onOpenDataTab(title, vector);
  };
  return (
    <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--border)" }}>
      <div style={{ fontSize: 11, color: "var(--accent)", marginBottom: 8 }}>
        Q/K/V Detail — last {qkv.positions.length} position{qkv.positions.length === 1 ? "" : "s"} (hover a cell for full vector, double-click to open in a tab)
      </div>
      <VectorPreviewTable label="Q" positions={qkv.positions} tokens={qkv.tokens} vectors={qkv.q} onOpenCell={(pos, v) => openCell("Q", pos, v)} />
      <VectorPreviewTable label="K" positions={qkv.positions} tokens={qkv.tokens} vectors={qkv.k} onOpenCell={(pos, v) => openCell("K", pos, v)} />
      <VectorPreviewTable label="V" positions={qkv.positions} tokens={qkv.tokens} vectors={qkv.v} onOpenCell={(pos, v) => openCell("V", pos, v)} />
    </div>
  );
}

// The embedding node's INPUT, not its output — direct user request
// 2026-07-15: the Runtime tab showed the embedding's float output vectors
// (what NodeVectorTable shows for every other node), but for embedding
// specifically what's actually illuminating is the one-hot encoded input:
// position, character, and a vocab_size-wide vector that's all zeros
// except a 1 at that token's id. Replaces the output-vector table for this
// one node entirely — not shown alongside it. Synthesized client-side from
// snapshot.position_tokens + vocab_size (from lm_head.logits_shape); no
// new per-node backend capture needed since a one-hot vector is fully
// determined by the token id alone. See docs/DESIGN_DECISIONS.md.
function EmbeddingOneHotTable({
  snapshot, onOpenDataTab,
}: {
  snapshot: DiagnosticSnapshot;
  onOpenDataTab: (title: string, content: number[]) => void;
}) {
  const positionTokens = snapshot.position_tokens;
  const shape = snapshot.lm_head.logits_shape;
  const vocabSize = shape.length > 0 ? shape[shape.length - 1] : null;

  if (!positionTokens || positionTokens.length === 0 || vocabSize == null) {
    return <div style={{ fontSize: 12, color: "var(--text-dim)", marginTop: 12 }}>Not captured</div>;
  }

  const oneHot = (id: number): number[] => {
    const v = new Array(vocabSize).fill(0);
    v[id] = 1;
    return v;
  };

  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ fontSize: 11, color: "var(--accent)", marginBottom: 8 }}>
        Input Vectors (one-hot, width {vocabSize}) — hover for full vector, double-click to open in a tab
      </div>
      <table style={{ borderCollapse: "collapse", fontSize: 10, fontFamily: "var(--font-mono)", width: "100%" }}>
        <thead>
          <tr>
            <th style={positionTableCellStyle}>Position</th>
            <th style={positionTableCellStyle}>Character</th>
            <th style={positionTableCellStyle}>One-Hot Vector</th>
          </tr>
        </thead>
        <tbody>
          {positionTokens.map((pt) => {
            const vec = oneHot(pt.id);
            const t = truncatedVector(vec);
            return (
              <tr key={pt.position}>
                <td style={{ ...positionTableCellStyle, color: "var(--text-dim)" }}>{pt.position + 1}</td>
                <td style={{ ...positionTableCellStyle, color: "var(--text)" }}>"{pt.token}"</td>
                <td
                  style={{ ...positionTableCellStyle, color: "var(--text)", cursor: "pointer" }}
                  title={t.full}
                  onDoubleClick={() => onOpenDataTab(`Embedding Input — "${pt.token}" — pos ${pt.position + 1}`, vec)}
                >
                  {t.preview}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// Direct user request, 2026-07-15: "a stepper that allows that window to
// slide backwards in time" for every node with a position_vectors view —
// same shape/stride-1 pattern as the attention heatmap's stepper. Shared
// by every generic node (LayerNorm, MLP, embedding, final_norm) via
// Runtime below — one stepper controls the shared node_window_offset,
// same as attention's single stepper controlling both the heatmap and
// qkv_detail. See docs/DESIGN_DECISIONS.md.
function NodeWindowStepper({
  pv, totalPositions, windowOffset, onWindowOffsetChange,
}: {
  pv: import("../types").NodePositionVectors;
  totalPositions: number;
  windowOffset: number;
  onWindowOffsetChange: (offset: number) => void;
}) {
  const windowSize = pv.positions.length;
  const windowStart = pv.positions[0] ?? 0;
  const maxOffset = Math.max(0, totalPositions - windowSize);
  const canShiftEarlier = windowOffset < maxOffset;
  const canShiftLater = windowOffset > 0;

  if (totalPositions <= windowSize) return null;

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, fontSize: 11 }}>
      <button
        onClick={() => onWindowOffsetChange(Math.min(maxOffset, windowOffset + 1))}
        disabled={!canShiftEarlier}
        title="Shift window one position earlier"
        style={{ fontSize: 11, padding: "2px 6px" }}
      >
        ◀ Earlier
      </button>
      <span style={{ color: "var(--text-dim)" }}>
        Positions {windowStart + 1}–{windowStart + windowSize} of {totalPositions}
      </span>
      <button
        onClick={() => onWindowOffsetChange(Math.max(0, windowOffset - 1))}
        disabled={!canShiftLater}
        title="Shift window one position later"
        style={{ fontSize: 11, padding: "2px 6px" }}
      >
        Later ▶
      </button>
    </div>
  );
}

// Same Colab-style table as QKVTable, generalized to any node's raw
// input/output vectors (embedding, layernorm, mlp, etc.) — no token text
// (position numbers alone are enough to correlate against the
// heatmap/top-k tables if needed, and re-decoding tokens at every one of
// ~18 nodes per step for a column that's already shown elsewhere isn't
// worth the payload). Shows both Input and Output columns when input is
// available (real gap flagged live, 2026-07-15: output-only, no way to see
// what a LayerNorm actually changed) — embedding has no input column since
// its input is token ids, not a per-position float vector. See
// docs/DESIGN_DECISIONS.md.
function NodeVectorTable({
  outputPv, inputPv, nodeId, onOpenDataTab,
}: {
  outputPv: import("../types").NodePositionVectors;
  inputPv?: import("../types").NodePositionVectors | null;
  nodeId: string | null;
  onOpenDataTab: (title: string, content: number[]) => void;
}) {
  const openCell = (kind: "Input" | "Output", pos: number, vector: number[]) => {
    onOpenDataTab(`${nodeId ?? "node"} — ${kind} — pos ${pos + 1}`, vector);
  };
  return (
    <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--border)" }}>
      <div style={{ fontSize: 11, color: "var(--accent)", marginBottom: 8 }}>
        {inputPv ? "Input / Output Vectors" : "Output Vectors"} — last {outputPv.positions.length} position
        {outputPv.positions.length === 1 ? "" : "s"} (hover for full vector, double-click to open in a tab)
      </div>
      {/* Input above Output — direct user request 2026-07-15, was
          side-by-side columns before. See docs/DESIGN_DECISIONS.md. */}
      {inputPv && (
        <VectorPreviewTable label="Input" positions={inputPv.positions} vectors={inputPv.vectors} onOpenCell={(pos, v) => openCell("Input", pos, v)} />
      )}
      <VectorPreviewTable label="Output" positions={outputPv.positions} vectors={outputPv.vectors} onOpenCell={(pos, v) => openCell("Output", pos, v)} />
    </div>
  );
}

// Phase 2: Render attention heatmap for attention nodes
function AttentionHeatmap({
  snapshot, blockNum, head, onOpenDataTab, windowOffset, onWindowOffsetChange,
}: {
  snapshot: DiagnosticSnapshot;
  blockNum: number | null;
  head: number | null;
  onOpenDataTab: (title: string, content: number[]) => void;
  windowOffset: number;
  onWindowOffsetChange: (offset: number) => void;
}) {
  const att = snapshot.attention;
  if (!att.available) {
    return (
      <div style={{ fontSize: 12, color: "var(--text-dim)" }}>
        {att.reason ? `Not captured: ${att.reason}` : "Not captured"}
      </div>
    );
  }

  if (!att.weights || !att.token_labels) {
    return (
      <div style={{ fontSize: 12, color: "var(--text-dim)" }}>
        Attention data unavailable
      </div>
    );
  }

  const tokenLabels = att.token_labels;
  const windowStart = att.window_start ?? 0;
  const totalPositions = att.total_positions ?? tokenLabels.length;
  const windowSize = tokenLabels.length;
  // offset=0 is "most recent" (window's end sits at totalPositions); larger
  // offset shifts the window's end earlier. Can't shift the window's end
  // past totalPositions (offset < 0, meaningless) or its start before 0
  // (offset so large the window would run off the front of the sequence).
  const maxOffset = Math.max(0, totalPositions - windowSize);
  const canShiftEarlier = windowOffset < maxOffset;
  const canShiftLater = windowOffset > 0;

  // Per-row normalization, not global: each row is its own probability
  // distribution (sums to 1), so comparing within a row is what's actually
  // meaningful. A global min/max was dominated by the trivial (0,0)=1.0
  // outlier (first token can only attend to itself under causal masking —
  // mathematically guaranteed, not learned), which made almost every other
  // cell normalize to near-zero opacity and vanish against the dark theme.
  // Masked cells (j > i, structurally always exactly 0 under causal
  // masking) get their own muted style instead of being mixed into the
  // color scale as if they were real low-attention values. A visible
  // opacity floor means even the lowest real attention value in a row is
  // still a faint, visible tile, not literally invisible. See
  // docs/DESIGN_DECISIONS.md.
  const OPACITY_FLOOR = 0.15;

  return (
    <div>
      <div style={{ marginBottom: 8, fontSize: 11, color: "var(--accent)" }}>
        Layer {att.layer != null ? att.layer + 1 : "?"}, Head {att.head != null ? att.head + 1 : "?"}
      </div>
      {/* Heatmap window stepper — real user report, 2026-07-13: an
          unwindowed T x T grid "gets very busy very quickly" as a session
          grows. Shows the same DIAGNOSTIC_POSITION_WINDOW-wide slice as
          qkv_detail below (shared window, one stepper controls both), and
          lets the user shift it earlier/later through the sequence instead
          of only ever seeing the tail. Stride 1 (smooth slide, one position
          at a time) — previously stepped by the full window size at once
          ("discontinuous", direct user report 2026-07-15). See
          docs/DESIGN_DECISIONS.md. */}
      {totalPositions > windowSize && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, fontSize: 11 }}>
          <button
            onClick={() => onWindowOffsetChange(Math.min(maxOffset, windowOffset + 1))}
            disabled={!canShiftEarlier}
            title="Shift window one position earlier"
            style={{ fontSize: 11, padding: "2px 6px" }}
          >
            ◀ Earlier
          </button>
          <span style={{ color: "var(--text-dim)" }}>
            Positions {windowStart + 1}–{windowStart + windowSize} of {totalPositions}
          </span>
          <button
            onClick={() => onWindowOffsetChange(Math.max(0, windowOffset - 1))}
            disabled={!canShiftLater}
            title="Shift window one position later"
            style={{ fontSize: 11, padding: "2px 6px" }}
          >
            Later ▶
          </button>
        </div>
      )}
      <div style={{ overflowX: "auto" }}>
        <table
          style={{
            borderCollapse: "collapse",
            fontSize: 10,
            fontFamily: "var(--font-mono)",
          }}
        >
          <thead>
            <tr>
              <th
                style={{ border: "1px solid var(--border)", padding: 4, textAlign: "center", width: 40, color: "var(--text-dim)", fontSize: 9 }}
                title="Rows = query position (attending from). Columns = key position (attended to)."
              >
                Q\K
              </th>
              {att.token_labels.map((token, j) => (
                <th
                  key={j}
                  style={{
                    border: "1px solid var(--border)",
                    padding: 4,
                    minWidth: 40,
                    textAlign: "center",
                    color: "var(--text-dim)",
                  }}
                  title={`Position ${windowStart + j + 1}`}
                >
                  {/* Actual token character, not the position number — direct
                      user request 2026-07-15 ("this letter A is character 1
                      ... it should have an A or B or C"). Position stays
                      available via hover. Internal indexing (key,
                      weights[i][j] lookups) stays 0-indexed. See
                      docs/DESIGN_DECISIONS.md. */}
                  {token}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {att.weights.map((row, i) => {
              const rowMax = Math.max(...row);
              const rowMin = Math.min(...row.filter((_, j) => j <= i)); // only real (unmasked) values
              return (
              <tr key={i}>
                <td
                  style={{
                    border: "1px solid var(--border)",
                    padding: 4,
                    textAlign: "right",
                    fontSize: 9,
                    color: "var(--text-dim)",
                  }}
                  title={`Position ${windowStart + i + 1}`}
                >
                  {tokenLabels[i]}
                </td>
                {row.map((weight, j) => {
                  const masked = j > i;
                  if (masked) {
                    return (
                      <td
                        key={j}
                        style={{
                          border: "1px solid var(--border)",
                          padding: 2,
                          textAlign: "center",
                          background: "repeating-linear-gradient(45deg, var(--bg), var(--bg) 3px, var(--border) 3px, var(--border) 4px)",
                          color: "var(--text-dim)",
                        }}
                        title="Masked — causal attention can't see future positions"
                      >
                        ·
                      </td>
                    );
                  }
                  const normalized = rowMax > rowMin ? (weight - rowMin) / (rowMax - rowMin) : 1;
                  const opacity = OPACITY_FLOOR + normalized * (1 - OPACITY_FLOOR);
                  const bgColor = `rgba(100, 150, 255, ${opacity})`;
                  return (
                    <td
                      key={j}
                      style={{
                        border: "1px solid var(--border)",
                        padding: 2,
                        textAlign: "center",
                        backgroundColor: bgColor,
                        color: opacity > 0.5 ? "white" : "var(--text)",
                      }}
                      title={weight.toFixed(3)}
                    >
                      {weight.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Runtime guard, not just the TS type: qkv_detail's shape changed
          2026-07-14 (single last-token vector -> per-position arrays) — a
          response from a trainer container built before that change would
          still have the old {position, q, k, v} shape and crash QKVTable's
          .positions.length access. See docs/DESIGN_DECISIONS.md. */}
      {att.qkv_detail && Array.isArray(att.qkv_detail.positions) && (
        <QKVTable qkv={att.qkv_detail} blockNum={blockNum} head={head} onOpenDataTab={onOpenDataTab} />
      )}
    </div>
  );
}

// Computed from the LM head's own logits vector (_compute_activation_extras
// (logits_last) in backend/training/diagnostics.py) — the exact same
// vector Top-k above is built from, just showing the raw values at the
// first 8 vocab positions (index order, not ranked) instead of the top-5
// by probability. Previously also showed a second "top 5" ranked by
// |logit| (magnitude, sign ignored) — a different, more confusing
// criterion than Top-k's probability ranking, pulling in strongly
// negative/unlikely logits alongside likely ones with no clear pedagogical
// payoff. Dropped rather than relabeled, since two overlapping top-5 lists
// with different orderings from the same vector was the actual confusion,
// not just the label. See docs/DESIGN_DECISIONS.md.
function LmHeadLogitsSlice({ snapshot }: { snapshot: DiagnosticSnapshot }) {
  const act = snapshot.activation_summaries;
  if (!act.available) {
    return (
      <div style={{ fontSize: 12, color: "var(--text-dim)", marginTop: 16 }}>
        {act.reason ? `Not captured: ${act.reason}` : "Not captured"}
      </div>
    );
  }
  if (!act.value_slice || act.value_slice.length === 0) return null;

  const shape = snapshot.lm_head.logits_shape;
  const vocabSize = shape.length > 0 ? shape[shape.length - 1] : undefined;

  return (
    <div style={{ marginTop: 16, paddingTop: 12, borderTop: "1px solid var(--border)" }}>
      <div style={{ fontSize: 11, color: "var(--text-dim)", marginBottom: 4 }}>
        LM Head Logits (first 8{vocabSize != null ? ` of ${vocabSize}` : ""}, raw — vocab index order, not ranked):
      </div>
      <div style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--text)" }}>
        [{act.value_slice.slice(0, 8).map((v) => v.toFixed(3)).join(", ")}]
      </div>
    </div>
  );
}

// The learned embedding table itself (vocab_size x n_embd), not per-position
// runtime activations for the current prompt — direct user request
// 2026-07-15: "for the embedding layer, I think we should also have the
// embedding table or embedding matrix." Fetched once per runId (static
// model weights, doesn't change per step). transformer/moe only — see the
// backend route's own docstring for why RNN has none. See
// docs/DESIGN_DECISIONS.md.
function EmbeddingTable({
  runId, onOpenDataTab,
}: {
  runId: number | null;
  onOpenDataTab: (title: string, content: number[]) => void;
}) {
  const [table, setTable] = useState<import("../types").EmbeddingTableData | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (runId == null) return;
    setTable(null);
    setError(false);
    fetchEmbeddingTable(runId)
      .then(setTable)
      .catch(() => setError(true));
  }, [runId]);

  if (error) {
    return <div style={{ fontSize: 12, color: "var(--text-dim)", marginTop: 12 }}>Embedding table not available (no checkpoint yet, or unsupported template).</div>;
  }
  if (!table) {
    return <div style={{ fontSize: 12, color: "var(--text-dim)", marginTop: 12 }}>Loading embedding table...</div>;
  }

  return (
    <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--border)" }}>
      <div style={{ fontSize: 11, color: "var(--accent)", marginBottom: 8 }}>
        Token Embedding Table — {table.vocab_size} tokens × {table.n_embd} dims (hover for full vector, double-click to open in a tab)
      </div>
      <div style={{ maxHeight: 400, overflowY: "auto" }}>
        <table style={{ borderCollapse: "collapse", fontSize: 10, fontFamily: "var(--font-mono)", width: "100%" }}>
          <thead>
            <tr>
              <th style={{ ...positionTableCellStyle, position: "sticky", top: 0, background: "var(--surface)" }}>Token</th>
              <th style={{ ...positionTableCellStyle, position: "sticky", top: 0, background: "var(--surface)" }}>Vector</th>
            </tr>
          </thead>
          <tbody>
            {table.tokens.map((tok, i) => {
              const v = truncatedVector(table.embedding[i]);
              return (
                <tr key={i}>
                  <td style={{ ...positionTableCellStyle, color: "var(--text)" }}>"{tok}"</td>
                  <td
                    style={{ ...positionTableCellStyle, color: "var(--text)", cursor: "pointer" }}
                    title={v.full}
                    onDoubleClick={() => onOpenDataTab(`Token Embedding Table — "${tok}"`, table.embedding[i])}
                  >
                    {v.preview}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Direct follow-up, 2026-07-13: "I can't see the position embedding
          table. I think they should both be on that tab." Only exists
          under pos_encoding="learned" — RoPE computes rotary embeddings on
          the fly, no table to show. See docs/DESIGN_DECISIONS.md. */}
      <div style={{ fontSize: 11, color: "var(--accent)", margin: "16px 0 8px" }}>
        {table.position_embedding
          ? `Position Embedding Table — ${table.block_size} positions × ${table.n_embd} dims (hover for full vector, double-click to open in a tab)`
          : "Position Embedding Table"}
      </div>
      {table.position_embedding ? (
        <div style={{ maxHeight: 400, overflowY: "auto" }}>
          <table style={{ borderCollapse: "collapse", fontSize: 10, fontFamily: "var(--font-mono)", width: "100%" }}>
            <thead>
              <tr>
                <th style={{ ...positionTableCellStyle, position: "sticky", top: 0, background: "var(--surface)" }}>Position</th>
                <th style={{ ...positionTableCellStyle, position: "sticky", top: 0, background: "var(--surface)" }}>Vector</th>
              </tr>
            </thead>
            <tbody>
              {table.position_embedding.map((vec, pos) => {
                const v = truncatedVector(vec);
                return (
                  <tr key={pos}>
                    <td style={{ ...positionTableCellStyle, color: "var(--text-dim)" }}>{pos + 1}</td>
                    <td
                      style={{ ...positionTableCellStyle, color: "var(--text)", cursor: "pointer" }}
                      title={v.full}
                      onDoubleClick={() => onOpenDataTab(`Position Embedding Table — pos ${pos + 1}`, vec)}
                    >
                      {v.preview}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div style={{ fontSize: 12, color: "var(--text-dim)" }}>
          Not applicable — this model uses RoPE (rotary position encoding), which has no learned position table.
        </div>
      )}
    </div>
  );
}

function Runtime({
  runId,
  snapshot,
  selectedNodeId,
  isLoading,
  attentionHead,
  onAttentionHeadChange,
  showQKVDetail,
  onShowQKVDetailChange,
  attentionWindowOffset,
  onAttentionWindowOffsetChange,
  nodeWindowOffset,
  onNodeWindowOffsetChange,
  numHeads,
  onOpenDataTab,
}: {
  runId: number | null;
  snapshot: DiagnosticSnapshot | null;
  selectedNodeId: string | null;
  isLoading: boolean;
  attentionHead: number | null;
  onAttentionHeadChange: (head: number | null) => void;
  showQKVDetail: boolean;
  onShowQKVDetailChange: (show: boolean) => void;
  attentionWindowOffset: number;
  onAttentionWindowOffsetChange: (offset: number) => void;
  nodeWindowOffset: number;
  onNodeWindowOffsetChange: (offset: number) => void;
  numHeads: number | null;
  onOpenDataTab: (title: string, content: number[]) => void;
}) {
  // Default Head to the first option as soon as an attention node is
  // selected and nothing's picked yet — a dropdown that starts blank,
  // showing nothing until manually chosen, was real reported friction
  // ("just frustrates the user," 2026-07-15). See docs/DESIGN_DECISIONS.md.
  useEffect(() => {
    if (selectedNodeId?.includes(".attention") && attentionHead === null && numHeads) {
      onAttentionHeadChange(0);
    }
  }, [selectedNodeId, attentionHead, numHeads, onAttentionHeadChange]);

  if (isLoading) {
    return (
      <div style={{ fontSize: 12, color: "var(--text-dim)" }}>
        <p>Loading diagnostic snapshot...</p>
      </div>
    );
  }

  if (!snapshot) {
    return (
      <p style={{ fontSize: 12, color: "var(--text-dim)" }}>
        No diagnostic step has been run yet — enter a prompt and click <strong>&gt;</strong> in
        Prompt Model below.
      </p>
    );
  }

  // Check if it's the lm_head
  if (selectedNodeId === "lm_head") {
    return (
      <div style={{ fontSize: 12 }}>
        <div style={{ marginBottom: 8 }}>
          <strong style={{ color: "var(--accent)" }}>Logits Shape:</strong>
          <div style={{ color: "var(--text-dim)" }}>{formatShape(snapshot.lm_head.logits_shape)}</div>
        </div>
        <div style={{ marginBottom: 8 }}>
          <strong style={{ color: "var(--accent)" }}>Top-k Tokens (k=5):</strong>
        </div>
        {/* > / >> sample with temperature (matching the Generate button),
            not greedy — the selected token is whichever id matches
            generated_token.id, not necessarily rank #1, and only at the
            most recent position. See docs/DESIGN_DECISIONS.md. */}
        <LmHeadStepper snapshot={snapshot} />
        <LmHeadLogitsSlice snapshot={snapshot} />
      </div>
    );
  }

  // Check if it's an attention node — show heatmap for Phase 2
  if (selectedNodeId?.includes(".attention")) {
    const notRequested = !snapshot.attention.available && snapshot.attention.reason === "Not requested";
    // Changing Head (or clicking a different block's node) only updates
    // local selection — it doesn't re-run a step. The heatmap below stays
    // frozen from whatever was actually captured on the last >/>> click, so
    // without this it silently looked like "changing Layer/Head does
    // nothing" (real bug report, 2026-07-14). Compare what's currently
    // selected against what the snapshot actually captured.
    const blockMatch = selectedNodeId.match(/^block\.(\d+)\.attention$/);
    const currentBlock = blockMatch ? parseInt(blockMatch[1], 10) : null;
    const stale =
      snapshot.attention.available &&
      (snapshot.attention.layer !== currentBlock || snapshot.attention.head !== attentionHead);
    return (
      <div style={{ fontSize: 12 }}>
        <strong style={{ color: "var(--accent)" }}>Attention Weights</strong>
        {/* Block is implied by selectedNodeId (this node) — only head needs
            picking here. Selecting a head doesn't retroactively affect the
            snapshot already captured; click > again in Prompt Model to
            capture attention for this head. See docs/DESIGN_DECISIONS.md. */}
        <div style={{ marginTop: 8, marginBottom: 8, display: "flex", alignItems: "center", gap: 8 }}>
          <label>
            Head:
            {/* Dropdown, not a number spinner — native spinner arrows were
                nearly illegible against the dark theme, and a head index is
                always a small bounded set (1..n_head), so a dropdown also
                makes an invalid value impossible to enter. Options and
                display are 1-indexed (matching the diagram's "Block N of M"
                convention); onAttentionHeadChange still receives the
                0-indexed value the backend expects. See
                docs/DESIGN_DECISIONS.md. */}
            <select
              value={attentionHead ?? 0}
              onChange={(e) => onAttentionHeadChange(parseInt(e.target.value, 10))}
              style={{ marginLeft: 4 }}
            >
              {Array.from({ length: numHeads ?? 0 }, (_, i) => (
                <option key={i} value={i}>{i + 1}</option>
              ))}
            </select>
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <input
              type="checkbox"
              checked={showQKVDetail}
              onChange={(e) => onShowQKVDetailChange(e.target.checked)}
              disabled={attentionHead === null}
              style={{ accentColor: "var(--accent)" }}
            />
            Show Q/K/V detail
          </label>
        </div>
        {notRequested && (
          <div style={{ marginBottom: 8, color: "var(--text-dim)" }}>
            Pick a head above, then click <strong>&gt;</strong> in Prompt Model to capture
            attention for this step.
          </div>
        )}
        {stale && (
          <div style={{
            marginBottom: 8, padding: "6px 8px", borderRadius: 4,
            background: "rgba(234, 179, 8, 0.12)", color: "#eab308", fontSize: 11,
          }}>
            Showing captured data for Block {(snapshot.attention.layer ?? 0) + 1}, Head {(snapshot.attention.head ?? 0) + 1}
            {" "}— currently selected: Block {currentBlock != null ? currentBlock + 1 : "?"}, Head {attentionHead != null ? attentionHead + 1 : "?"}.
            Click <strong>&gt;</strong> to refresh.
          </div>
        )}
        <AttentionHeatmap
          snapshot={snapshot}
          blockNum={currentBlock}
          head={attentionHead}
          onOpenDataTab={onOpenDataTab}
          windowOffset={attentionWindowOffset}
          onWindowOffsetChange={onAttentionWindowOffsetChange}
        />
      </div>
    );
  }

  // For other nodes, show shapes + summary stats + activation summaries
  const runtimeData = selectedNodeId ? snapshot.nodes[selectedNodeId] : null;

  if (!runtimeData) {
    return (
      <p style={{ fontSize: 12, color: "var(--text-dim)" }}>Not captured</p>
    );
  }

  return (
    <div style={{ fontSize: 12 }}>
      {runtimeData.input_shape && (
        <div style={{ marginBottom: 6 }}>
          <strong style={{ color: "var(--accent)" }}>Input:</strong>
          <div style={{ color: "var(--text-dim)" }}>{formatShape(runtimeData.input_shape)}</div>
        </div>
      )}
      {runtimeData.output_shape && (
        <div style={{ marginBottom: 6 }}>
          <strong style={{ color: "var(--accent)" }}>Output:</strong>
          <div style={{ color: "var(--text-dim)" }}>{formatShape(runtimeData.output_shape)}</div>
        </div>
      )}
      {runtimeData.summary && (
        <div>
          {/* Confirmed against the backend hook (register_diagnostic_hooks
              in each template's model.py): summary = _compute_summary(output)
              — always the OUTPUT tensor, never input. Made explicit here,
              direct user request 2026-07-14: "I think it's probably the
              output you're showing, but just to make it clearer... so
              people aren't having to guess." See docs/DESIGN_DECISIONS.md. */}
          <strong style={{ color: "var(--accent)" }}>Output Summary Stats:</strong>
          <div style={{ color: "var(--text-dim)", fontSize: 11, marginTop: 4 }}>
            <div>mean: {runtimeData.summary.mean.toFixed(4)}</div>
            <div>std: {runtimeData.summary.std.toFixed(4)}</div>
            <div>L2 norm: {runtimeData.summary.l2_norm.toFixed(2)}</div>
            <div>min: {runtimeData.summary.min.toFixed(4)}</div>
            <div>max: {runtimeData.summary.max.toFixed(4)}</div>
          </div>
        </div>
      )}
      {/* Direct user request, 2026-07-15: "a stepper that allows that
          window to slide backwards in time" for every node with a
          position_vectors view (LayerNorm, MLP, embedding, final_norm) —
          previously only the attention heatmap had this. Same shared
          sequence length as everywhere else in this snapshot
          (generated_token.position + 1) — one forward pass, one sequence
          length, applies to every node in it. See docs/DESIGN_DECISIONS.md. */}
      {runtimeData.position_vectors && (
        <NodeWindowStepper
          pv={runtimeData.position_vectors}
          totalPositions={(snapshot.generated_token?.position ?? runtimeData.position_vectors.positions[runtimeData.position_vectors.positions.length - 1]) + 1}
          windowOffset={nodeWindowOffset}
          onWindowOffsetChange={onNodeWindowOffsetChange}
        />
      )}
      {selectedNodeId === "embedding" ? (
        <>
          {/* Input (one-hot, synthesized) above Output (real, captured) —
              direct follow-up 2026-07-13: both wanted on this tab, not
              just one. See docs/DESIGN_DECISIONS.md. */}
          <EmbeddingOneHotTable snapshot={snapshot} onOpenDataTab={onOpenDataTab} />
          {runtimeData.position_vectors && (
            <NodeVectorTable
              outputPv={runtimeData.position_vectors}
              nodeId={selectedNodeId}
              onOpenDataTab={onOpenDataTab}
            />
          )}
        </>
      ) : (
        runtimeData.position_vectors && (
          <NodeVectorTable
            outputPv={runtimeData.position_vectors}
            inputPv={runtimeData.input_position_vectors}
            nodeId={selectedNodeId}
            onOpenDataTab={onOpenDataTab}
          />
        )
      )}
      {selectedNodeId === "embedding" && <EmbeddingTable runId={runId} onOpenDataTab={onOpenDataTab} />}
    </div>
  );
}

export default function Inspector({
  runId,
  selectedNode,
  selectedNodeId,
  diagnosticSnapshot,
  currentStep,
  isLoading,
  attentionHead,
  onAttentionHeadChange,
  showQKVDetail,
  onShowQKVDetailChange,
  attentionWindowOffset,
  onAttentionWindowOffsetChange,
  nodeWindowOffset,
  onNodeWindowOffsetChange,
  numHeads,
  onOpenDataTab,
  activeTab,
  onActiveTabChange,
}: Props) {
  if (!selectedNode) {
    return (
      <div className="panel">
        <h3>Inspector</h3>
        <p style={{ fontSize: 12, color: "var(--text-dim)" }}>Click a node in the architecture diagram to inspect it.</p>
      </div>
    );
  }

  return (
    <div className="panel">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <h3 style={{ margin: 0 }}>Inspector</h3>
        <span style={{ fontSize: 11, color: "var(--text-dim)" }}>
          {currentStep != null ? `Step ${currentStep}` : ""}
        </span>
      </div>

      <div style={{ marginBottom: 12, padding: "8px 0", borderBottom: "1px solid var(--border)" }}>
        <div style={{ fontSize: 12, color: "var(--text-dim)" }}>
          <strong style={{ color: "var(--text)" }}>{selectedNode.label}</strong>
          <div style={{ fontSize: 11, marginTop: 2 }}>kind: {selectedNode.kind}</div>
        </div>
      </div>

      {/* Sub-tabs */}
      <div style={tabStyle}>
        {(["overview", "shapes", "math", "config", "runtime"] as SubTab[]).map((tab) => (
          <button
            key={tab}
            onClick={() => onActiveTabChange(tab)}
            style={tabButtonStyle(activeTab === tab)}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div style={{ minHeight: 120, paddingBottom: 8 }}>
        {activeTab === "overview" && <Overview node={selectedNode} />}
        {activeTab === "shapes" && (
          <Shapes node={selectedNode} snapshot={diagnosticSnapshot} selectedNodeId={selectedNodeId} />
        )}
        {activeTab === "math" && <MathTab node={selectedNode} />}
        {activeTab === "config" && <Config node={selectedNode} />}
        {activeTab === "runtime" && (
          <Runtime
            runId={runId}
            snapshot={diagnosticSnapshot}
            selectedNodeId={selectedNodeId}
            isLoading={isLoading}
            attentionHead={attentionHead}
            onAttentionHeadChange={onAttentionHeadChange}
            showQKVDetail={showQKVDetail}
            onShowQKVDetailChange={onShowQKVDetailChange}
            attentionWindowOffset={attentionWindowOffset}
            onAttentionWindowOffsetChange={onAttentionWindowOffsetChange}
            nodeWindowOffset={nodeWindowOffset}
            onNodeWindowOffsetChange={onNodeWindowOffsetChange}
            numHeads={numHeads}
            onOpenDataTab={onOpenDataTab}
          />
        )}
      </div>
    </div>
  );
}
