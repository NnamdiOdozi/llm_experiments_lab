import { useState, useEffect, CSSProperties } from "react";
import { ArchitectureNode, DiagnosticSnapshot } from "../types";

interface Props {
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
  // config.model.n_head — bounds the Head dropdown so it only ever lists
  // real options, never a value the model doesn't have.
  numHeads: number | null;
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

type SubTab = "overview" | "shapes" | "math" | "config" | "runtime";

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

  // Highlighting "what actually got selected" only makes sense at the most
  // recent position (generated_token is the one token this whole snapshot
  // just sampled) — earlier positions' next tokens aren't reconstructed
  // here (would need combining input_tokens/history in a way the response
  // doesn't cleanly expose), kept simple deliberately.
  const isMostRecent = clampedIndex === entries.length - 1;
  // Real growing sequence length (prompt + everything generated so far) —
  // same value used to fix the token-count label in §36. entries.length is
  // only the WINDOW size (capped to DIAGNOSTIC_POSITION_WINDOW, 12) — for
  // a 15-token sequence, entries.length is 12, not 15. Displaying "12 of
  // 12" with no mention of the real 15 read as a second, contradicting
  // position counter right next to the already-correct "Position 14"
  // label — real bug report, 2026-07-14. See docs/DESIGN_DECISIONS.md.
  const totalPositions = (snapshot.generated_token?.position ?? entries.length - 1) + 1;
  const isWindowed = entries.length < totalPositions;

  return (
    <div style={{ fontSize: 11 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <button onClick={() => setIndex(Math.max(0, clampedIndex - 1))} disabled={clampedIndex === 0}>
          ◀
        </button>
        <span>
          Position {entry.position} ("{entry.token}") — {clampedIndex + 1} of {entries.length} shown
          {isWindowed && ` (last ${entries.length} of ${totalPositions} total)`}
        </span>
        <button onClick={() => setIndex(Math.min(entries.length - 1, clampedIndex + 1))} disabled={clampedIndex === entries.length - 1}>
          ▶
        </button>
      </div>
      {entry.top_k.map((tk) => {
        const isSelected = isMostRecent && tk.token_id === snapshot.generated_token?.id;
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
                <strong>#{tk.rank}</strong> {tk.token} (id={tk.token_id})
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
      {isMostRecent && !entry.top_k.some((tk) => tk.token_id === snapshot.generated_token?.id) && (
        <div style={{ marginTop: 6, color: "var(--text-dim)" }}>
          Selected token "{snapshot.generated_token?.text}" (id={snapshot.generated_token?.id}) fell outside the top 5 — temperature sampling can pick a lower-probability token.
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
function truncatedVector(v: number[], previewLen = 4): { preview: string; full: string } {
  return {
    preview: `[${v.slice(0, previewLen).map((x) => x.toFixed(2)).join(", ")}${v.length > previewLen ? ", …" : ""}]`,
    full: `[${v.map((x) => x.toFixed(4)).join(", ")}]`,
  };
}

function QKVTable({ qkv }: { qkv: import("../types").QKVDetail }) {
  return (
    <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--border)" }}>
      <div style={{ fontSize: 11, color: "var(--accent)", marginBottom: 8 }}>
        Q/K/V Detail — last {qkv.positions.length} position{qkv.positions.length === 1 ? "" : "s"} (hover a cell for full vector)
      </div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ borderCollapse: "collapse", fontSize: 10, fontFamily: "var(--font-mono)" }}>
          <thead>
            <tr>
              <th style={positionTableCellStyle}>Position</th>
              <th style={positionTableCellStyle}>Token</th>
              <th style={positionTableCellStyle}>Q</th>
              <th style={positionTableCellStyle}>K</th>
              <th style={positionTableCellStyle}>V</th>
            </tr>
          </thead>
          <tbody>
            {qkv.positions.map((pos, i) => {
              const q = truncatedVector(qkv.q[i]);
              const k = truncatedVector(qkv.k[i]);
              const v = truncatedVector(qkv.v[i]);
              return (
                <tr key={pos}>
                  <td style={{ ...positionTableCellStyle, color: "var(--text-dim)" }}>{pos}</td>
                  <td style={{ ...positionTableCellStyle, color: "var(--text)" }}>"{qkv.tokens[i]}"</td>
                  <td style={{ ...positionTableCellStyle, color: "var(--text)" }} title={q.full}>{q.preview}</td>
                  <td style={{ ...positionTableCellStyle, color: "var(--text)" }} title={k.full}>{k.preview}</td>
                  <td style={{ ...positionTableCellStyle, color: "var(--text)" }} title={v.full}>{v.preview}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// Same Colab-style table as QKVTable, generalized to any node's raw output
// vector (embedding, layernorm, mlp, etc.) — one column instead of three,
// no token text (position numbers alone are enough to correlate against
// the heatmap/top-k tables if needed, and re-decoding tokens at every one
// of ~18 nodes per step for a column that's already shown elsewhere isn't
// worth the payload). See docs/DESIGN_DECISIONS.md.
function NodeVectorTable({ pv }: { pv: import("../types").NodePositionVectors }) {
  return (
    <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--border)" }}>
      <div style={{ fontSize: 11, color: "var(--accent)", marginBottom: 8 }}>
        Output Vectors — last {pv.positions.length} position{pv.positions.length === 1 ? "" : "s"} (hover for full vector)
      </div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ borderCollapse: "collapse", fontSize: 10, fontFamily: "var(--font-mono)" }}>
          <thead>
            <tr>
              <th style={positionTableCellStyle}>Position</th>
              <th style={positionTableCellStyle}>Vector</th>
            </tr>
          </thead>
          <tbody>
            {pv.positions.map((pos, i) => {
              const v = truncatedVector(pv.vectors[i]);
              return (
                <tr key={pos}>
                  <td style={{ ...positionTableCellStyle, color: "var(--text-dim)" }}>{pos}</td>
                  <td style={{ ...positionTableCellStyle, color: "var(--text)" }} title={v.full}>{v.preview}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// Phase 2: Render attention heatmap for attention nodes
function AttentionHeatmap({ snapshot }: { snapshot: DiagnosticSnapshot }) {
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
                  title={token}
                >
                  {j}
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
                >
                  {i}
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
      {att.qkv_detail && Array.isArray(att.qkv_detail.positions) && <QKVTable qkv={att.qkv_detail} />}
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

function Runtime({
  snapshot,
  selectedNodeId,
  isLoading,
  attentionHead,
  onAttentionHeadChange,
  showQKVDetail,
  onShowQKVDetailChange,
  numHeads,
}: {
  snapshot: DiagnosticSnapshot | null;
  selectedNodeId: string | null;
  isLoading: boolean;
  attentionHead: number | null;
  onAttentionHeadChange: (head: number | null) => void;
  showQKVDetail: boolean;
  onShowQKVDetailChange: (show: boolean) => void;
  numHeads: number | null;
}) {
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
              value={attentionHead ?? ""}
              onChange={(e) => onAttentionHeadChange(e.target.value === "" ? null : parseInt(e.target.value, 10))}
              style={{ marginLeft: 4 }}
            >
              <option value="">—</option>
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
        <AttentionHeatmap snapshot={snapshot} />
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
          <strong style={{ color: "var(--accent)" }}>Summary Stats:</strong>
          <div style={{ color: "var(--text-dim)", fontSize: 11, marginTop: 4 }}>
            <div>mean: {runtimeData.summary.mean.toFixed(4)}</div>
            <div>std: {runtimeData.summary.std.toFixed(4)}</div>
            <div>L2 norm: {runtimeData.summary.l2_norm.toFixed(2)}</div>
            <div>min: {runtimeData.summary.min.toFixed(4)}</div>
            <div>max: {runtimeData.summary.max.toFixed(4)}</div>
          </div>
        </div>
      )}
      {runtimeData.position_vectors && <NodeVectorTable pv={runtimeData.position_vectors} />}
    </div>
  );
}

export default function Inspector({
  selectedNode,
  selectedNodeId,
  diagnosticSnapshot,
  currentStep,
  isLoading,
  attentionHead,
  onAttentionHeadChange,
  showQKVDetail,
  onShowQKVDetailChange,
  numHeads,
}: Props) {
  const [activeTab, setActiveTab] = useState<SubTab>("overview");

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
            onClick={() => setActiveTab(tab)}
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
            snapshot={diagnosticSnapshot}
            selectedNodeId={selectedNodeId}
            isLoading={isLoading}
            attentionHead={attentionHead}
            onAttentionHeadChange={onAttentionHeadChange}
            showQKVDetail={showQKVDetail}
            onShowQKVDetailChange={onShowQKVDetailChange}
            numHeads={numHeads}
          />
        )}
      </div>
    </div>
  );
}
