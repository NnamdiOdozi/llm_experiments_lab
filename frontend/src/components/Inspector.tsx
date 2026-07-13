import { useState, CSSProperties } from "react";
import { ArchitectureNode, DiagnosticSnapshot } from "../types";

interface Props {
  selectedNode: ArchitectureNode | null;
  selectedNodeId: string | null;
  diagnosticSnapshot: DiagnosticSnapshot | null;
  currentStep: number | null;
  isLoading: boolean;
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

  const maxWeight = Math.max(...att.weights.flat());
  const minWeight = Math.min(...att.weights.flat());

  return (
    <div>
      <div style={{ marginBottom: 8, fontSize: 11, color: "var(--accent)" }}>
        Layer {att.layer}, Head {att.head}
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
              <th style={{ border: "1px solid var(--border)", padding: 4, textAlign: "right", width: 40 }} />
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
            {att.weights.map((row, i) => (
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
                  // Normalize weight to 0-1 for color intensity
                  const normalized = (weight - minWeight) / (maxWeight - minWeight || 1);
                  const bgColor = `rgba(100, 150, 255, ${normalized})`;
                  return (
                    <td
                      key={j}
                      style={{
                        border: "1px solid var(--border)",
                        padding: 2,
                        textAlign: "center",
                        backgroundColor: bgColor,
                        color: normalized > 0.5 ? "white" : "var(--text)",
                      }}
                      title={weight.toFixed(3)}
                    >
                      {weight.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Phase 4: Q/K/V Detail */}
      {att.qkv_detail && (
        <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--border)" }}>
          <div style={{ fontSize: 11, color: "var(--accent)", marginBottom: 8 }}>
            Q/K/V Detail (Position {att.qkv_detail.position})
          </div>
          <div style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--text)" }}>
            <div style={{ marginBottom: 6 }}>
              <strong style={{ color: "var(--text-dim)" }}>Q:</strong>
              <div style={{ marginLeft: 8, color: "var(--text)" }}>
                [{att.qkv_detail.q.slice(0, 8).map((v) => v.toFixed(3)).join(", ")}...]
              </div>
            </div>
            <div style={{ marginBottom: 6 }}>
              <strong style={{ color: "var(--text-dim)" }}>K:</strong>
              <div style={{ marginLeft: 8, color: "var(--text)" }}>
                [{att.qkv_detail.k.slice(0, 8).map((v) => v.toFixed(3)).join(", ")}...]
              </div>
            </div>
            <div>
              <strong style={{ color: "var(--text-dim)" }}>V:</strong>
              <div style={{ marginLeft: 8, color: "var(--text)" }}>
                [{att.qkv_detail.v.slice(0, 8).map((v) => v.toFixed(3)).join(", ")}...]
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Phase 2: Render activation summaries. Always computed from the LM head's
// logits only (_compute_activation_extras(logits_last) in
// backend/training/diagnostics.py) — NOT per-node, despite the name. Only
// rendered under the lm_head node's Runtime view for that reason; showing it
// under every node (as before) implied it was specific to whatever node was
// selected, which it never was. See docs/DESIGN_DECISIONS.md.
function ActivationSummaries({ snapshot }: { snapshot: DiagnosticSnapshot }) {
  const act = snapshot.activation_summaries;
  if (!act.available) {
    return (
      <div style={{ fontSize: 12, color: "var(--text-dim)", marginTop: 16 }}>
        {act.reason ? `Not captured: ${act.reason}` : "Not captured"}
      </div>
    );
  }

  return (
    <div style={{ marginTop: 16, paddingTop: 12, borderTop: "1px solid var(--border)" }}>
      <strong style={{ color: "var(--accent)", fontSize: 11 }}>
        LM Head Logit Extras (top absolute components of the vocabulary logits)
      </strong>

      {act.top_abs_components && act.top_abs_components.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 11, color: "var(--text-dim)", marginBottom: 4 }}>
            Top Absolute Components:
          </div>
          {act.top_abs_components.map((comp, idx) => (
            <div key={idx} style={{ fontSize: 11, marginBottom: 2, color: "var(--text)" }}>
              <span style={{ color: "var(--text-dim)" }}>index {comp.index}:</span> {comp.value.toFixed(4)}
            </div>
          ))}
        </div>
      )}

      {act.value_slice && act.value_slice.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 11, color: "var(--text-dim)", marginBottom: 4 }}>
            Value Slice (first 8):
          </div>
          <div style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--text)" }}>
            [{act.value_slice.slice(0, 8).map((v) => v.toFixed(3)).join(", ")}]
          </div>
        </div>
      )}
    </div>
  );
}

function Runtime({
  snapshot,
  selectedNodeId,
  isLoading,
}: {
  snapshot: DiagnosticSnapshot | null;
  selectedNodeId: string | null;
  isLoading: boolean;
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
      <p style={{ fontSize: 12, color: "var(--text-dim)" }}>Not captured</p>
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
        <div style={{ fontSize: 11 }}>
          {snapshot.lm_head.top_k.map((entry) => (
            <div
              key={entry.rank}
              style={{
                display: "flex",
                justifyContent: "space-between",
                marginBottom: 4,
                padding: 4,
                background: entry.rank === 1 ? "rgba(76, 175, 80, 0.1)" : "transparent",
                borderRadius: 4,
              }}
            >
              <span>
                <strong>#{entry.rank}</strong> {entry.token} (id={entry.token_id})
              </span>
              <span style={{ color: "var(--text-dim)" }}>
                logit {entry.logit.toFixed(2)} prob {(entry.probability * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
        <ActivationSummaries snapshot={snapshot} />
      </div>
    );
  }

  // Check if it's an attention node — show heatmap for Phase 2
  if (selectedNodeId?.includes(".attention")) {
    return (
      <div style={{ fontSize: 12 }}>
        <strong style={{ color: "var(--accent)" }}>Attention Weights</strong>
        <div style={{ marginTop: 8 }}>
          <AttentionHeatmap snapshot={snapshot} />
        </div>
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
    </div>
  );
}

export default function Inspector({
  selectedNode,
  selectedNodeId,
  diagnosticSnapshot,
  currentStep,
  isLoading,
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
          <Runtime snapshot={diagnosticSnapshot} selectedNodeId={selectedNodeId} isLoading={isLoading} />
        )}
      </div>
    </div>
  );
}
